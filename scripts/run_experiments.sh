#!/usr/bin/env bash
# Regenerates every raw CSV under paper/results/ from the noether engine.
#
#   E1  integrator comparison   drift, CPU, dt sweep {10,20,40}, 24 h + 7 d
#   E2  reduction repeatability reduce, modes x n=2^10..2^24 x 100 repeats,
#                               inputs lcg (benign) + adversarial (cancellation)
#   E3  diagnostic noise floor  7 d ensemble drift through each reduction mode
#   E4  precision Pareto        drift accuracy (fp64/fp32/fp32k, 7 d dt sweep)
#                               + throughput (bench, n=2^20)
#
# CPU paths are used for E1 (bit-reproducible; GPU/CPU equivalence is enforced
# by tests/test_cuda.cpp). Runtime: ~90 s on the dev machine with a GPU.

set -euo pipefail
cd "$(dirname "$0")/.."

BIN=${BIN:-./build/noether}
OUT=${OUT:-paper/results}
mkdir -p "$OUT"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

if [[ ! -x "$BIN" ]]; then
    echo "error: $BIN not found — build first:" >&2
    echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
    exit 1
fi

# ── E1: RK4 vs Verlet vs Yoshida, dH/H0 vs t ─────────────────────────────────
log "E1: integrator dt sweep (rk4/verlet/yoshida4, 24 h + 7 d, CPU)"
for dt in 10 20 40; do
    for t in 86400 604800; do
        dur=7d
        [[ $t == 86400 ]] && dur=24h
        for ig in rk4 verlet yoshida4; do
            "$BIN" drift --backend cpu --integrator "$ig" --dt "$dt" --t "$t" \
                --sample 600 --out "$OUT/e1_${ig}_dt${dt}_${dur}.csv" >/dev/null
        done
    done
done
log "E1: 18 files"

# ── E2: bitwise repeatability + spread of every reduction mode ───────────────
log "E2: reduce repeatability (n=2^10..2^24, lcg + adversarial, R=100)"
tmp_e2=$(mktemp)
trap 'rm -f "$tmp_e2"' EXIT
first=1
for input in lcg adversarial; do
    for e in 10 12 14 16 18 20 22 24; do
        n=$((1 << e))
        "$BIN" reduce --n "$n" --repeats 100 --input "$input" --out "$tmp_e2" >/dev/null
        if [[ $first -eq 1 ]]; then
            cat "$tmp_e2" >"$OUT/e2_reduce.csv"
            first=0
        else
            tail -n +2 "$tmp_e2" >>"$OUT/e2_reduce.csv"
        fi
    done
done
log "E2: $(($(wc -l <"$OUT/e2_reduce.csv") - 1)) rows"

# ── E3: reduction-mode noise floor vs symplectic oscillation (7 d, n=4096) ──
log "E3: 7-day ensemble drift through each diagnostic reduction mode"
"$BIN" drift --backend cpu --n 4096 --t 604800 --dt 10 --sample 300 \
    --out "$OUT/e3_cpu.csv" >/dev/null
for mode in atomic fixed kahan; do
    for prec in fp64 fp32; do
        "$BIN" drift --backend gpu --n 4096 --t 604800 --dt 10 --sample 300 \
            --red-mode "$mode" --red-prec "$prec" \
            --out "$OUT/e3_${mode}_${prec}.csv" >/dev/null
    done
done
log "E3: 7 files"

# ── E4: precision Pareto — accuracy + throughput ─────────────────────────────
log "E4: fp64/fp32/fp32k accuracy (7 d, dt sweep, single satellite)"
for prec in fp64 fp32 fp32k; do
    for dt in 5 10 20 40; do
        "$BIN" drift --backend gpu --precision "$prec" --t 604800 --dt "$dt" \
            --sample 600 --out "$OUT/e4_${prec}_dt${dt}.csv" >/dev/null
    done
done
log "E4: throughput (n=2^20, best-of)"
"$BIN" bench --n 1048576 --steps 200 --repeats 5 --cpu \
    --out "$OUT/e4_bench.csv" >/dev/null

log "done -> $OUT"
ls -la "$OUT"
