#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "core/cuda/api.hpp"
#include "core/cuda/device_mem.hpp"
#include "core/ensemble.hpp"
#include "core/force.hpp"
#include "core/integrators.hpp"
#include "core/reductions.hpp"

// CUDA reduction correctness + bitwise repeatability, and GPU integrator
// sanity. Returns 77 (CTest SKIP_RETURN_CODE) when no CUDA device is present.

using namespace noether;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

std::vector<double> lcg_input(std::size_t n, std::uint64_t seed = 42) {
    std::vector<double> v(n);
    std::uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
    }
    return v;
}

void test_roundtrip() {
    std::vector<State> h = make_ensemble(64, 400.0, 51.6);
    cuda::DeviceEnsemble ens(h);
    std::vector<State> back;
    ens.download(back);
    bool exact = back.size() == h.size();
    for (std::size_t i = 0; exact && i < h.size(); ++i)
        exact = std::bit_cast<std::uint64_t>(back[i].x) == std::bit_cast<std::uint64_t>(h[i].x) &&
                std::bit_cast<std::uint64_t>(back[i].vz) == std::bit_cast<std::uint64_t>(h[i].vz);
    check(exact, "DeviceEnsemble upload/download bit-exact");
}

void test_reduction_accuracy() {
    const std::size_t n = 1 << 18;
    std::vector<double> in = lcg_input(n);
    cuda::DeviceMem<double> d(n);
    d.upload(in);

    const long double oracle = sum_longdouble(in.data(), n);
    const double ref = static_cast<double>(oracle);

    struct Case {
        cuda::ReduceMode mode;
        cuda::ReducePrec prec;
        double tol;
        const char* name;
    };
    // Calibrated on RTX 2050; tolerances are relative to |oracle|.
    const Case cases[] = {
        {cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp64, 1e-15, "fixed/fp64"},
        {cuda::ReduceMode::Kahan, cuda::ReducePrec::Fp64, 1e-15, "kahan/fp64"},
        {cuda::ReduceMode::Atomic, cuda::ReducePrec::Fp64, 1e-15, "atomic/fp64"},
        {cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp32Acc, 1e-6, "fixed/fp32"},
        {cuda::ReduceMode::Kahan, cuda::ReducePrec::Fp32Acc, 1e-6, "kahan/fp32"},
        // Atomic fp32 error includes order variation; bound stays well below
        // the fp32-representable noise floor of ~1e-4.
        {cuda::ReduceMode::Atomic, cuda::ReducePrec::Fp32Acc, 1e-5, "atomic/fp32"},
    };
    for (const Case& c : cases) {
        double v = cuda::reduce(d.data(), n, c.mode, c.prec, 1);
        double rel = std::fabs(v - ref) / std::fabs(ref);
        std::printf("%-12s rel_err = %.3e (tol %.1e)\n", c.name, rel, c.tol);
        check(rel <= c.tol, c.name);
    }
}

void test_bitwise_stability() {
    const std::size_t n = 1 << 18;
    const int repeats = 100;
    std::vector<double> in = lcg_input(n);
    cuda::DeviceMem<double> d(n);
    d.upload(in);

    struct Case {
        cuda::ReduceMode mode;
        cuda::ReducePrec prec;
        const char* name;
    };
    const Case cases[] = {
        {cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp64, "fixed/fp64"},
        {cuda::ReduceMode::Kahan, cuda::ReducePrec::Fp64, "kahan/fp64"},
        {cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp32Acc, "fixed/fp32"},
        {cuda::ReduceMode::Kahan, cuda::ReducePrec::Fp32Acc, "kahan/fp32"},
    };
    for (const Case& c : cases) {
        std::uint64_t first = 0;
        bool stable = true;
        for (int r = 0; r < repeats; ++r) {
            double v = cuda::reduce(d.data(), n, c.mode, c.prec, 1);
            std::uint64_t b = std::bit_cast<std::uint64_t>(v);
            if (r == 0)
                first = b;
            else if (b != first)
                stable = false;
        }
        std::printf("%-12s stable over %d repeats: %s\n", c.name, repeats, stable ? "yes" : "NO");
        check(stable, c.name);
    }
}

// Records atomic repeat-spread — the nondeterminism evidence for the paper.
// Single-stream may appear stable on a fixed launch config; multi-stream
// chunked accumulation is the fallback if spread ever fails to appear.
void test_atomic_spread() {
    const std::size_t n = 1 << 18;
    const int repeats = 100;
    std::vector<double> in = lcg_input(n);
    cuda::DeviceMem<double> d(n);
    d.upload(in);

    for (int streams : {1, 8}) {
        std::vector<std::uint64_t> bits;
        double vmin = 1e300, vmax = -1e300;
        for (int r = 0; r < repeats; ++r) {
            double v = cuda::reduce(d.data(), n, cuda::ReduceMode::Atomic, cuda::ReducePrec::Fp64,
                                    streams);
            bits.push_back(std::bit_cast<std::uint64_t>(v));
            vmin = std::min(vmin, v);
            vmax = std::max(vmax, v);
        }
        std::sort(bits.begin(), bits.end());
        bits.erase(std::unique(bits.begin(), bits.end()), bits.end());
        std::printf("atomic/fp64 streams=%d: unique=%zu/%d spread=%.3e\n", streams, bits.size(),
                    repeats, vmax - vmin);
        // Values must stay accurate regardless of order.
        check(vmax - vmin < 1e-9, "atomic spread bounded");
    }
}

void test_unity_exact() {
    // Sum of 1.0 is an exact integer in fp64 trees (< 2^53) — strong check.
    const std::size_t n = 1 << 16;
    std::vector<double> in(n, 1.0);
    cuda::DeviceMem<double> d(n);
    d.upload(in);
    double v = cuda::reduce(d.data(), n, cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp64, 1);
    check(v == static_cast<double>(n), "fixed/fp64 unity sum exact");
    double k = cuda::reduce(d.data(), n, cuda::ReduceMode::Kahan, cuda::ReducePrec::Fp64, 1);
    check(k == static_cast<double>(n), "kahan/fp64 unity sum exact");
}

void test_kernels_vs_host() {
    const std::size_t n = 128;
    std::vector<State> h = make_ensemble(static_cast<int>(n), 400.0, 51.6);
    std::vector<State> host = h;
    cuda::DeviceEnsemble ens(h);

    const double dt = 10.0;
    const int steps = 100;
    for (int s = 0; s < steps; ++s) {
        cuda::step_verlet(ens, dt);
        for (State& st : host)
            st = verlet_step(st, dt);
    }
    std::vector<State> back;
    ens.download(back);

    double max_rel = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double dr = std::fabs(back[i].x - host[i].x) + std::fabs(back[i].y - host[i].y) +
                    std::fabs(back[i].z - host[i].z);
        double scale = std::fabs(host[i].x) + std::fabs(host[i].y) + std::fabs(host[i].z);
        max_rel = std::max(max_rel, dr / scale);
    }
    // GPU may fuse to FMA (fmad=true); bit-exactness is NOT claimed across
    // backends — trajectories may differ at rounding level only.
    std::printf("gpu vs cpu verlet after %d steps: max_rel = %.3e\n", steps, max_rel);
    check(max_rel < 1e-12, "GPU/CPU verlet trajectory agreement");
}

void test_gpu_energy_bounded() {
    // Mirrors test_integrators on the GPU: 7 days, dt=10 s, bounded dH/H0.
    std::vector<State> h{circular_state(400.0, 0.0, 0.0, 0.0)};
    cuda::DeviceEnsemble ens(h);
    const double dt = 10.0;
    const int steps = 7 * 86400 / static_cast<int>(dt);

    auto mode = cuda::ReduceMode::Fixed;
    auto prec = cuda::ReducePrec::Fp64;
    double h0 = cuda::energy_sum(ens, mode, prec, 1);
    double max_dh = 0.0;
    for (int i = 0; i < steps; ++i) {
        cuda::step_verlet(ens, dt);
        if (i % 600 == 0 || i == steps - 1) {
            double h = cuda::energy_sum(ens, mode, prec, 1);
            max_dh = std::max(max_dh, std::fabs((h - h0) / std::fabs(h0)));
        }
    }
    std::printf("gpu verlet 7d dt=10: max_dh_rel = %.3e\n", max_dh);
    check(max_dh <= 1e-6, "GPU 7-day energy boundedness");
}

// ── fp32 mixed-precision (E4) ────────────────────────────────────────────────

void test_f32_step_vs_host() {
    // GPU k_verlet_f32 must track the host T=float integrator (FMA contraction
    // may prevent bit-exactness; tolerance ~ fp32 accuracy over 100 steps).
    const std::size_t n = 128;
    std::vector<State> h = make_ensemble(static_cast<int>(n), 400.0, 51.6);
    cuda::F32Ensemble ens(h);

    std::vector<StateT<float>> host(n);
    for (std::size_t i = 0; i < n; ++i)
        host[i] = StateT<float>{static_cast<float>(h[i].x),  static_cast<float>(h[i].y),
                                static_cast<float>(h[i].z),  static_cast<float>(h[i].vx),
                                static_cast<float>(h[i].vy), static_cast<float>(h[i].vz)};

    const float dt = 10.0f;
    const int steps = 100;
    for (int s = 0; s < steps; ++s) {
        cuda::step_verlet_f32(ens, 10.0);
        for (StateT<float>& st : host)
            st = verlet_step(st, dt);
    }
    std::vector<State> back;
    ens.download(back);

    double max_rel = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double dr = std::fabs(back[i].x - static_cast<double>(host[i].x)) +
                    std::fabs(back[i].y - static_cast<double>(host[i].y)) +
                    std::fabs(back[i].z - static_cast<double>(host[i].z));
        double scale = std::fabs(back[i].x) + std::fabs(back[i].y) + std::fabs(back[i].z);
        max_rel = std::max(max_rel, dr / scale);
    }
    std::printf("gpu fp32 vs host fp32 verlet after %d steps: max_rel = %.3e\n", steps, max_rel);
    check(max_rel < 1e-5, "GPU/host fp32 verlet agreement");
}

void test_f32_energy_consistency() {
    // F32Ensemble energies (fp64 from fp32 state, fixed tree) must match the
    // host long-double total over the float-rounded states.
    std::vector<State> h = make_ensemble(64, 400.0, 51.6);
    std::vector<State> rounded = h;
    for (State& s : rounded) {
        s.x = static_cast<double>(static_cast<float>(s.x));
        s.y = static_cast<double>(static_cast<float>(s.y));
        s.z = static_cast<double>(static_cast<float>(s.z));
        s.vx = static_cast<double>(static_cast<float>(s.vx));
        s.vy = static_cast<double>(static_cast<float>(s.vy));
        s.vz = static_cast<double>(static_cast<float>(s.vz));
    }
    double ref = total_energy(rounded);

    cuda::F32Ensemble ens(h);
    double v = cuda::energy_sum(ens, cuda::ReduceMode::Fixed, cuda::ReducePrec::Fp64, 1);
    double rel = std::fabs(v - ref) / std::fabs(ref);
    std::printf("f32 energy vs host: rel_err = %.3e\n", rel);
    check(rel < 1e-14, "F32Ensemble energy consistency");
}

double f32_max_dh(bool kahan_kick) {
    std::vector<State> h{circular_state(400.0, 0.0, 0.0, 0.0)};
    cuda::F32Ensemble ens(h);
    const double dt = 10.0;
    const int steps = 7 * 86400 / static_cast<int>(dt);

    auto mode = cuda::ReduceMode::Fixed;
    auto prec = cuda::ReducePrec::Fp64;
    double h0 = cuda::energy_sum(ens, mode, prec, 1);
    double max_dh = 0.0;
    for (int i = 0; i < steps; ++i) {
        if (kahan_kick)
            cuda::step_verlet_f32k(ens, dt);
        else
            cuda::step_verlet_f32(ens, dt);
        if (i % 600 == 0 || i == steps - 1) {
            double h = cuda::energy_sum(ens, mode, prec, 1);
            max_dh = std::max(max_dh, std::fabs((h - h0) / std::fabs(h0)));
        }
    }
    return max_dh;
}

void test_f32_energy_bounded() {
    // E4 accuracy axis: both fp32 variants stay bounded over 7 days, and the
    // Kahan-compensated kick must beat the plain fp32 kick (measured 4.2x on
    // RTX 2050 at dt=10 s; generous margin).
    double dh_plain = f32_max_dh(false);
    double dh_kahan = f32_max_dh(true);
    std::printf("gpu verlet 7d dt=10: fp32 max_dh_rel = %.3e, fp32k = %.3e\n", dh_plain, dh_kahan);
    check(dh_plain <= 1e-4, "fp32 7-day energy boundedness");
    check(dh_kahan <= 1e-4, "fp32k 7-day energy boundedness");
    check(dh_kahan <= dh_plain, "fp32k kick improves on plain fp32");
}

}  // namespace

int main() {
    if (!cuda::available()) {
        std::printf("test_cuda: no CUDA device — skipped\n");
        return 77;
    }
    test_roundtrip();
    test_reduction_accuracy();
    test_bitwise_stability();
    test_atomic_spread();
    test_unity_exact();
    test_kernels_vs_host();
    test_gpu_energy_bounded();
    test_f32_step_vs_host();
    test_f32_energy_consistency();
    test_f32_energy_bounded();
    if (failures) {
        std::printf("test_cuda: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_cuda: OK\n");
    return 0;
}
