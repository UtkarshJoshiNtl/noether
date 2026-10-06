// CUDA kernels for the noether engine.
//
//   k_verlet_soa / k_rk4_soa / k_yoshida_soa — SoA integrator steps (the
//     force/integrator math is shared with the host via NOETHER_HD annotations)
//   k_verlet_f32 / k_verlet_f32k — fp32 state; the latter adds persistent
//     per-particle Kahan compensation on the velocity kick (E4)
//   k_energy        — per-particle Hamiltonian evaluation
//   k_partial       — block-shared fixed tree (plain or per-thread Kahan)
//   k_halve         — deterministic out-of-place tree pass (fixed reduction)
//   k_atomic_add    — nondeterministic finalize via atomicAdd
//
// Reduction modes: Atomic (block partials + atomicAdd finalize), Fixed
// (ordered halving), Kahan (per-thread strided Kahan + ordered halving),
// each with Fp64 or Fp32Acc accumulators.

#include <cuda_runtime.h>

#include <cmath>
#include <string>
#include <utility>

#include "core/cuda/api.hpp"
#include "core/cuda/device_mem.hpp"
#include "core/force.hpp"
#include "core/integrators.hpp"

namespace noether::cuda {
namespace {

constexpr int BLOCK = 256;
constexpr std::size_t MAX_PARTIALS = 16384;
constexpr int ATOMIC_MAX_BLOCKS = 1024;

int blocks_for(std::size_t n) {
    std::size_t g = (n + BLOCK - 1) / BLOCK;
    if (g < 1)
        g = 1;
    if (g > MAX_PARTIALS)
        g = MAX_PARTIALS;
    return static_cast<int>(g);
}

// ── Integrators ───────────────────────────────────────────────────────────────

__global__ void k_verlet_soa(double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
                             double* __restrict__ vx, double* __restrict__ vy,
                             double* __restrict__ vz, std::size_t n, double dt) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    State s{x[i], y[i], z[i], vx[i], vy[i], vz[i]};
    s = verlet_step(s, dt);
    x[i] = s.x;
    y[i] = s.y;
    z[i] = s.z;
    vx[i] = s.vx;
    vy[i] = s.vy;
    vz[i] = s.vz;
}

__global__ void k_rk4_soa(double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
                          double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
                          std::size_t n, double dt) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    State s{x[i], y[i], z[i], vx[i], vy[i], vz[i]};
    s = rk4_step(s, dt);
    x[i] = s.x;
    y[i] = s.y;
    z[i] = s.z;
    vx[i] = s.vx;
    vy[i] = s.vy;
    vz[i] = s.vz;
}

__global__ void k_yoshida_soa(double* __restrict__ x, double* __restrict__ y,
                              double* __restrict__ z, double* __restrict__ vx,
                              double* __restrict__ vy, double* __restrict__ vz, std::size_t n,
                              double w1, double w0) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    State s{x[i], y[i], z[i], vx[i], vy[i], vz[i]};
    s = verlet_step(s, w1);
    s = verlet_step(s, w0);
    s = verlet_step(s, w1);
    x[i] = s.x;
    y[i] = s.y;
    z[i] = s.z;
    vx[i] = s.vx;
    vy[i] = s.vy;
    vz[i] = s.vz;
}

__global__ void k_energy(const double* __restrict__ x, const double* __restrict__ y,
                         const double* __restrict__ z, const double* __restrict__ vx,
                         const double* __restrict__ vy, const double* __restrict__ vz,
                         double* __restrict__ e, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    e[i] = hamiltonian(x[i], y[i], z[i], vx[i], vy[i], vz[i]);
}

// ── fp32 mixed-precision integrators (E4) ────────────────────────────────────

// Plain fp32 velocity Verlet — same function the host would run for T=float.
__global__ void k_verlet_f32(float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
                             float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
                             std::size_t n, float dt) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    StateT<float> s{x[i], y[i], z[i], vx[i], vy[i], vz[i]};
    s = verlet_step(s, dt);
    x[i] = s.x;
    y[i] = s.y;
    z[i] = s.z;
    vx[i] = s.vx;
    vy[i] = s.vy;
    vz[i] = s.vz;
}

// Velocity Verlet with the persistent velocity kick accumulated via Kahan
// compensation: each particle carries cx,cy,cz across steps, so rounding bits
// dropped by the fp32 kick are re-injected on the next step. The drift
// (position update) stays plain fp32.
template <bool KAHAN>
__global__ void k_verlet_f32k(float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
                              float* __restrict__ vx, float* __restrict__ vy,
                              float* __restrict__ vz, float* __restrict__ cx,
                              float* __restrict__ cy, float* __restrict__ cz, std::size_t n,
                              float dt) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    Vec3T<float> a1 = gravity_accel(x[i], y[i], z[i]);

    float hx = x[i] + dt * (vx[i] + 0.5f * dt * a1.x);
    float hy = y[i] + dt * (vy[i] + 0.5f * dt * a1.y);
    float hz = z[i] + dt * (vz[i] + 0.5f * dt * a1.z);
    Vec3T<float> a2 = gravity_accel(hx, hy, hz);

    float ix = 0.5f * dt * (a1.x + a2.x);
    float iy = 0.5f * dt * (a1.y + a2.y);
    float iz = 0.5f * dt * (a1.z + a2.z);
    if constexpr (KAHAN) {
        float yx = ix - cx[i];
        float tx = vx[i] + yx;
        cx[i] = (tx - vx[i]) - yx;
        vx[i] = tx;
        float yy = iy - cy[i];
        float ty = vy[i] + yy;
        cy[i] = (ty - vy[i]) - yy;
        vy[i] = ty;
        float yz = iz - cz[i];
        float tz = vz[i] + yz;
        cz[i] = (tz - vz[i]) - yz;
        vz[i] = tz;
    } else {
        vx[i] = vx[i] + ix;
        vy[i] = vy[i] + iy;
        vz[i] = vz[i] + iz;
    }

    x[i] = hx;
    y[i] = hy;
    z[i] = hz;
}

// Energies evaluated in fp64 from the fp32 state.
__global__ void k_energy_f32(const float* __restrict__ x, const float* __restrict__ y,
                             const float* __restrict__ z, const float* __restrict__ vx,
                             const float* __restrict__ vy, const float* __restrict__ vz,
                             double* __restrict__ e, std::size_t n) {
    std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    e[i] = hamiltonian(static_cast<double>(x[i]), static_cast<double>(y[i]),
                       static_cast<double>(z[i]), static_cast<double>(vx[i]),
                       static_cast<double>(vy[i]), static_cast<double>(vz[i]));
}

// ── Reductions ────────────────────────────────────────────────────────────────

// Stage 1: each thread Kahan- (or plainly-) accumulates strided elements of
// its block's chunk; fixed shared-memory tree merge; one partial per block.
template <typename Acc, bool Kahan>
__global__ void k_partial(const double* __restrict__ in, long long n, long long chunk,
                          Acc* __restrict__ out) {
    long long begin = static_cast<long long>(blockIdx.x) * chunk;
    long long end = begin + chunk;
    if (end > n)
        end = n;

    __shared__ Acc s[BLOCK];
    const int tid = threadIdx.x;

    Acc acc = static_cast<Acc>(0);
    if constexpr (Kahan) {
        Acc c = static_cast<Acc>(0);
        for (long long i = begin + tid; i < end; i += blockDim.x) {
            Acc v = static_cast<Acc>(in[i]);
            Acc y = v - c;
            Acc t = acc + y;
            c = (t - acc) - y;
            acc = t;
        }
    } else {
        for (long long i = begin + tid; i < end; i += blockDim.x) {
            acc += static_cast<Acc>(in[i]);
        }
    }

    s[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            s[tid] = s[tid] + s[tid + stride];
        __syncthreads();
    }
    if (tid == 0)
        out[blockIdx.x] = s[0];
}

// Deterministic out-of-place halving: out[i] = in[2i] + in[2i+1].
template <typename Acc>
__global__ void k_halve(const Acc* __restrict__ in, Acc* __restrict__ out, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int out_n = (m + 1) >> 1;
    if (i >= out_n)
        return;
    Acc a = in[2 * i];
    Acc b = (2 * i + 1 < m) ? in[2 * i + 1] : static_cast<Acc>(0);
    out[i] = a + b;
}

// Nondeterministic finalize: contended atomicAdd over partials [begin, end).
template <typename Acc>
__global__ void k_atomic_add(const Acc* __restrict__ partials, int begin, int end, Acc* out) {
    long long i = begin + static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
    for (; i < end; i += stride)
        atomicAdd(out, partials[i]);
}

// ── Reduction driver ──────────────────────────────────────────────────────────

template <typename Acc>
struct Pool {
    DeviceMem<Acc> a{MAX_PARTIALS};
    DeviceMem<Acc> b{MAX_PARTIALS};
    DeviceMem<Acc> out{1};
};

template <typename Acc>
Pool<Acc>& pool() {
    static Pool<Acc> p;
    return p;
}

cudaStream_t stream_for(int idx) {
    static cudaStream_t streams[8] = {nullptr};
    if (streams[idx] == nullptr)
        cuda_check(cudaStreamCreate(&streams[idx]), "stream create");
    return streams[idx];
}

template <typename Acc, bool Kahan>
Acc run_fixed(const double* d_in, std::size_t n, int nblocks) {
    Pool<Acc>& p = pool<Acc>();
    long long chunk = (static_cast<long long>(n) + nblocks - 1) / nblocks;
    k_partial<Acc, Kahan><<<nblocks, BLOCK>>>(d_in, static_cast<long long>(n), chunk, p.a.data());

    int m = nblocks;
    Acc* src = p.a.data();
    Acc* dst = p.b.data();
    while (m > 1) {
        int out_n = (m + 1) >> 1;
        k_halve<<<blocks_for(static_cast<std::size_t>(out_n)), BLOCK>>>(src, dst, m);
        std::swap(src, dst);
        m = out_n;
    }

    Acc result = 0;
    cuda_check(cudaMemcpy(&result, src, sizeof(Acc), cudaMemcpyDeviceToHost), "result D2H");
    return result;
}

template <typename Acc>
Acc run_atomic(const double* d_in, std::size_t n, int nblocks, int streams) {
    Pool<Acc>& p = pool<Acc>();
    long long chunk = (static_cast<long long>(n) + nblocks - 1) / nblocks;
    k_partial<Acc, false><<<nblocks, BLOCK>>>(d_in, static_cast<long long>(n), chunk, p.a.data());
    p.out.zero();

    int m = nblocks;
    if (streams <= 1) {
        int g = m < ATOMIC_MAX_BLOCKS ? m : ATOMIC_MAX_BLOCKS;
        k_atomic_add<Acc><<<g, BLOCK>>>(p.a.data(), 0, m, p.out.data());
        cuda_check(cudaDeviceSynchronize(), "atomic sync");
    } else {
        // Chunked across independent streams: completion/atomic interleaving
        // order varies run to run — forces observable order variation (E2).
        for (int s = 0; s < streams; ++s) {
            int begin = static_cast<int>(static_cast<long long>(m) * s / streams);
            int end = static_cast<int>(static_cast<long long>(m) * (s + 1) / streams);
            if (begin >= end)
                continue;
            int span = end - begin;
            int g = span < ATOMIC_MAX_BLOCKS ? span : ATOMIC_MAX_BLOCKS;
            k_atomic_add<Acc><<<g, BLOCK, 0, stream_for(s)>>>(p.a.data(), begin, end, p.out.data());
        }
        cuda_check(cudaDeviceSynchronize(), "atomic streams sync");
    }

    Acc result = 0;
    cuda_check(cudaMemcpy(&result, p.out.data(), sizeof(Acc), cudaMemcpyDeviceToHost),
               "result D2H");
    return result;
}

}  // namespace

// ── Public API ────────────────────────────────────────────────────────────────

bool available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        cudaGetLastError();  // clear sticky error
        return false;
    }
    return count > 0;
}

const char* to_string(ReduceMode m) {
    switch (m) {
        case ReduceMode::Atomic:
            return "atomic";
        case ReduceMode::Fixed:
            return "fixed";
        case ReduceMode::Kahan:
            return "kahan";
    }
    return "?";
}

const char* to_string(ReducePrec p) {
    return p == ReducePrec::Fp64 ? "fp64" : "fp32";
}

bool parse_reduce_mode(const std::string& s, ReduceMode& out) {
    if (s == "atomic") {
        out = ReduceMode::Atomic;
        return true;
    }
    if (s == "fixed") {
        out = ReduceMode::Fixed;
        return true;
    }
    if (s == "kahan") {
        out = ReduceMode::Kahan;
        return true;
    }
    return false;
}

bool parse_reduce_prec(const std::string& s, ReducePrec& out) {
    if (s == "fp64") {
        out = ReducePrec::Fp64;
        return true;
    }
    if (s == "fp32" || s == "fp32acc") {
        out = ReducePrec::Fp32Acc;
        return true;
    }
    return false;
}

DeviceEnsemble::DeviceEnsemble(std::size_t n)
    : n_(n), x_(n), y_(n), z_(n), vx_(n), vy_(n), vz_(n), e_(n) {}

DeviceEnsemble::DeviceEnsemble(const std::vector<State>& h) : DeviceEnsemble(h.size()) {
    upload(h);
}

void DeviceEnsemble::upload(const std::vector<State>& h) {
    if (h.size() != n_)
        throw std::range_error("DeviceEnsemble::upload size mismatch");
    std::vector<double> buf(n_);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = h[i].x;
    x_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = h[i].y;
    y_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = h[i].z;
    z_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = h[i].vx;
    vx_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = h[i].vy;
    vy_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = h[i].vz;
    vz_.upload(buf);
}

void DeviceEnsemble::download(std::vector<State>& h) const {
    if (h.size() != n_)
        h.resize(n_);
    std::vector<double> bx = x_.download();
    std::vector<double> by = y_.download();
    std::vector<double> bz = z_.download();
    std::vector<double> bxv = vx_.download();
    std::vector<double> byv = vy_.download();
    std::vector<double> bzv = vz_.download();
    for (std::size_t i = 0; i < n_; ++i) {
        h[i] = State{bx[i], by[i], bz[i], bxv[i], byv[i], bzv[i]};
    }
}

void step_verlet(DeviceEnsemble& ens, double dt) {
    std::size_t n = ens.size();
    if (n == 0)
        return;
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    k_verlet_soa<<<g, BLOCK>>>(ens.x(), ens.y(), ens.z(), ens.vx(), ens.vy(), ens.vz(), n, dt);
    cuda_check(cudaGetLastError(), "verlet launch");
}

void step_rk4(DeviceEnsemble& ens, double dt) {
    std::size_t n = ens.size();
    if (n == 0)
        return;
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    k_rk4_soa<<<g, BLOCK>>>(ens.x(), ens.y(), ens.z(), ens.vx(), ens.vy(), ens.vz(), n, dt);
    cuda_check(cudaGetLastError(), "rk4 launch");
}

void step_yoshida(DeviceEnsemble& ens, double dt) {
    std::size_t n = ens.size();
    if (n == 0)
        return;
    const double c2 = std::cbrt(2.0);
    const double w1 = (1.0 / (2.0 - c2)) * dt;
    const double w0 = (-c2 / (2.0 - c2)) * dt;
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    k_yoshida_soa<<<g, BLOCK>>>(ens.x(), ens.y(), ens.z(), ens.vx(), ens.vy(), ens.vz(), n, w1, w0);
    cuda_check(cudaGetLastError(), "yoshida launch");
}

void energies(DeviceEnsemble& ens) {
    std::size_t n = ens.size();
    if (n == 0)
        return;
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    k_energy<<<g, BLOCK>>>(ens.x(), ens.y(), ens.z(), ens.vx(), ens.vy(), ens.vz(), ens.energy(),
                           n);
    cuda_check(cudaGetLastError(), "energy launch");
}

double reduce(const double* d_in, std::size_t n, ReduceMode mode, ReducePrec prec,
              int atomic_streams) {
    if (n == 0)
        return 0.0;
    if (atomic_streams < 1)
        atomic_streams = 1;
    if (atomic_streams > 8)
        atomic_streams = 8;
    int nblocks = blocks_for(n);

    if (prec == ReducePrec::Fp64) {
        double v = 0.0;
        switch (mode) {
            case ReduceMode::Fixed:
                v = run_fixed<double, false>(d_in, n, nblocks);
                break;
            case ReduceMode::Kahan:
                v = run_fixed<double, true>(d_in, n, nblocks);
                break;
            case ReduceMode::Atomic:
                v = run_atomic<double>(d_in, n, nblocks, atomic_streams);
                break;
        }
        cuda_check(cudaGetLastError(), "reduce kernel");
        return v;
    }

    float v = 0.0f;
    switch (mode) {
        case ReduceMode::Fixed:
            v = run_fixed<float, false>(d_in, n, nblocks);
            break;
        case ReduceMode::Kahan:
            v = run_fixed<float, true>(d_in, n, nblocks);
            break;
        case ReduceMode::Atomic:
            v = run_atomic<float>(d_in, n, nblocks, atomic_streams);
            break;
    }
    cuda_check(cudaGetLastError(), "reduce kernel");
    return static_cast<double>(v);
}

double energy_sum(DeviceEnsemble& ens, ReduceMode mode, ReducePrec prec, int atomic_streams) {
    energies(ens);
    return reduce(ens.energy(), ens.size(), mode, prec, atomic_streams);
}

// ── fp32 ensemble ────────────────────────────────────────────────────────────

F32Ensemble::F32Ensemble(std::size_t n)
    : n_(n), x_(n), y_(n), z_(n), vx_(n), vy_(n), vz_(n), cx_(n), cy_(n), cz_(n), e_(n) {}

F32Ensemble::F32Ensemble(const std::vector<State>& h) : F32Ensemble(h.size()) {
    upload(h);
}

void F32Ensemble::upload(const std::vector<State>& h) {
    if (h.size() != n_)
        throw std::range_error("F32Ensemble::upload size mismatch");
    std::vector<float> buf(n_);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = static_cast<float>(h[i].x);
    x_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = static_cast<float>(h[i].y);
    y_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = static_cast<float>(h[i].z);
    z_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = static_cast<float>(h[i].vx);
    vx_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = static_cast<float>(h[i].vy);
    vy_.upload(buf);
    for (std::size_t i = 0; i < n_; ++i)
        buf[i] = static_cast<float>(h[i].vz);
    vz_.upload(buf);
    cx_.zero();  // Kahan compensation starts clean on every (re)upload
    cy_.zero();
    cz_.zero();
}

void F32Ensemble::download(std::vector<State>& h) const {
    if (h.size() != n_)
        h.resize(n_);
    std::vector<float> bx = x_.download();
    std::vector<float> by = y_.download();
    std::vector<float> bz = z_.download();
    std::vector<float> bxv = vx_.download();
    std::vector<float> byv = vy_.download();
    std::vector<float> bzv = vz_.download();
    for (std::size_t i = 0; i < n_; ++i) {
        h[i] = State{static_cast<double>(bx[i]),  static_cast<double>(by[i]),
                     static_cast<double>(bz[i]),  static_cast<double>(bxv[i]),
                     static_cast<double>(byv[i]), static_cast<double>(bzv[i])};
    }
}

void step_verlet_f32(F32Ensemble& ens, double dt) {
    std::size_t n = ens.size();
    if (n == 0)
        return;
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    k_verlet_f32<<<g, BLOCK>>>(ens.x(), ens.y(), ens.z(), ens.vx(), ens.vy(), ens.vz(), n,
                               static_cast<float>(dt));
    cuda_check(cudaGetLastError(), "verlet_f32 launch");
}

void step_verlet_f32k(F32Ensemble& ens, double dt) {
    std::size_t n = ens.size();
    if (n == 0)
        return;
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    k_verlet_f32k<true><<<g, BLOCK>>>(ens.x(), ens.y(), ens.z(), ens.vx(), ens.vy(), ens.vz(),
                                      ens.cx(), ens.cy(), ens.cz(), n, static_cast<float>(dt));
    cuda_check(cudaGetLastError(), "verlet_f32k launch");
}

void energies(F32Ensemble& ens) {
    std::size_t n = ens.size();
    if (n == 0)
        return;
    int g = static_cast<int>((n + BLOCK - 1) / BLOCK);
    k_energy_f32<<<g, BLOCK>>>(ens.x(), ens.y(), ens.z(), ens.vx(), ens.vy(), ens.vz(),
                               ens.energy(), n);
    cuda_check(cudaGetLastError(), "energy_f32 launch");
}

double energy_sum(F32Ensemble& ens, ReduceMode mode, ReducePrec prec, int atomic_streams) {
    energies(ens);
    return reduce(ens.energy(), ens.size(), mode, prec, atomic_streams);
}

}  // namespace noether::cuda
