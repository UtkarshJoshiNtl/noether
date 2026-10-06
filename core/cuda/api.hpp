#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/cuda/device_mem.hpp"
#include "core/force.hpp"

// Host-callable CUDA API. Included by main.cpp and tests only when built with
// NOETHER_USE_CUDA; all entry points are implemented in kernels.cu.

namespace noether::cuda {

// True when at least one CUDA device is present at runtime.
bool available();

enum class ReduceMode : int { Atomic = 0, Fixed = 1, Kahan = 2 };
enum class ReducePrec : int { Fp64 = 0, Fp32Acc = 1 };

const char* to_string(ReduceMode m);
const char* to_string(ReducePrec p);
bool parse_reduce_mode(const std::string& s, ReduceMode& out);
bool parse_reduce_prec(const std::string& s, ReducePrec& out);

// Device-resident SoA ensemble.
class DeviceEnsemble {
  public:
    explicit DeviceEnsemble(std::size_t n);
    explicit DeviceEnsemble(const std::vector<State>& h);

    void upload(const std::vector<State>& h);
    void download(std::vector<State>& h) const;

    std::size_t size() const { return n_; }

    double* x() { return x_.data(); }
    double* y() { return y_.data(); }
    double* z() { return z_.data(); }
    double* vx() { return vx_.data(); }
    double* vy() { return vy_.data(); }
    double* vz() { return vz_.data(); }
    double* energy() { return e_.data(); }

    const double* x() const { return x_.data(); }
    const double* y() const { return y_.data(); }
    const double* z() const { return z_.data(); }
    const double* vx() const { return vx_.data(); }
    const double* vy() const { return vy_.data(); }
    const double* vz() const { return vz_.data(); }

  private:
    std::size_t n_ = 0;
    DeviceMem<double> x_, y_, z_, vx_, vy_, vz_, e_;
};

// One integration step over the whole ensemble (asynchronous on the null stream).
void step_verlet(DeviceEnsemble& ens, double dt);
void step_rk4(DeviceEnsemble& ens, double dt);
void step_yoshida(DeviceEnsemble& ens, double dt);

// Per-particle Hamiltonians into the ensemble's energy buffer.
void energies(DeviceEnsemble& ens);

// fp32 mixed-precision ensemble (E4): state stored as float, force math in
// float, optional persistent per-particle Kahan compensation on the velocity
// kick (step_verlet_f32k). Energies are evaluated in fp64 from the fp32 state,
// so diagnostics measure trajectory error, not diagnostic noise.
class F32Ensemble {
  public:
    explicit F32Ensemble(std::size_t n);
    explicit F32Ensemble(const std::vector<State>& h);

    void upload(const std::vector<State>& h);    // casts to float; zeroes kick comps
    void download(std::vector<State>& h) const;  // casts back to double

    std::size_t size() const { return n_; }

    float* x() { return x_.data(); }
    float* y() { return y_.data(); }
    float* z() { return z_.data(); }
    float* vx() { return vx_.data(); }
    float* vy() { return vy_.data(); }
    float* vz() { return vz_.data(); }
    float* cx() { return cx_.data(); }
    float* cy() { return cy_.data(); }
    float* cz() { return cz_.data(); }
    double* energy() { return e_.data(); }

  private:
    std::size_t n_ = 0;
    DeviceMem<float> x_, y_, z_, vx_, vy_, vz_;
    DeviceMem<float> cx_, cy_, cz_;  // Kahan kick compensation, persistent
    DeviceMem<double> e_;
};

// One fp32 step over the whole ensemble (asynchronous on the null stream).
void step_verlet_f32(F32Ensemble& ens, double dt);   // plain fp32 kick
void step_verlet_f32k(F32Ensemble& ens, double dt);  // Kahan-compensated kick

void energies(F32Ensemble& ens);
double energy_sum(F32Ensemble& ens, ReduceMode mode, ReducePrec prec, int atomic_streams = 1);

// Reduction over an arbitrary device array. Synchronizes; returns the host value.
double reduce(const double* d_in, std::size_t n, ReduceMode mode, ReducePrec prec,
              int atomic_streams = 1);

// k_energy + reduce in one call.
double energy_sum(DeviceEnsemble& ens, ReduceMode mode, ReducePrec prec, int atomic_streams = 1);

}  // namespace noether::cuda
