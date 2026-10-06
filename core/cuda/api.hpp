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

// Reduction over an arbitrary device array. Synchronizes; returns the host value.
double reduce(const double* d_in, std::size_t n, ReduceMode mode, ReducePrec prec,
              int atomic_streams = 1);

// k_energy + reduce in one call.
double energy_sum(DeviceEnsemble& ens, ReduceMode mode, ReducePrec prec, int atomic_streams = 1);

}  // namespace noether::cuda
