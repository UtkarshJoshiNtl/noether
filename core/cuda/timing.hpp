#pragma once

#include <cuda_runtime.h>

#include "device_mem.hpp"

namespace noether::cuda {

// cudaEvent pair for GPU-side timing (kernel time, excluding host overhead).
class GpuTimer {
  public:
    GpuTimer() {
        cuda_check(cudaEventCreate(&start_), "event create");
        cuda_check(cudaEventCreate(&stop_), "event create");
    }

    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() { cuda_check(cudaEventRecord(start_), "event record"); }

    // Records the stop event and returns elapsed milliseconds (synchronizes).
    float stop_ms() {
        cuda_check(cudaEventRecord(stop_), "event record");
        cuda_check(cudaEventSynchronize(stop_), "event sync");
        float ms = 0.0f;
        cuda_check(cudaEventElapsedTime(&ms, start_, stop_), "event elapsed");
        return ms;
    }

  private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

}  // namespace noether::cuda
