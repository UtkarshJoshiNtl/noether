#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace noether::cuda {

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cuda: ") + what + ": " + cudaGetErrorString(err));
    }
}

// RAII device buffer. Move-only; throws std::runtime_error on CUDA failures.
template <typename T>
class DeviceMem {
  public:
    DeviceMem() = default;
    explicit DeviceMem(std::size_t n) { allocate(n); }

    DeviceMem(const DeviceMem&) = delete;
    DeviceMem& operator=(const DeviceMem&) = delete;

    DeviceMem(DeviceMem&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
        o.ptr_ = nullptr;
        o.size_ = 0;
    }
    DeviceMem& operator=(DeviceMem&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = o.ptr_;
            size_ = o.size_;
            o.ptr_ = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    ~DeviceMem() { release(); }

    void allocate(std::size_t n) {
        release();
        if (n > 0)
            cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr_), n * sizeof(T)), "malloc");
        size_ = n;
    }

    void upload(const T* src, std::size_t n, std::size_t dst_off = 0) {
        if (dst_off + n > size_)
            throw std::range_error("DeviceMem::upload out of range");
        cuda_check(cudaMemcpy(ptr_ + dst_off, src, n * sizeof(T), cudaMemcpyHostToDevice), "H2D");
    }

    void upload(const std::vector<T>& v) { upload(v.data(), v.size(), 0); }

    void download(T* dst, std::size_t n, std::size_t src_off = 0) const {
        if (src_off + n > size_)
            throw std::range_error("DeviceMem::download out of range");
        cuda_check(cudaMemcpy(dst, ptr_ + src_off, n * sizeof(T), cudaMemcpyDeviceToHost), "D2H");
    }

    std::vector<T> download() const {
        std::vector<T> v(size_);
        if (size_ > 0)
            download(v.data(), size_, 0);
        return v;
    }

    void zero() {
        if (size_ > 0)
            cuda_check(cudaMemset(ptr_, 0, size_ * sizeof(T)), "memset");
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }

  private:
    void release() {
        if (ptr_)
            cudaFree(ptr_);  // best-effort in destructor
        ptr_ = nullptr;
        size_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

}  // namespace noether::cuda
