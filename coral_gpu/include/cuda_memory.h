#ifndef CUDA_MEMORY_H_
#define CUDA_MEMORY_H_

#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/cuda_runtime_api.h"

#include <glog/logging.h>
#include <cstdint.h>

#define DEVICE_ONLY_CALL __device__
#define HOST_DEVICE_CALL __host__ __device__

#define ErrorCheckCuda(error_code)                                             \
  { cuda::memory::ErrorCheck(error_code, __FILE__, __LINE__); }


#ifdef __CUDA_ARCH__
#define DEVICE_CODE
#else
#define HOST_CODE
#endif
namespace cuda {
namespace memory {

void AddAddressConsumer(const void *address);

void AddTextureConsumer(cudaTextureObject_t texture);

void FreeMemory(void *address);

void CreateTexture(cudaTextureObject_t &texture,
                   cudaResourceDesc &resource_descriptor,
                   cudaTextureDesc &texture_descriptor);

void RemoveTexture(cudaTextureObject_t texture);
void ErrorCheck(cudaError_t code, const char *file, int32_t line,
                bool abort = true);

void Copy(void *destination, const void *source, size_t num_bytes);

template <typename T>
void Malloc2D(T **address, size_t &pitch, size_t width, size_t height);
//------------------------------------------------------------------------------
template <typename T>
void Copy2D(void *destination, size_t pitch_destination, const void *source,
            size_t pitch_source, size_t width, size_t height);
//------------------------------------------------------------------------------
template <typename T>
void Malloc2D(T **address, size_t &pitch, size_t width, size_t height) {
  size_t BYTES_PER_ROW = sizeof(T) * width;
  ErrorCheckCuda(cudaMallocPitch(address, &pitch, BYTES_PER_ROW, height));
  AddAddressConsumer(*address);
}
template <typename T>
void Copy2D(void *destination, size_t pitch_destination, const void *source,
            size_t pitch_source, size_t width, size_t height) {
  size_t BYTES_PER_ROW = sizeof(T) * width;
  ErrorCheckCuda(cudaMemcpy2D(destination, pitch_destination, source,
                              pitch_source, BYTES_PER_ROW, height,
                              cudaMemcpyDefault));
}

} // namespace memory
} // namespace cuda
#endif // CUDA_MATRIX_H_