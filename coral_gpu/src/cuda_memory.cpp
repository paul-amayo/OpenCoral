#include "../include/cuda_memory.h"
#include <boost/thread/mutex.hpp>
#include <iostream>
#include <map>
//------------------------------------------------------------------------------
boost::mutex num_address_consumers_mutex;
std::map<const void *, int> num_address_consumers;
//------------------------------------------------------------------------------
boost::mutex num_texture_consumers_mutex;
std::map<const cudaTextureObject_t, int> num_texture_consumers;
//------------------------------------------------------------------------------
void RemoveConsumerForAddress(void *address) {
  int &count = num_address_consumers[address];
  count--;
}
//------------------------------------------------------------------------------
void RemoveConsumerForTexture(cudaTextureObject_t texture) {
  int &count = num_texture_consumers[texture];
  count--;
}
//------------------------------------------------------------------------------
int NumTextureConsumers(cudaTextureObject_t texture) {
  return num_texture_consumers[texture];
}
//------------------------------------------------------------------------------
int NumAddressConsumers(void *address) {
  return num_address_consumers[address];
}
namespace cuda {
namespace memory {

void AddAddressConsumer(const void *address) {
  boost::mutex::scoped_lock lock(num_address_consumers_mutex);
  if (num_address_consumers.count(address) == 0) {
    num_address_consumers[address] = 1;
  } else {
    int &count = num_address_consumers[address];
    count++;
  }
}
//------------------------------------------------------------------------------
void AddTextureConsumer(cudaTextureObject_t texture) {
  boost::mutex::scoped_lock lock(num_texture_consumers_mutex);
  if (texture == 0) {
    return;
  }
  if (num_texture_consumers.count(texture) == 0) {
    num_texture_consumers[texture] = 1;
  } else {
    int &count = num_texture_consumers[texture];
    count++;
  }
}
//------------------------------------------------------------------------------
void CreateTexture(cudaTextureObject_t &texture,
                   cudaResourceDesc &resource_descriptor,
                   cudaTextureDesc &texture_descriptor) {
  ErrorCheckCuda(cudaCreateTextureObject(&texture, &resource_descriptor,
                                         &texture_descriptor, nullptr));
  AddTextureConsumer(texture);
}
//------------------------------------------------------------------------------
void RemoveTexture(cudaTextureObject_t texture) {
  boost::mutex::scoped_lock lock(num_texture_consumers_mutex);
  if (texture == 0) {
    return;
  }
  RemoveConsumerForTexture(texture);
  if (NumTextureConsumers(texture) == 0) {
    cudaDestroyTextureObject(texture);
  }
}
//------------------------------------------------------------------------------
void ErrorCheck(cudaError_t code, const char *file, int32_t line, bool abort) {
  if (code != cudaSuccess) {

    std::cerr << "Cuda Error :" << cudaGetErrorString(code) << "\n";
    if (abort) {
      exit(code);
    }
  }
}
//------------------------------------------------------------------------------
void Copy(void *destination, const void *source, size_t num_bytes) {
  ErrorCheckCuda(cudaMemcpy(destination, source, num_bytes, cudaMemcpyDefault));
}
//------------------------------------------------------------------------------
void FreeMemory(void *memory) {
  boost::mutex::scoped_lock lock(num_address_consumers_mutex);
  RemoveConsumerForAddress(memory);

  if (NumAddressConsumers(memory) == 0) {
    ErrorCheckCuda(cudaFree(memory));
  }
}

} // namespace memory
} // namespace cuda