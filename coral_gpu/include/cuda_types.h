#ifndef CUDA_CV_TYPES_H_
#define CUDA_CV_TYPES_H_

#include "/usr/local/cuda/include/cuda_runtime.h"
#include "/usr/local/cuda/include/cuda_runtime_api.h"
#include "cuda_memory.h"

#include <stdint.h>
#include <opencv2/opencv.hpp>

//------------------------------------------------------------------------------
template <typename T> int32_t OpenCVType();
//------------------------------------------------------------------------------
template <> inline int32_t OpenCVType<float>() { return CV_32F; }
//------------------------------------------------------------------------------
template <> inline int32_t OpenCVType<float2>() { return CV_32FC2; }
//------------------------------------------------------------------------------
template <> inline int32_t OpenCVType<float4>() { return CV_32FC4; }
// TODO: Fix Cuda includes for cudamath.h functions

//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL void operator+=(float2 &a, float2 b) {
  a.x += b.x;
  a.y += b.y;
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float2 operator+(float2 a, float b) {
  return make_float2(a.x + b, a.y + b);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float2 operator+(float b, float2 a) {
  return make_float2(a.x + b, a.y + b);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL void operator+=(float2 &a, float b) {
  a.x += b;
  a.y += b;
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float4 operator+(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL void operator+=(float4 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float4 operator+(float4 a, float b) {
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float4 operator+(float b, float4 a) {
  return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL void operator+=(float4 &a, float b) {
  a.x += b;
  a.y += b;
  a.z += b;
  a.w += b;
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL void operator-=(float2 &a, float2 b) {
  a.x -= b.x;
  a.y -= b.y;
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float2 operator-(float2 a, float b) {
  return make_float2(a.x - b, a.y - b);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL float2 operator-(float b, float2 a) {
  return make_float2(b - a.x, b - a.y);
}
//------------------------------------------------------------------------------
inline HOST_DEVICE_CALL void operator-=(float2 &a, float b) {
  a.x -= b;
  a.y -= b;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator-(float4 a, float4 b) {
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator-=(float4 &a, float4 b) {
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  a.w -= b.w;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator-(float4 a, float b) {
  return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator-=(float4 &a, float b) {
  a.x -= b;
  a.y -= b;
  a.z -= b;
  a.w -= b;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float2 operator*(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator*=(float2 &a, float2 b) {
  a.x *= b.x;
  a.y *= b.y;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float2 operator*(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}
//------------------------------------------------------------------------------
inline __host__ __device__ float2 operator*(float b, float2 a) {
  return make_float2(b * a.x, b * a.y);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator*=(float2 &a, float b) {
  a.x *= b;
  a.y *= b;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator*(float4 a, float4 b) {
  return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator*=(float4 &a, float4 b) {
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  a.w *= b.w;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator*(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator*(float b, float4 a) {
  return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator*=(float4 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float2 operator/(float2 a, float2 b) {
  return make_float2(a.x / b.x, a.y / b.y);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator/=(float2 &a, float2 b) {
  a.x /= b.x;
  a.y /= b.y;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float2 operator/(float2 a, float b) {
  return make_float2(a.x / b, a.y / b);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator/=(float2 &a, float b) {
  a.x /= b;
  a.y /= b;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float2 operator/(float b, float2 a) {
  return make_float2(b / a.x, b / a.y);
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator/(float4 a, float4 b) {
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator/=(float4 &a, float4 b) {
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  a.w /= b.w;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator/(float4 a, float b) {
  return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
//------------------------------------------------------------------------------
inline __host__ __device__ void operator/=(float4 &a, float b) {
  a.x /= b;
  a.y /= b;
  a.z /= b;
  a.w /= b;
}
//------------------------------------------------------------------------------
inline __host__ __device__ float4 operator/(float b, float4 a) {
  return make_float4(b / a.x, b / a.y, b / a.z, b / a.w);
}
//------------------------------------------------------------------------------

#endif // CUDA_MATRIX_H_