#ifndef CUDA_MATRIX_H_
#define CUDA_MATRIX_H_

#include "../../../../../../../usr/local/cuda/include/driver_types.h"
#include "cuda_memory.h"
#include "cuda_types.h"
#include <typeinfo>

namespace cuda {
namespace matrix {
template <typename T> class CudaMatrix {
public:
  CudaMatrix();

  CudaMatrix(size_t num_rows, size_t num_cols);

  CudaMatrix(const CudaMatrix<T> &cuda_matrix);

  CudaMatrix(cv::Mat opencv_matrix);

  ~CudaMatrix();

  CudaMatrix<T> &operator=(const cv::Mat &opencv_matrix);

  CudaMatrix<T> &operator=(const CudaMatrix<T> &cuda_matrix);

  CudaMatrix<T> DeepCopy() const;

  T GetValue(size_t row, size_t col);

  void SetSize(size_t num_rows, size_t num_cols);

  void SetValue(cv::Mat opencv_matrix);

  void Clear();

  cv::Mat GetMatrix();

  DEVICE_ONLY_CALL
  void StoreElement(T element, size_t row, size_t col);

  HOST_DEVICE_CALL

  size_t NumRows() const { return num_rows_; };

  HOST_DEVICE_CALL

  size_t NumCols() const { return num_cols_; };

  HOST_DEVICE_CALL

  size_t GetPitch() const { return pitch_; };

  HOST_DEVICE_CALL

  size_t GetStride() const { return stride_; };

  HOST_DEVICE_CALL

  T *data() { return matrix_; }

  HOST_DEVICE_CALL

  const T *data() const { return matrix_; }

  T operator()(float row, float col) const;

private:
  void CreateTexture();

  T *matrix_;
  size_t num_rows_;
  size_t num_cols_;
  size_t pitch_;
  size_t stride_;

  cudaTextureObject_t texture_;
  cudaTextureAddressMode texture_address_mode_;
  cudaTextureFilterMode texture_filter_mode_;
};

//------------------------------------------------------------------------------
template <typename T>
CudaMatrix<T>::CudaMatrix()
    : num_rows_(0), num_cols_(0), pitch_(0), stride_(0), matrix_(NULL),
      texture_(0) {}

//------------------------------------------------------------------------------
template <typename T>
CudaMatrix<T>::CudaMatrix(size_t num_rows, size_t num_cols)
    : num_rows_(0), num_cols_(0), pitch_(0), stride_(0), matrix_(NULL),
      texture_(0) {
  SetSize(num_rows, num_cols);
}

//------------------------------------------------------------------------------
template <typename T>
CudaMatrix<T>::CudaMatrix(const CudaMatrix<T> &cuda_matrix)
    : num_rows_(cuda_matrix.num_rows_), num_cols_(cuda_matrix.num_cols_),
      pitch_(cuda_matrix.pitch_), stride_(cuda_matrix.stride_),
      texture_(cuda_matrix.texture_), matrix_(cuda_matrix.matrix_),
      texture_address_mode_(cuda_matrix.texture_address_mode_),
      texture_filter_mode_(cuda_matrix.texture_filter_mode_) {
  if (matrix_) {
    cuda::memory::AddAddressConsumer(matrix_);
    cuda::memory::AddTextureConsumer(texture_);
  }
}

//------------------------------------------------------------------------------
template <typename T>
CudaMatrix<T>::CudaMatrix(cv::Mat open_cv_matrix)
    : num_rows_(0), num_cols_(0), pitch_(0), stride_(0), matrix_(NULL),
      texture_(0) {
  *this = open_cv_matrix;
}

//------------------------------------------------------------------------------
template <typename T> CudaMatrix<T>::~CudaMatrix() {
  cuda::memory::FreeMemory(matrix_);
  cuda::memory::RemoveTexture(texture_);
}

//------------------------------------------------------------------------------
template <typename T> void CudaMatrix<T>::Clear() {
  int opencv_type = OpenCVType<T>();
  cv::Mat open_cv_matrix(num_rows_, num_cols_, opencv_type, cv::Scalar::all(0));
  *this = open_cv_matrix;
}

//------------------------------------------------------------------------------
template <typename T>
CudaMatrix<T> &CudaMatrix<T>::operator=(const cv::Mat &opencv_matrix) {
  SetSize(opencv_matrix.rows, opencv_matrix.cols);
  cuda::memory::Copy2D<T>(matrix_, pitch_, opencv_matrix.data,
                          opencv_matrix.step, num_cols_, num_rows_);
  return *this;
}

//------------------------------------------------------------------------------
template <typename T> CudaMatrix<T> CudaMatrix<T>::DeepCopy() const {
  CudaMatrix<T> temp_matrix(num_rows_, num_cols_);
  cuda::memory::Copy2D<T>(temp_matrix.matrix_, temp_matrix.pitch_, matrix_,
                          pitch_, num_cols_, num_rows_);

  ErrorCheckCuda(cudaPeekAtLastError());
  ErrorCheckCuda(cudaDeviceSynchronize());

  temp_matrix.texture_address_mode_ = texture_address_mode_;
  temp_matrix.texture_filter_mode_ = texture_filter_mode_;
  temp_matrix.CreateTexture();

  return temp_matrix;
}

//------------------------------------------------------------------------------
template <typename T>
CudaMatrix<T> &CudaMatrix<T>::operator=(const CudaMatrix<T> &cuda_matrix) {
  cuda::memory::FreeMemory(matrix_);
  cuda::memory::RemoveTexture(texture_);

  num_rows_ = cuda_matrix.num_rows_;
  num_cols_ = cuda_matrix.num_cols_;
  pitch_ = cuda_matrix.pitch_;
  stride_ = cuda_matrix.stride_;
  matrix_ = cuda_matrix.matrix_;
  texture_ = cuda_matrix.texture_;
  texture_address_mode_ = cuda_matrix.texture_address_mode_;
  texture_filter_mode_ = cuda_matrix.texture_filter_mode_;

  cuda::memory::AddAddressConsumer(matrix_);
  cuda::memory::AddTextureConsumer(texture_);

  return *this;
}

//------------------------------------------------------------------------------
template <typename T> cv::Mat CudaMatrix<T>::GetMatrix() {
  int opencv_type = OpenCVType<T>();
  cv::Mat open_cv_matrix(num_rows_, num_cols_, opencv_type);
  cuda::memory::Copy2D<T>(open_cv_matrix.data, open_cv_matrix.step, matrix_,
                          pitch_, num_cols_, num_rows_);

  return open_cv_matrix;
}

//------------------------------------------------------------------------------
template <typename T>
void CudaMatrix<T>::SetSize(size_t num_rows, size_t num_cols) {
  if ((num_rows != num_rows_) || (num_cols != num_cols_)) {
    cuda::memory::FreeMemory(matrix_);
    num_rows_ = num_rows;
    num_cols_ = num_cols;
    cuda::memory::Malloc2D<T>(&matrix_, pitch_, num_cols_, num_rows_);
    stride_ = pitch_ / sizeof(T);
    if (texture_ == 0) {
      texture_address_mode_ = cudaAddressModeClamp;
      texture_filter_mode_ = cudaFilterModeLinear;
      CreateTexture();
    } else {
      CreateTexture();
    }
  }
}

//------------------------------------------------------------------------------
template <typename T> void CudaMatrix<T>::SetValue(cv::Mat opencv_matrix) {
  *this = opencv_matrix;
}

//------------------------------------------------------------------------------
template <typename T> T CudaMatrix<T>::GetValue(size_t row, size_t col) {
  T value = 0;
  if (row < num_rows_ && col < num_cols_) {
    T *source = &matrix_[row * stride_ + col];
    cuda::memory::Copy(&value, source, sizeof(T));
  }
  return value;
}

//------------------------------------------------------------------------------
template <typename T>
DEVICE_ONLY_CALL void CudaMatrix<T>::StoreElement(T element, size_t row,
                                                  size_t col) {
  matrix_[row * stride_ + col] = element;
}

//------------------------------------------------------------------------------
template <typename T>
DEVICE_ONLY_CALL T

CudaMatrix<T>::operator()(float row, float col) const {
  T value;
  tex2D(&value, texture_, col + 0.5f, row + 0.5f);
  return value;
  // return matrix_[row * stride_ + col];
}

//------------------------------------------------------------------------------
template <typename T> void CudaMatrix<T>::CreateTexture() {

  cuda::memory::RemoveTexture(texture_);
  cudaChannelFormatDesc channel_descr = cudaCreateChannelDesc<T>();
  struct cudaResourceDesc resource_descriptor;
  memset(&resource_descriptor, 0, sizeof(resource_descriptor));
  resource_descriptor.resType = cudaResourceTypePitch2D;
  resource_descriptor.res.pitch2D.devPtr = matrix_;
  resource_descriptor.res.pitch2D.desc = channel_descr;
  resource_descriptor.res.pitch2D.width = num_cols_;
  resource_descriptor.res.pitch2D.height = num_rows_;
  resource_descriptor.res.pitch2D.pitchInBytes = pitch_;

  struct cudaTextureDesc texture_descriptor;
  memset(&texture_descriptor, 0, sizeof(texture_descriptor));
  texture_descriptor.addressMode[0] = texture_address_mode_;
  texture_descriptor.addressMode[1] = texture_address_mode_;
  texture_descriptor.filterMode = texture_filter_mode_;

  cuda::memory::CreateTexture(texture_, resource_descriptor,
                              texture_descriptor);
}

} // namespace matrix
} // namespace cuda
#endif // CUDA_MATRIX_H_