// TODO: Fix relative includes for CUDA header files


#include "../../../../../../../usr/local/cuda/include/curand_mtgp32_kernel.h"
#include "../../../../../../../usr/local/cuda/include/device_launch_parameters.h"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "../include/cuda_optimiser.h"

namespace cuda {
namespace optimiser {

#define BLOCKSIZEX 16
#define BLOCKSIZEY 16

#define MAXNLABELS 20
//------------------------------------------------------------------------------
__global__ void
update_primal(cuda::matrix::CudaMatrix<float> primal,
              cuda::matrix::CudaMatrix<float> primal_relaxed,
              cuda::matrix::CudaMatrix<float> smoothness_dual,
              cuda::matrix::CudaMatrix<float> compactness_dual,
              cuda::matrix::CudaMatrix<float> model_costs,
              cuda::matrix::CudaMatrix<float> inverse_neighbour_index,
              int num_labels, int width, int num_features, double tau,
              double lambda, double beta, int max_neighbours) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  int curr_feature = col + row * width;

  if (curr_feature < num_features && col < width && row < width) {

    double curr_feature_primal[MAXNLABELS];

    for (int curr_label = 0; curr_label < num_labels; ++curr_label) {
      float smoothness_dual_update =
          smoothness_dual(curr_label, curr_feature) +
          smoothness_dual(curr_label, curr_feature + num_features);

      // Add the contribution of neighbours
      for (int curr_neighbour = 0; curr_neighbour < max_neighbours;
           ++curr_neighbour) {
        if (inverse_neighbour_index(curr_neighbour, curr_feature) >= 0) {
          smoothness_dual_update -= smoothness_dual(
              curr_label,
              inverse_neighbour_index(curr_neighbour, curr_feature));
        }
      }

      __syncthreads();
      // Perform the update step
      curr_feature_primal[curr_label] =
          primal(curr_label, curr_feature) +
          tau * (lambda * smoothness_dual_update -
                 model_costs(curr_label, curr_feature) -
                 beta * compactness_dual(curr_label, curr_feature));

   }

    // Project to simplex

    float simplex_sum = 0;
    int num_non_zero = 0;
    float simplex_var;
    bool projection_complete = false;
    while (!projection_complete) {
      simplex_sum = 0;
      num_non_zero = 0;
      projection_complete = true;
      for (int curr_label = 0; curr_label < num_labels; ++curr_label) {
        simplex_var = curr_feature_primal[curr_label];
        if (simplex_var != 0) {
          num_non_zero++;
          simplex_sum += simplex_var;
        }
      }
      if (num_non_zero) {
        simplex_var = (simplex_sum - 1) / num_non_zero;
        for (int curr_label = 0; curr_label < num_labels; ++curr_label) {
          if (curr_feature_primal[curr_label] != 0) {
            curr_feature_primal[curr_label] -= simplex_var;
            if (curr_feature_primal[curr_label] < 0) {
              projection_complete = false;
              curr_feature_primal[curr_label] = 0;
            }
          }
        }
      } else {
        curr_feature_primal[0] = 1;
      }
    }

    // std::printf("Simplex projected \n");

    // Do the relaxation
    for (int curr_label = 0; curr_label < num_labels; ++curr_label) {
      primal_relaxed.StoreElement(2 * curr_feature_primal[curr_label] -
                                      primal(curr_label, curr_feature),
                                  curr_label, curr_feature);
      primal.StoreElement(curr_feature_primal[curr_label], curr_label,
                          curr_feature);
    }
  }
}
//------------------------------------------------------------------------------
__global__ void
update_compactness_dual(cuda::matrix::CudaMatrix<float> compactness_dual,
                        cuda::matrix::CudaMatrix<float> primal_relaxed,
                        int num_labels, int width, int num_features, double nu,
                        double beta) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  int curr_feature = col + row * width;

  if (curr_feature < num_features && col < width && row < width) {
    for (int curr_label = 0; curr_label < num_labels; ++curr_label) {
      float update = compactness_dual(curr_label, curr_feature) +
                     nu * beta * primal_relaxed(curr_label, curr_feature);

      compactness_dual.StoreElement(update, curr_label, curr_feature);
    }
  }
}
//------------------------------------------------------------------------------
__global__ void
update_smoothness_dual(cuda::matrix::CudaMatrix<float> smoothness_dual,
                       cuda::matrix::CudaMatrix<float> primal_relaxed,
                       cuda::matrix::CudaMatrix<float> nabla, int num_labels,
                       int width, int num_features, double alpha,
                       double lambda) {
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  int curr_feature = col + row * width;
  // TODO: Expand to N Neighbours
  if (curr_feature < num_features && col < width && row < width) {
    for (int curr_label = 0; curr_label < num_labels; ++curr_label) {
      float2 smoothness_dual_update;
      smoothness_dual_update.x =
          smoothness_dual((float)curr_label, (float)curr_feature) +
          alpha * lambda *
              (primal_relaxed(
                  curr_label,
                  nabla(0, (float)curr_feature) -
                      primal_relaxed((float)curr_label, (float)curr_feature)));
      smoothness_dual_update.y =
          smoothness_dual((float)curr_label, (float)curr_feature) +
          alpha * lambda *
              (primal_relaxed(
                  (float)curr_label,
                  nabla(1, (float)(curr_feature + num_features)) -
                      primal_relaxed((float)curr_label, (float)curr_feature)));

      float clipped_dual = fmaxf(
          1.0f, sqrtf(smoothness_dual_update.x * smoothness_dual_update.x +
                      smoothness_dual_update.y * smoothness_dual_update.y));

      smoothness_dual.StoreElement(smoothness_dual_update.x / clipped_dual,
                                   curr_label, curr_feature);
      smoothness_dual.StoreElement(smoothness_dual_update.y / clipped_dual,
                                   curr_label, curr_feature);
    }
  }
}
//------------------------------------------------------------------------------
CudaOptimiser::CudaOptimiser(const cv::Mat &model_costs, cv::Mat nabla,
                             const cv::Mat &inverse_neighbour_index_,
                             CudaOptimiserParams params) {
  params_ = params;

  //  // Create and Set the Matrices;

  primal_.SetSize(params_.num_labels, params_.num_features);
  primal_.Clear();

  primal_relaxed_.SetSize(params_.num_labels, params_.num_features);
  primal_relaxed_.Clear();

  smoothness_dual_.SetSize(params_.num_labels,
                           params_.num_features * params_.num_neighbours);
  smoothness_dual_.Clear();

  compactness_dual_.SetSize(params_.num_labels, params_.num_features);
  compactness_dual_.Clear();

  nabla_.SetSize(nabla.rows, nabla.cols);
  nabla_.SetValue(nabla);

  inverse_neighbour_index__.SetSize(inverse_neighbour_index_.rows,
                                    inverse_neighbour_index_.cols);
  inverse_neighbour_index__.SetValue(inverse_neighbour_index_);

  model_costs_.SetSize(params_.num_labels, params_.num_features);
  model_costs_.SetValue(model_costs);
}
//------------------------------------------------------------------------------
cuda::matrix::CudaMatrix<float> CudaOptimiser::Optimise() {
  for (int iter = 0; iter < params_.num_iterations; ++iter) {
    CompactnessDualOptimisation();
    SmoothnessDualOptimisation();
    PrimalOptimisation();
  }

  return primal_;
}
//------------------------------------------------------------------------------
void CudaOptimiser::CompactnessDualOptimisation() {
  int square_sides = sqrt(params_.num_features) + 1;
  dim3 num_threads(N_THREADS_BLOCK, N_THREADS_BLOCK);
  dim3 num_blocks(BlocksPerSide(square_sides, N_THREADS_BLOCK),
                  BlocksPerSide(square_sides, N_THREADS_BLOCK));

  update_compactness_dual<<<num_blocks, num_threads>>>(
      compactness_dual_, primal_relaxed_, params_.num_labels, square_sides,
      params_.num_features, params_.nu, params_.beta);
  ErrorCheckCuda(cudaPeekAtLastError());
  ErrorCheckCuda(cudaDeviceSynchronize());
}
//------------------------------------------------------------------------------
void CudaOptimiser::SmoothnessDualOptimisation() {
  int square_sides = sqrt(params_.num_features) + 1;
  dim3 num_threads(N_THREADS_BLOCK, N_THREADS_BLOCK);
  dim3 num_blocks(BlocksPerSide(square_sides, N_THREADS_BLOCK),
                  BlocksPerSide(square_sides, N_THREADS_BLOCK));

  update_smoothness_dual<<<num_blocks, num_threads>>>(
      smoothness_dual_, primal_relaxed_, nabla_, params_.num_labels,
      square_sides, params_.num_features, params_.alpha, params_.lambda);
  ErrorCheckCuda(cudaPeekAtLastError());
  ErrorCheckCuda(cudaDeviceSynchronize());
}

//------------------------------------------------------------------------------
void CudaOptimiser::PrimalOptimisation() {
  int square_sides = sqrt(params_.num_features) + 1;
  dim3 num_threads(N_THREADS_BLOCK, N_THREADS_BLOCK);
  dim3 num_blocks(BlocksPerSide(square_sides, N_THREADS_BLOCK),
                  BlocksPerSide(square_sides, N_THREADS_BLOCK));

  update_primal<<<num_blocks, num_threads>>>(
      primal_, primal_relaxed_, smoothness_dual_, compactness_dual_,
      model_costs_, inverse_neighbour_index__, params_.num_labels, square_sides,
      params_.num_features, params_.tau, params_.lambda, params_.beta,
      params_.max_neighbours);

  ErrorCheckCuda(cudaPeekAtLastError());
  ErrorCheckCuda(cudaDeviceSynchronize());
}

} // namespace optimiser
} // namespace cuda