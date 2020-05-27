#ifndef CORAL_CUDA_WRAPPER_H_
#define CORAL_CUDA_WRAPPER_H_

#include "../open_coral/include/coral.h"
#include "cuda_optimiser.h"
#include <iostream>
#include <opencv2/core/eigen.hpp>

namespace cuda {
namespace coral_wrapper {

template <typename ModelType>
class CoralCudaWrapper : public coral::optimiser::CoralOptimiser<ModelType> {

public:
  CoralCudaWrapper();
  CoralCudaWrapper(const coral::optimiser::CoralOptimiserParams params);
  ~CoralCudaWrapper() = default;

  void EnergyMinimisation(const coral::features::FeatureVectorSPtr features,
                          coral::models::ModelVectorSPtr models);

  void FindNearestNeighbours(const coral::features::FeatureVectorSPtr features,
                             cv::Mat &neighbour_index,
                             cv::Mat &inverse_neighbour_index);
  cv::Mat Eigen2Cv(Eigen::MatrixXf eigen_matrix);

  Eigen::MatrixXf Cv2Eigen(cv::Mat opencv_matrix);

private:
  void WrapParams(const coral::optimiser::CoralOptimiserParams params);

  cuda::optimiser::CudaOptimiserParams params_;
};
//------------------------------------------------------------------------------
template <typename ModelType>
CoralCudaWrapper<ModelType>::CoralCudaWrapper(
    const coral::optimiser::CoralOptimiserParams params)
    : coral::optimiser::CoralOptimiser<ModelType>(params) {
  WrapParams(params);
}
//------------------------------------------------------------------------------
template <typename ModelType>
void CoralCudaWrapper<ModelType>::WrapParams(
    const coral::optimiser::CoralOptimiserParams params) {
  params_.num_labels = params.num_labels;
  params_.num_features = params.num_features;
  params_.num_neighbours = params.num_neighbours;
  params_.num_iterations = params.num_iterations;
  params_.num_loops = params.num_loops;

  params_.lambda = params.lambda;
  params_.beta = params.beta;
  params_.nu = params.nu;
  params_.alpha = params.alpha;
  params_.tau=params.tau;

  params_.max_neighbours = params.max_neighbours;
  params_.outlier_threshold = params.outlier_threshold;
}
//------------------------------------------------------------------------------
template <typename ModelType>
void CoralCudaWrapper<ModelType>::FindNearestNeighbours(
    const coral::features::FeatureVectorSPtr features, cv::Mat &neighbour_index,
    cv::Mat &inverse_neighbour_index) {
  int nn = params_.num_neighbours + 1;
  int dim = 2;

  Eigen::MatrixXd target_features(features->size(), dim);

  int feat_no = 0;
  std::vector<std::vector<double>> query_points;
  for (auto feature : *features) {
    Eigen::VectorXd feat_point = feature->GetFeatureValue();
    std::vector<double> current_point;
    current_point.push_back(feat_point(0));
    current_point.push_back(feat_point(1));
    query_points.push_back(current_point);
    target_features.row(feat_no) = feat_point;
    feat_no++;
  }

  // ------------------------------------------------------------
  // construct a kd-tree index:
  //    Some of the different possibilities (uncomment just one)
  // ------------------------------------------------------------
  // Dimensionality set at run-time (default: L2)
  typedef nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd> my_kd_tree_t;

  my_kd_tree_t mat_index(dim, std::cref(target_features), 10 /* max leaf */);
  mat_index.index->buildIndex();

  neighbour_index =
      cv::Mat(params_.num_features, params_.num_neighbours, CV_32F);
  std::vector<std::vector<float>> inverse_neighbour_vector(
      params_.num_features, std::vector<float>());
  uint n_inverse_neighbours = 0;
  for (int curr_feature = 0; curr_feature < params_.num_features;
       ++curr_feature) {
    std::vector<size_t> ret_indexes(nn);
    std::vector<double> out_dists_sqr(nn);

    nanoflann::KNNResultSet<double> resultSet(nn);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    mat_index.index->findNeighbors(resultSet, &query_points[curr_feature][0],
                                   nanoflann::SearchParams(10));

    for (int k = 1; k < nn; ++k) {
      // Get the neighbours of the current feature
      neighbour_index.at<float>(curr_feature, k - 1) = ret_indexes[k];

      // Get the indexes of features the current feature is a neighbour to
      inverse_neighbour_vector[ret_indexes[k]].push_back(curr_feature);
      if (n_inverse_neighbours <
          inverse_neighbour_vector[ret_indexes[k]].size()) {
        n_inverse_neighbours = inverse_neighbour_vector[ret_indexes[k]].size();
      }
    }
  }
  params_.max_neighbours = n_inverse_neighbours;
  for (int curr_feature = 0; curr_feature < params_.num_features;
       ++curr_feature) {
    // pad the rest of the inverse vector with zeros
    std::vector<float> curr_feature_neighbour(params_.max_neighbours, -1);
    std::copy(std::begin(inverse_neighbour_vector[curr_feature]),
              std::end(inverse_neighbour_vector[curr_feature]),
              std::begin(curr_feature_neighbour));

    // push to inverse neighbour
    inverse_neighbour_index.push_back(cv::Mat1f(curr_feature_neighbour).t());
  }

  // Take the transpose to reduce storage size
  neighbour_index = neighbour_index.t();
  inverse_neighbour_index = inverse_neighbour_index.t();
}
//------------------------------------------------------------------------------
template <typename ModelType>
void CoralCudaWrapper<ModelType>::EnergyMinimisation(
    const coral::features::FeatureVectorSPtr features,
    coral::models::ModelVectorSPtr models) {

  // Update the params
  params_.num_features = features->size();
  std::cout << "Number of features is " << params_.num_features << "\n";
  this->UpdateNumFeatures(params_.num_features);
  params_.num_labels = models->size() + 1;
  this->UpdateNumLabels(params_.num_labels);

  cv::Mat neighbour_index, inverse_neighbour_index;
  FindNearestNeighbours(features, neighbour_index, inverse_neighbour_index);
  Eigen::MatrixXd temp_costs = this->EvaluateModelCost(features, models);
  Eigen::MatrixXf model_costs = temp_costs.transpose().cast<float>();
  cv::Mat model_costs_cv = Eigen2Cv(model_costs);

  // Initilaise CUDA Optimiser
  cuda::optimiser::CudaOptimiser cuda_optimiser(
      model_costs_cv, neighbour_index, inverse_neighbour_index, params_);

  for (int curr_loop = 0; curr_loop < params_.num_loops; ++curr_loop) {

    std::cout << "Number of labels is " << params_.num_labels << "\n";

    cudaEvent_t start, stop;
    ErrorCheckCuda(cudaEventCreate(&start));
    ErrorCheckCuda(cudaEventCreate(&stop));
    ErrorCheckCuda(cudaEventRecord(start, 0));

    cuda::matrix::CudaMatrix<float> primal = cuda_optimiser.Optimise();
    ErrorCheckCuda(cudaEventRecord(stop, 0));
    ErrorCheckCuda(cudaEventSynchronize(stop));
    float optimisation_time;
    ErrorCheckCuda(cudaEventElapsedTime(&optimisation_time, start, stop));
    ErrorCheckCuda(cudaEventDestroy(start));
    ErrorCheckCuda(cudaEventDestroy(stop));

    std::cout << " The optimisation time was " << optimisation_time
              << " miliseconds \n";

  }
}
//------------------------------------------------------------------------------
template <typename ModelType>
cv::Mat CoralCudaWrapper<ModelType>::Eigen2Cv(Eigen::MatrixXf eigen_matrix) {
  cv::Mat output(eigen_matrix.rows(), eigen_matrix.cols(), CV_32F);
  cv::eigen2cv(eigen_matrix, output);
  return output;
}
//------------------------------------------------------------------------------
template <typename ModelType>
Eigen::MatrixXf CoralCudaWrapper<ModelType>::Cv2Eigen(cv::Mat opencv_matrix) {
  Eigen::MatrixXf output(opencv_matrix.rows, opencv_matrix.cols);
  cv::cv2eigen(opencv_matrix, output);
  return output;
}

} // namespace coral_wrapper
} // namespace cuda
#endif // CORAL_CUDA_WRAPPER_H_