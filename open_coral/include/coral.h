#ifndef CORAL_H_
#define CORAL_H_

#include "models/coral_model.h"
#include "nanoflann.hpp"
#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <glog/logging.h>
#include <iostream>

namespace coral {
namespace optimiser {

struct CoralOptimiserParams {
  int num_neighbours;

  double outlier_threshold;

  double lambda;

  double beta;
  double tau;

  double alpha;

  double nu;

  int num_features;

  int num_labels;
  int num_loops;

  int num_iterations;

  uint max_neighbours;

  int height_image;
  int width_image;

  bool use_label_dual;

  bool use_pyramid;

  bool update_models;

  double pyramid_scale;

  int pyramid_levels;
};

template <typename Model> class CoralOptimiser {

  typedef Eigen::MatrixXd Dual;
  typedef Eigen::MatrixXd Primal;
  typedef Eigen::MatrixXd RelaxedPrimal;
  typedef Eigen::SparseMatrix<double> Gradient;
  typedef Eigen::MatrixXd Label;

public:
  CoralOptimiser(const CoralOptimiserParams coral_optimiser_params);

  ~CoralOptimiser();

public:
  Eigen::MatrixXd
  EnergyMinimisation(const features::FeatureVectorSPtr &features,
                     models::ModelVectorSPtr models);

  Eigen::MatrixXd EvaluateModelCost(const features::FeatureVectorSPtr &features,
                                    const models::ModelVectorSPtr &models);

  void UpdateNumFeatures(int num_features) {
    coral_optimiser_params_.num_features = num_features;
  }
  void UpdateNumLabels(int num_labels) {
    coral_optimiser_params_.num_labels = num_labels;
  }

  static Eigen::MatrixXd SimplexProjectionVector(Eigen::MatrixXd matrix);

    Gradient GetGradient(){return neighbour_index_;};
void FindNearestNeighbours(const features::FeatureVectorSPtr &features);

private:
  void InitialiseVariables();

  void UpdateSmoothnessDual();

  void UpdatePrimal();

  Dual GetClampedDualNorm(Dual dual, double clamp_value);

  static void ClampVariable(Primal &primal, double clamp_value);

  void LabelsFromPrimal();

  void UpdateModels(features::FeatureVectorSPtr features,
                    models::ModelVectorSPtr models);

  void SimplexProjection();

  static Eigen::MatrixXd SortMatrix(Primal primal_matrix);

private:
  CoralOptimiserParams coral_optimiser_params_;

  Primal primal_;
  Primal primal_relaxed_;

  Dual smoothness_dual_;
  Dual compactness_dual_;

  Label label_;
  Gradient neighbour_index_;

  Eigen::MatrixXd model_costs_;
};

//------------------------------------------------------------------------------
template <typename InputType>
CoralOptimiser<InputType>::CoralOptimiser(
    const CoralOptimiserParams coral_optimiser_params)
    : coral_optimiser_params_(coral_optimiser_params) {}

//------------------------------------------------------------------------------
template <typename InputType>
CoralOptimiser<InputType>::~CoralOptimiser() = default;
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::InitialiseVariables() {
  // Set up the optimisation variables
  model_costs_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                       coral_optimiser_params_.num_labels);
  primal_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                  coral_optimiser_params_.num_labels);
  primal_relaxed_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features,
                                          coral_optimiser_params_.num_labels);
  smoothness_dual_ =
      Eigen::MatrixXd::Zero(coral_optimiser_params_.num_neighbours *
                                coral_optimiser_params_.num_features,
                            coral_optimiser_params_.num_labels);
  compactness_dual_ = Eigen::MatrixXd::Zero(
      coral_optimiser_params_.num_features, coral_optimiser_params_.num_labels);

  label_ = Eigen::MatrixXd::Zero(coral_optimiser_params_.num_features, 1);
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::FindNearestNeighbours(
    const coral::features::FeatureVectorSPtr &features) {

  int nn = coral_optimiser_params_.num_neighbours + 1;
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

  // Create the gradient variable

  neighbour_index_ = Gradient(coral_optimiser_params_.num_neighbours *
                                  coral_optimiser_params_.num_features,
                              coral_optimiser_params_.num_features);
  neighbour_index_.reserve(coral_optimiser_params_.num_neighbours *
                           coral_optimiser_params_.num_features);

  uint cont = 0;
  for (int i = 0; i < features->size(); ++i) {
    std::vector<size_t> ret_indexes(nn);
    std::vector<double> out_dists_sqr(nn);

    nanoflann::KNNResultSet<double> resultSet(nn);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);

    mat_index.index->findNeighbors(resultSet, &query_points[i][0],
                                   nanoflann::SearchParams(10));

    for (int k = 1; k < nn; ++k) {
      neighbour_index_.coeffRef(cont, i)=-1;
      neighbour_index_.coeffRef(cont, ret_indexes[k])=1;
      cont++;
    }
  }
}
//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd CoralOptimiser<InputType>::EvaluateModelCost(
    const features::FeatureVectorSPtr &features,
    const models::ModelVectorSPtr &models) {
  Eigen::MatrixXd ModelMatrix = Eigen::MatrixXd::Constant(
      coral_optimiser_params_.num_features, coral_optimiser_params_.num_labels,
      coral_optimiser_params_.outlier_threshold);

  for (int i = 0; i < coral_optimiser_params_.num_labels - 1; ++i) {
    ModelMatrix.col(i) = (*models)[i]->EvaluateCost(features);
  }
  return ModelMatrix;
}
//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd
CoralOptimiser<InputType>::GetClampedDualNorm(Dual dual, double clamp_value) {

  Dual l2_norm = Dual::Zero(coral_optimiser_params_.num_features,
                            coral_optimiser_params_.num_labels);
  uint counter = 0;
  for (int feat_no = 0; feat_no < coral_optimiser_params_.num_features;
       ++feat_no) {
    for (int neighbour_no = 0;
         neighbour_no < coral_optimiser_params_.num_neighbours;
         ++neighbour_no) {
      l2_norm.row(feat_no) =
          l2_norm.row(feat_no) + (dual.row(counter)).array().square().matrix();
      counter++;
    }
  }
  l2_norm = l2_norm.cwiseSqrt();
  Dual replicated_l2_norm = Dual::Zero(coral_optimiser_params_.num_neighbours *
                                           coral_optimiser_params_.num_features,
                                       coral_optimiser_params_.num_labels);
  counter = 0;
  for (int feat_no = 0; feat_no < coral_optimiser_params_.num_features;
       ++feat_no) {
    for (int neighbour_no = 0;
         neighbour_no < coral_optimiser_params_.num_neighbours;
         ++neighbour_no) {
      replicated_l2_norm.row(counter) = l2_norm.row(feat_no);
      counter++;
    }
  }

  // Clamp the norm

  for (int i = 0; i < replicated_l2_norm.rows(); ++i) {
    for (int j = 0; j < replicated_l2_norm.cols(); ++j) {
      if (replicated_l2_norm(i, j) < clamp_value)
        replicated_l2_norm(i, j) = clamp_value;
    }
  }
  return replicated_l2_norm;
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::UpdateSmoothnessDual() {

  Eigen::MatrixXd intermediate_dual, l2_norm_dual;

  intermediate_dual = smoothness_dual_ + coral_optimiser_params_.lambda *
                                             coral_optimiser_params_.alpha *
                                             neighbour_index_ * primal_relaxed_;
  l2_norm_dual = GetClampedDualNorm(intermediate_dual, 1);

  smoothness_dual_ =
      (intermediate_dual.array() / l2_norm_dual.array()).matrix();
}
//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd CoralOptimiser<InputType>::SortMatrix(Primal primal_matrix) {
  Primal sorted_matrix = primal_matrix;
  for (int i = 0; i < primal_matrix.rows(); ++i) {
    Eigen::VectorXd x = primal_matrix.row(i);
    std::sort(x.data(), x.data() + x.size());
    sorted_matrix.row(i) = x;
  }
  return sorted_matrix;
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::ClampVariable(Primal &primal,
                                              double clamp_value) {
  for (int i = 0; i < primal.rows(); ++i) {
    for (int j = 0; j < primal.cols(); ++j) {
      if (primal(i, j) < clamp_value)
        primal(i, j) = clamp_value;
    }
  }
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::SimplexProjection() {
  Primal primal_sorted = SortMatrix(primal_);

  // create x zero
  int final_column_no = primal_sorted.cols() - 1;
  Eigen::VectorXd x_zero = primal_sorted.col(final_column_no) -
                           Primal::Constant(primal_sorted.rows(), 1, 1);

  Primal primal_new(primal_sorted.rows(), primal_sorted.cols() + x_zero.cols());
  primal_new << primal_sorted, x_zero;
  Primal primal_new_sorted = SortMatrix(primal_new);

  // Create the f matrix

  int num_rows = primal_new_sorted.rows();
  int num_cols = primal_new_sorted.cols();

  Primal f = Primal::Zero(num_rows, num_cols);

  for (int i = 0; i < num_cols; ++i) {
    Primal curr_delta = primal_new_sorted.col(i);
    Primal tau_vec = primal_new_sorted - curr_delta.replicate(1, num_cols);
    ClampVariable(tau_vec, 0);
    f.col(i) = tau_vec.rowwise().sum();
  }

  // calculate the indices
  Eigen::MatrixXi minimum_matrix = (f.array() >= 1).cast<int>();
  Eigen::MatrixXi index(num_rows, 1);

  for (int i = 0; i < num_rows; ++i) {
    Eigen::MatrixXi::Index col_no;
    minimum_matrix.row(i).minCoeff(&col_no);
    index(i) = col_no - 1;
  }
  // Calculate the optimal value of v
  Primal v(num_rows, 1);
  for (int i = 0; i < num_rows; ++i) {
    int curr_index = index(i);
    v(i) = primal_new_sorted(i, curr_index) +
           (1 - f(i, curr_index)) *
               (primal_new_sorted(i, curr_index) -
                primal_new_sorted(i, curr_index + 1)) /
               (f(i, curr_index) - f(i, curr_index + 1));
  }
  // Calculate the new primal variable
  Primal updated_primal = primal_ - v.replicate(1, primal_.cols());
  ClampVariable(updated_primal, 0);
  primal_ = updated_primal;
}

//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd
CoralOptimiser<InputType>::SimplexProjectionVector(Eigen::MatrixXd matrix) {
  Primal matrix_sorted = SortMatrix(matrix);

  // create x zero
  int final_column_no = matrix_sorted.cols() - 1;
  Eigen::VectorXd x_zero = matrix_sorted.col(final_column_no) -
                           Primal::Constant(matrix_sorted.rows(), 1, 1);

  Primal matrix_new(matrix_sorted.rows(), matrix_sorted.cols() + x_zero.cols());
  matrix_new << matrix_sorted, x_zero;
  Primal matrix_new_sorted = SortMatrix(matrix_new);

  // Create the f matrix

  int num_rows = matrix_new_sorted.rows();
  int num_cols = matrix_new_sorted.cols();

  Primal f = Primal::Zero(num_rows, num_cols);

  for (int i = 0; i < num_cols; ++i) {
    Primal curr_delta = matrix_new_sorted.col(i);
    Primal tau_vec = matrix_new_sorted - curr_delta.replicate(1, num_cols);
    ClampVariable(tau_vec, 0);
    f.col(i) = tau_vec.rowwise().sum();
  }

  // calculate the indices
  Eigen::MatrixXi minimum_matrix = (f.array() >= 1).cast<int>();
  Eigen::MatrixXi index(num_rows, 1);

  for (int i = 0; i < num_rows; ++i) {
    Eigen::MatrixXi::Index col_no;
    minimum_matrix.row(i).minCoeff(&col_no);
    index(i) = col_no - 1;
  }
  // Calculate the optimal value of v
  Primal v(num_rows, 1);
  for (int i = 0; i < num_rows; ++i) {
    int curr_index = index(i);
    v(i) = matrix_new_sorted(i, curr_index) +
           (1 - f(i, curr_index)) *
               (matrix_new_sorted(i, curr_index) -
                matrix_new_sorted(i, curr_index + 1)) /
               (f(i, curr_index) - f(i, curr_index + 1));
  }
  // Calculate the new primal variable
  Primal updated_matrix = matrix - v.replicate(1, matrix.cols());
  ClampVariable(updated_matrix, 0);
  return updated_matrix;
}
//------------------------------------------------------------------------------
template <typename InputType> void CoralOptimiser<InputType>::UpdatePrimal() {
  Primal intermediate_primal, prev_primal;

  intermediate_primal = model_costs_ +
                        coral_optimiser_params_.lambda *
                            neighbour_index_.transpose() * smoothness_dual_ +
                        coral_optimiser_params_.beta * compactness_dual_;
  prev_primal = primal_;
  // Update primal
  primal_ = prev_primal - coral_optimiser_params_.tau * intermediate_primal;
  SimplexProjection();
  primal_relaxed_ = 2 * primal_ - prev_primal;
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::LabelsFromPrimal() {
  for (int i = 0; i < coral_optimiser_params_.num_features; ++i) {
    Eigen::MatrixXi::Index index = 0;
    primal_.row(i).maxCoeff(&index);
    label_(i, 0) = index;
  }
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralOptimiser<InputType>::UpdateModels(
    features::FeatureVectorSPtr features, models::ModelVectorSPtr models) {
  for (int i = 0; i < coral_optimiser_params_.num_labels - 1; ++i) {
    features::FeatureVectorSPtr model_update_features(
        new features::FeatureVector);

    for (int j = 0; j < coral_optimiser_params_.num_features; ++j) {
      if (label_(j) == 1) {
        model_update_features->push_back((*features)[j]);
      }
    }
    // Update the models
    (*models)[i]->UpdateModel(model_update_features);
  }
}
//------------------------------------------------------------------------------
template <typename InputType>
Eigen::MatrixXd CoralOptimiser<InputType>::EnergyMinimisation(
    const features::FeatureVectorSPtr &features,
    models::ModelVectorSPtr models) {
  // Update the params
  coral_optimiser_params_.num_features = features->size();
  coral_optimiser_params_.num_labels = models->size() + 1;
  LOG(INFO) << "Number of labels is " << coral_optimiser_params_.num_labels
            << std::endl;

  InitialiseVariables();
  LOG(INFO) << "Variables initialised" << std::endl;
  // Update the gradient  and model cost variables
  FindNearestNeighbours(features);
  LOG(INFO) << "Nearest neighbours found" << std::endl;
  model_costs_ = EvaluateModelCost(features, models);

  for (int curr_loop = 0; curr_loop < coral_optimiser_params_.num_loops;
       ++curr_loop) {
    for (int iter = 0; iter < coral_optimiser_params_.num_iterations; ++iter) {
      // Update dual and primal
      UpdateSmoothnessDual();
      UpdatePrimal();
    }

    // get the labels
    LabelsFromPrimal();
    if (coral_optimiser_params_.update_models)
      UpdateModels(features, models);

    model_costs_ = EvaluateModelCost(features, models);
  }
  LOG(INFO) << "Model assignment is " << primal_.colwise().sum();
  return label_;
}

} // namespace optimiser
} // namespace coral

#endif // CORAL_H_
