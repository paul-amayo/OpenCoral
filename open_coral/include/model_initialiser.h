#include "nanoflann.hpp"
#include "models/coral_pnp_model.h"
#include <Eigen/Sparse>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <glog/logging.h>
#include <iostream>

namespace coral {
namespace models {

struct ModelInitialiserParams {
  int ransac_max_iterations;
  double ransac_prob;

  ModelInitialiserParams() : ransac_max_iterations(0), ransac_prob(0) {}

  ModelInitialiserParams(int iterations, double probability)
      : ransac_max_iterations(iterations), ransac_prob(probability) {}
};

template <typename ModelType> class ModelInitialiser {

public:
  ModelInitialiser(const ModelInitialiserParams model_init_params);

  ~ModelInitialiser() = default;

  void Initialise(const features::FeatureVectorSPtr input_features,
                  int num_models, float threshold, ModelVectorSPtr models);

  void Ransac(features::FeatureVectorSPtr input_features, float threshold,
              ModelSPtr output_model);

  void SetCameraMatrix(Eigen::Matrix3d K){K_=K;}

private:
  ModelInitialiserParams model_init_params_;

  Eigen::Matrix3d K_;
};

//------------------------------------------------------------------------------
template <typename InputType>
ModelInitialiser<InputType>::ModelInitialiser(
    const ModelInitialiserParams model_init_params)
    : model_init_params_(model_init_params) {}

//------------------------------------------------------------------------------
template <typename InputType>
void ModelInitialiser<InputType>::Initialise(
    const features::FeatureVectorSPtr input_features, int num_models,
    float threshold, ModelVectorSPtr models) {
  // clear the model vector
  models->clear();
  LOG(INFO) << "Begin initialising";

  features::FeatureVectorSPtr features_init(new features::FeatureVector);
  *features_init = *input_features;

  boost::shared_ptr<InputType> new_model_ptr(new InputType());

  int i = 0;
  while (i < num_models &&
         features_init->size() > new_model_ptr->ModelDegreesOfFreedom()) {
    boost::shared_ptr<InputType> model_ptr(new InputType());
    Ransac(features_init, threshold, model_ptr);
    models->push_back(model_ptr);
    i++;
  }
}

//------------------------------------------------------------------------------
template <>
void ModelInitialiser<CoralPNPModel>::Initialise(
    const features::FeatureVectorSPtr input_features, int num_models,
    float threshold, ModelVectorSPtr models) {
  // clear the model vector
  models->clear();
  LOG(INFO) << "Begin initialising Pnp Model";

  features::FeatureVectorSPtr features_init(new features::FeatureVector);
  *features_init = *input_features;

  boost::shared_ptr<CoralPNPModel> new_model_ptr(new CoralPNPModel(K_));

  int i = 0;
  while (i < num_models &&
         features_init->size() > new_model_ptr->ModelDegreesOfFreedom()) {
    boost::shared_ptr<CoralPNPModel> model_ptr(new CoralPNPModel(K_));
    Ransac(features_init, threshold, model_ptr);
    models->push_back(model_ptr);
    i++;
  }
}
//------------------------------------------------------------------------------
template <typename InputType>
void ModelInitialiser<InputType>::Ransac(
    coral::features::FeatureVectorSPtr input_features, float threshold,
    coral::models::ModelSPtr output_model) {

  LOG(INFO) << "Ransac Init";
  int DOF = output_model->ModelDegreesOfFreedom();
  int num_points = input_features->size();
  float current_prob = (float)DOF / num_points;

  uint max_iterations_prob =
      log(1 - model_init_params_.ransac_prob) / log(1 - pow(current_prob, DOF));
  uint num_iterations = 0;

  int max_inliers = 0;
  Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> output_inlier_matrix;

  while (num_iterations < max_iterations_prob &&
         num_iterations < model_init_params_.ransac_max_iterations) {

    Eigen::VectorXi ind_features = Eigen::VectorXi::Constant(DOF, num_points);
    features::FeatureVectorSPtr model_selection_features(
        new features::FeatureVector);
    for (int i = 0; i < DOF; ++i) {
      int ind_random_feature = rand() % num_points;
      while ((ind_features.array() == ind_random_feature).any())
        ind_random_feature = rand() % num_points;
      ind_features(i) = ind_random_feature;
      features::FeatureSPtr curr_feature_sptr =
          (*input_features)[ind_random_feature];
      model_selection_features->push_back(curr_feature_sptr);
    }

    output_model->UpdateModel(model_selection_features);
    Eigen::MatrixXd feature_costs_model =
        output_model->EvaluateCost(model_selection_features);

    Eigen::MatrixXd feature_costs = output_model->EvaluateCost(input_features);
    uint num_inliers = (feature_costs.array() < threshold).count();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> inlier_matrix =
        (feature_costs.array() < threshold).matrix();

    if (num_inliers > max_inliers) {
      LOG(INFO)<<"Number inliers is "<<num_inliers;
      max_inliers = num_inliers;
      output_inlier_matrix = inlier_matrix;
      current_prob = (float)max_inliers / num_points;

      max_iterations_prob = log(1 - model_init_params_.ransac_prob) /
                            log(1 - pow(current_prob, DOF));
    }
      num_iterations++;
  }
  // Select the inliers and use them to update the initial model
  features::FeatureVectorSPtr inlier_features(new features::FeatureVector);
  features::FeatureVector outlier_features;

  for (int i = 0; i < num_points; ++i) {
    if (output_inlier_matrix(i)) {
      inlier_features->push_back((*input_features)[i]);
    } else {
      outlier_features.push_back((*input_features)[i]);
    }
  }
  LOG(INFO)<<"Size of input features is "<<inlier_features->size();
  output_model->UpdateModel(inlier_features);

  *input_features = outlier_features;
}

} // namespace models
} // namespace coral