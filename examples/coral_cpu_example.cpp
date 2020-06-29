
// TODO: Fix includes
#include "../open_coral/include/coral.h"
#include "../open_coral/include/features/coral_feature_point.h"
#include "../open_coral/include/model_initialiser.h"
#include "../open_coral/include/models/coral_model_line.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <iostream>

#include <glog/logging.h>
#include <gflags/gflags.h>

using namespace coral::features;
using namespace coral;
using namespace coral::models;
using namespace coral::optimiser;

int main(int argc, char* argv[])
{
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  LOG(INFO) << "This is an info  message";

  int num_points = 10;
  int num_dimensions = 2;

  coral::features::FeatureVectorSPtr line_features(
      new coral::features::FeatureVector);

  // y=2x+3
  for (int i = 0; i < num_points; ++i) {
    line_features->push_back(
        boost::make_shared<CoralFeaturePoint>(Eigen::Vector2d(i, 2 * i + 3)));
  }

  // y=2x-3
  for (int i = 0; i < num_points; ++i) {
    line_features->push_back(
        boost::make_shared<CoralFeaturePoint>(Eigen::Vector2d(i, 2 * i - 3)));
  }

  // y=-2x-3
  for (int i = 0; i < num_points; ++i) {
    line_features->push_back(
        boost::make_shared<CoralFeaturePoint>(Eigen::Vector2d(i, -2 * i - 3)));
  }

  ModelInitialiserParams mi_params(100, 0.95);

  ModelInitialiser<CoralModelLine> line_model_initialiser(mi_params);

  int num_models = 3;
  float threshold = 0.1;

  ModelVectorSPtr line_models(new ModelVector);
  line_model_initialiser.Initialise(line_features, num_models, threshold,
                                    line_models);

  std::cout << "Number of models is " << line_models->size() << "\n";
  cv::Mat neighbour_index, neighbour_transpose;

  coral::optimiser::CoralOptimiserParams params{};

  params.num_features = line_features->size();
  params.num_neighbours = 2;
  params.outlier_threshold=3;

  params.num_labels = line_models->size()+1;
  params.num_iterations = 1;
  params.num_loops = 1;

  params.lambda = 1;
  params.beta = 1;
  params.nu = 1;
  params.alpha =1;
  params.tau=1;

  CoralOptimiser<CoralModelLine> optimiser(params);
  LOG(INFO) << "Number of  models is " << line_models->size();
  LOG(INFO) << "Number of features is " << line_features->size();
  optimiser.EnergyMinimisation(line_features, line_models);


  return 0;
}
