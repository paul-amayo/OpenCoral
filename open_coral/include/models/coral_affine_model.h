#ifndef CORAL_MODEL_AFFINE_H_
#define CORAL_MODEL_AFFINE_H_

#include "coral_model.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace coral {

namespace models {

class CoralModelAffine : public CoralModelBase {
public:
  CoralModelAffine();

  CoralModelAffine(Eigen::MatrixXd affine);

  virtual ~CoralModelAffine() = default;

public:
  Eigen::MatrixXd EvaluateCost(const features::FeatureVectorSPtr &features);

  void UpdateModel(const features::FeatureVectorSPtr &features);

  int ModelDegreesOfFreedom();

  Eigen::MatrixXd ModelEquation();

private:
  Eigen::MatrixXd affine_;
  Eigen::Matrix2d A_;
  Eigen::Vector2d b_;
};
} // namespace models
} // namespace coral
#endif // CORAL_MODEL_AFFINE_H_