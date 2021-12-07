#ifndef CORAL_MODEL_LINE_H_
#define CORAL_MODEL_LINE_H_

#include "coral_model.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace coral {

namespace models {

class CoralModelLine : public CoralModelBase {
public:
  CoralModelLine();

  CoralModelLine(Eigen::VectorXd line_params);

  virtual ~CoralModelLine() = default;

public:
  Eigen::MatrixXd EvaluateCost(const features::FeatureVectorSPtr &features);

  void UpdateModel(const features::FeatureVectorSPtr &features);

  static Eigen::VectorXd CalculateLeastSquaresModel(Eigen::VectorXd x_values,
                                             Eigen::VectorXd y_values);

  int ModelDegreesOfFreedom();

  Eigen::MatrixXd ModelEquation();

private:
 Eigen::VectorXd  line_params_;
  float angle_thresh_;
  float dist_thresh_;
};
} // namespace models
} // namespace coral
#endif // CORAL_MODEL_LINE_H_