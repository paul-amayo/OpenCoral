#include "../include/models/coral_model_line.h"
#include <Eigen/src/Eigenvalues/SelfAdjointEigenSolver.h>

namespace coral {
namespace models {
//------------------------------------------------------------------------------
CoralModelLine::CoralModelLine() {
  line_params_ = Eigen::VectorXd::Zero(3);
  angle_thresh_ = 10;
  dist_thresh_ = 10;
}
//------------------------------------------------------------------------------
 CoralModelLine::CoralModelLine(
      Eigen::VectorXd line_params)
    : line_params_(line_params) {
  angle_thresh_ = 10;
  dist_thresh_ = 10;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd
CoralModelLine::EvaluateCost(const features::FeatureVectorSPtr &features) {
  const int num_features = features->size();
  const int num_dimensions = 3;

  Eigen::MatrixXd point_values(num_features, num_dimensions);
  int feat_no = 0;
  for (const auto &feature : *features) {
    Eigen::VectorXd point_curr = feature->GetFeatureValue();
    point_values(feat_no, 0) = point_curr(0);
    point_values(feat_no, 1) = point_curr(1);
    point_values(feat_no, 2) = 1;

    feat_no++;
  }
  float denom = sqrt(pow(line_params_(0), 2) + pow(line_params_(1), 2));
  Eigen::MatrixXd output = (line_params_.transpose() * point_values.transpose())
                               .cwiseAbs()
                               .transpose() /
                           denom;

  return output;
}
//------------------------------------------------------------------------------
void CoralModelLine::UpdateModel(const features::FeatureVectorSPtr &features) {
  const int num_features = features->size();
  const int num_dimensions = 3;
  Eigen::VectorXd x_values(num_features);
  Eigen::VectorXd y_values(num_features);

  int feat_no = 0;
  for (const auto &feature : *features) {
    Eigen::MatrixXd point_curr = feature->GetFeatureValue();
    x_values(feat_no) = point_curr(0);
    y_values(feat_no) = point_curr(1);
    feat_no++;
  }

  line_params_ = CalculateLeastSquaresModel(x_values, y_values);
}
//------------------------------------------------------------------------------
Eigen::VectorXd
CoralModelLine::CalculateLeastSquaresModel(Eigen::VectorXd x_values,
                                           Eigen::VectorXd y_values) {
  const int num_points = x_values.rows();

  float sum_x = x_values.sum() / num_points;
  float sum_y = y_values.sum() / num_points;
  float sum_x2 = x_values.array().square().sum() / num_points;
  float sum_y2 = y_values.array().square().sum() / num_points;
  float sum_xy =
      (x_values.array() * y_values.array()).matrix().sum() / num_points;

  Eigen::Matrix2d C;
  C(0, 0) = sum_x2 - sum_x * sum_x;
  C(1, 0) = sum_xy - sum_x * sum_y;
  C(0, 1) = sum_xy - sum_x * sum_y;
  C(1, 1) = sum_y2 - sum_y * sum_y;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen_solver(C);
  if (eigen_solver.info() != Eigen::Success) {
    std::cerr << "Failed eigen solve";
  }

  Eigen::Matrix2d V = eigen_solver.eigenvectors();
  Eigen::Vector3d line_params;
  line_params(0) = V(0, 0);
  line_params(1) = V(1, 0);

  float normaliser =
      sqrt(line_params(0) * line_params(0) + line_params(1) * line_params(1));

  line_params(0) /= normaliser;
  line_params(1) /= normaliser;
  line_params(2) = -line_params(0) * sum_x - line_params(1) * sum_y;

  return line_params;
}
//------------------------------------------------------------------------------
int CoralModelLine::ModelDegreesOfFreedom() { return 2; }
//------------------------------------------------------------------------------
Eigen::MatrixXd CoralModelLine::ModelEquation() { return line_params_; }

} // namespace models
} // namespace coral
