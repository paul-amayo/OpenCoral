#include "../include/models/coral_affine_model.h"
#include "../include/features/coral_feature_curvature.h"
#include <opencv2/core/eigen.hpp>

namespace coral {
namespace models {
//------------------------------------------------------------------------------
CoralModelAffine::CoralModelAffine() {
  affine_ = Eigen::MatrixXd::Zero(2, 3);
  A_ = affine_.block(0, 0, 2, 2);
  b_ = affine_.block(0, 2, 2, 1);
}
//------------------------------------------------------------------------------
Eigen::MatrixXd
CoralModelAffine::EvaluateCost(const features::FeatureVectorSPtr &features) {
  const int num_features = features->size();
  const int num_dimensions = 3;

  Eigen::MatrixXd output(num_features, 1);
  int feat_no = 0;
  for (const auto &feature : *features) {
    features::CoralFeatureCurvatureSptr curv_feature =
        boost::dynamic_pointer_cast<features ::CoralFeatureCurvature>(feature);
    Eigen::Vector2d point_1 = curv_feature->GetPoint1();
    Eigen::Vector2d point_2 = curv_feature->GetPoint2();

    Eigen::Vector2d point_proj = A_ * point_1 + b_;

    output(feat_no, 0) = (point_2 - point_proj).norm() + 0.001;

    feat_no++;
  }

  return output;
}
//------------------------------------------------------------------------------
void CoralModelAffine::UpdateModel(
    const features::FeatureVectorSPtr &features) {
  const int num_features = features->size();
  const int num_dimensions = 3;
  Eigen::VectorXd x_values(num_features);
  Eigen::VectorXd y_values(num_features);

  std::vector<cv::Point2f> points_1, points_2;
  int feat_no = 0;
  for (const auto &feature : *features) {
    features::CoralFeatureCurvatureSptr curv_feature =
        boost::dynamic_pointer_cast<features ::CoralFeatureCurvature>(feature);
    Eigen::Vector2d point_1 = curv_feature->GetPoint1();
    Eigen::Vector2d point_2 = curv_feature->GetPoint2();

    points_1.emplace_back(cv::Point2f(float(point_1(0)), float(point_1(1))));
    points_2.emplace_back(cv::Point2f(float(point_2(0)), float(point_2(1))));
    feat_no++;
  }
  cv::Mat M = cv::estimateAffine2D(points_1, points_2);
  cv::cv2eigen(M, affine_);
  A_ = affine_.block(0, 0, 2, 2);
  b_ = affine_.block(0, 2, 2, 1);
}

//------------------------------------------------------------------------------
int CoralModelAffine::ModelDegreesOfFreedom() { return 3; }
//------------------------------------------------------------------------------
Eigen::MatrixXd CoralModelAffine::ModelEquation() { return affine_; }

} // namespace models
} // namespace coral
