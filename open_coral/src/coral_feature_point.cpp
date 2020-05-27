#include "../include/features/coral_feature_point.h"
#include <Eigen/Dense>

namespace coral {
namespace features {

CoralFeaturePoint::CoralFeaturePoint(const Eigen::VectorXd& point) { point_ =
point; }

float CoralFeaturePoint::Compare(
    boost::shared_ptr<CoralFeatureBase> &other_feature) {
  return (point_ - other_feature->GetFeatureValue()).norm();
}

Eigen::MatrixXd CoralFeaturePoint::GetFeatureValue() { return point_; }

void CoralFeaturePoint::SetPoint(const Eigen::VectorXd& point) { point_ =
point; }

} // namespace features
} // namespace coral
