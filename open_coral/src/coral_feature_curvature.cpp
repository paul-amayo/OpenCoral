#include "../include/features/coral_feature_curvature.h"
#include <Eigen/Dense>

namespace coral {
namespace features {
//------------------------------------------------------------------------------
CoralFeatureCurvature::CoralFeatureCurvature(
    const Eigen::Vector2d &point_uv_1, const Eigen::Vector2d &point_uv_2)
    :point_uv_1_(point_uv_1),point_uv_2_(point_uv_2) {
}

//------------------------------------------------------------------------------
float CoralFeatureCurvature::Compare(
    boost::shared_ptr<CoralFeatureBase> &other_feature) {
  return 0;
}

//------------------------------------------------------------------------------
Eigen::MatrixXd CoralFeatureCurvature::GetFeatureValue() {
  return point_uv_1_;
}

} // namespace features
} // namespace coral
