#include "../include/features/coral_feature_stereo_correspondance.h"
#include <Eigen/Dense>

namespace coral {
namespace features {

CoralFeatureStereoCorrespondance::CoralFeatureStereoCorrespondance(const Eigen::Vector2d &point_uv, const Eigen::Vector3d &point_3d) {
  point_uv_=point_uv;
  point_3d_=point_3d;
}

float CoralFeatureStereoCorrespondance::Compare(boost::shared_ptr<CoralFeatureBase> &other_feature) {
  return 0;
}
void CoralFeatureStereoCorrespondance::SetPoint(const Eigen::Vector2d &point_uv, const Eigen::Vector3d &point_3d) {
  point_uv_=point_uv;
  point_3d_=point_3d;
}

Eigen::MatrixXd CoralFeatureStereoCorrespondance::GetFeatureValue() {
  return point_uv_;
}

} // namespace features
} // namespace coral
