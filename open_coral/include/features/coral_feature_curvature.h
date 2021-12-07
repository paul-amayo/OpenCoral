
#ifndef CORAL_FEATURES_CURVATURE_H_
#define CORAL_FEATURES_CURVATURE_H_

#include "coral_feature.h"
#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace coral {
namespace features {

class CoralFeatureCurvature : public CoralFeatureBase {
public:
  CoralFeatureCurvature() = default;

  CoralFeatureCurvature(const Eigen::Vector2d &point_uv_1,
                        const Eigen::Vector2d &point_uv_2);

  virtual ~CoralFeatureCurvature() = default;

  float Compare(boost::shared_ptr<CoralFeatureBase> &other_feature);

  Eigen::MatrixXd GetFeatureValue();

  Eigen::Vector2d GetPoint1() { return point_uv_1_; }
  Eigen::Vector2d GetPoint2() { return point_uv_2_; }

private:
  Eigen::Vector2d point_uv_1_;
  Eigen::Vector2d point_uv_2_;
};
typedef boost::shared_ptr<CoralFeatureCurvature> CoralFeatureCurvatureSptr;
typedef std::vector<CoralFeatureCurvatureSptr> CoralFeatureCurvatureVector;
typedef boost::shared_ptr<CoralFeatureCurvatureVector>
    CoralFeatureCurvatureVectorSPtr;
} // namespace features
} // namespace coral

#endif // CORAL_FEATURES_STEREO_POINT_H_
