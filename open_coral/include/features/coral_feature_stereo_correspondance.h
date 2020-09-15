
#ifndef CORAL_FEATURES_STEREO_POINT_H_
#define CORAL_FEATURES_STEREO_POINT_H_

#include "coral_feature.h"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <boost/shared_ptr.hpp>
#include <iostream>

namespace coral {
namespace features {

class CoralFeatureStereoCorrespondance : public CoralFeatureBase {
public:
  CoralFeatureStereoCorrespondance(const Eigen::Vector2d &point_uv,const Eigen::Vector3d &point_3d);
  virtual ~CoralFeatureStereoCorrespondance() = default;

  float Compare(boost::shared_ptr<CoralFeatureBase> &other_feature);

  void SetPoint(const Eigen::Vector2d &point_uv,const Eigen::Vector3d &point_3d);

  Eigen::Vector3d GetPoint3d(){return point_3d_;}

  Eigen::Vector2d GetPointUV(){return point_uv_;}

  Eigen::MatrixXd GetFeatureValue();

private:
  Eigen::Vector2d point_uv_;
  Eigen::Vector3d point_3d_;
};
typedef boost::shared_ptr<CoralFeatureStereoCorrespondance> CoralFeatureStereoCorrespondanceSPtr;
typedef std::vector<CoralFeatureStereoCorrespondanceSPtr> CoralFeatureStereoCorrespondanceVector;
typedef boost::shared_ptr<CoralFeatureStereoCorrespondanceVector> CoralFeatureStereoCorrespondanceVectorSPtr;
} // namespace features
} // namespace coral

#endif // CORAL_FEATURES_STEREO_POINT_H_
