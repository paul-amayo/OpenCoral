
#ifndef CORAL_FEATURES_POINT_H_
#define CORAL_FEATURES_POINT_H_

#include "coral_feature.h"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <boost/shared_ptr.hpp>
#include <iostream>

namespace coral {
namespace features {

class CoralFeaturePoint : public CoralFeatureBase {
public:
  CoralFeaturePoint(const Eigen::VectorXd &point);
  virtual ~CoralFeaturePoint() = default;

  float Compare(boost::shared_ptr<CoralFeatureBase> &other_feature);

  void SetPoint(const Eigen::VectorXd &point);

  Eigen::MatrixXd GetFeatureValue();

private:
  Eigen::MatrixXd point_;
};
typedef boost::shared_ptr<CoralFeaturePoint> CoralFeaturePointSPtr;
typedef std::vector<CoralFeaturePointSPtr> CoralFeaturePointVector;
typedef boost::shared_ptr<CoralFeaturePointVector> CoralFeaturePointVectorSPtr;
} // namespace features
} // namespace coral

#endif // CORAL_FEATURES_POINT_H_
