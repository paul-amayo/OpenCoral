
#ifndef CORAL_FEATURES_BASE_H_
#define CORAL_FEATURES_BASE_H_

#include <Eigen/Dense>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <vector>
#include "glog/logging.h"

namespace coral {
namespace features {

class CoralFeatureBase {
public:
  virtual ~CoralFeatureBase() = default;

  virtual float Compare(boost::shared_ptr<CoralFeatureBase> &other_feature) = 0;

  virtual Eigen::MatrixXd GetFeatureValue() = 0;
};

typedef boost::shared_ptr<CoralFeatureBase> FeatureSPtr;

typedef std::vector<FeatureSPtr> FeatureVector;

typedef boost::shared_ptr<FeatureVector> FeatureVectorSPtr;

} // namespace features
} // namespace coral

#endif //CORAL_FEATURES_BASE_H