#ifndef CORAL_MODEL_BASE_H_
#define CORAL_MODEL_BASE_H_

#include "../features/coral_feature.h"

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace coral {

namespace models {

class CoralModelBase {
public:


  virtual ~CoralModelBase() = default;

public:
  virtual Eigen::MatrixXd
  EvaluateCost(const features::FeatureVectorSPtr &features) = 0;

  virtual void UpdateModel(const features::FeatureVectorSPtr &features) = 0;


  //virtual  float Compare(boost::shared_ptr<CoralModelBase> other_model)=0;

  virtual int ModelDegreesOfFreedom() = 0;

  virtual Eigen::MatrixXd ModelEquation() = 0;
};

typedef boost::shared_ptr<CoralModelBase> ModelSPtr;

typedef std::vector<ModelSPtr> ModelVector;

typedef boost::shared_ptr<ModelVector> ModelVectorSPtr;
} // namespace models
} // namespace coral
#endif // CORAL_MODEL_BASE_H_