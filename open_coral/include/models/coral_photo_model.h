#ifndef CORAL_PHOTO_MODEL_H_
#define CORAL_PHOTO_MODEL_H_

#include "../features/coral_feature.h"
#include "../features/coral_feature_stereo_correspondence.h"
#include "coral_model.h"

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace coral {

namespace models {

typedef Eigen::Matrix3d Rot;
typedef Eigen::Vector3d Trans;

class CoralPhotoModel : public CoralModelBase {
public:
  CoralPhotoModel(Eigen::Matrix3d K, double baseline);

  virtual ~CoralPhotoModel() = default;

public:
  Eigen::MatrixXd EvaluateCost(const features::FeatureVectorSPtr &features);

  void UpdateModel(const features::FeatureVectorSPtr &features);

  int ModelDegreesOfFreedom();

  Eigen::MatrixXd ModelEquation();

  cv::Mat TransformImageMotion(cv::Mat image_colour, cv::Mat disparity,
                               const Eigen::Matrix3d &Rotation,
                               const Eigen::Vector3d &translation);

private:
  Eigen::Matrix3d K_;
  double baseline_;
};

} // namespace models
} // namespace coral
#endif // CORAL_PHOTO_MODEL_H_