
#ifndef CORAL_FEATURES_STEREO_POINT_H_
#define CORAL_FEATURES_STEREO_POINT_H_

#include "coral_feature.h"
#include <Eigen/Dense>
#include <Eigen/src/Core/Matrix.h>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace coral {
namespace features {

class CoralFeatureStereoCorrespondence : public CoralFeatureBase {
public:
  CoralFeatureStereoCorrespondence();

  CoralFeatureStereoCorrespondence(const Eigen::Vector2d &point_uv_1,
                                   const Eigen::Vector2d &point_uv_2,
                                   const cv::Mat& image_2_left, const cv::Mat& image_2_right,
                                   double baseline,Eigen::Matrix3d K);

  CoralFeatureStereoCorrespondence(const Eigen::Vector2d &point_uv,
                                   const Eigen::Vector3d &point_world);

  virtual ~CoralFeatureStereoCorrespondence() = default;

  float Compare(boost::shared_ptr<CoralFeatureBase> &other_feature);

  void SetPoint(const Eigen::Vector2d &point_uv,
                const Eigen::Vector3d &point_3d);

  Eigen::Vector3d GetPoint3d() { return point_world_; }

  Eigen::Vector2d GetPointUV() { return point_uv_; }

  void GetStereoDisparity(const Eigen::Vector2d &point_uv_left,
                                 const cv::Mat& image_left, const cv::Mat& image_right);

  Eigen::Vector3d GetWorldPoint(const Eigen::Vector2d &point_uv_left,
                                double baseline,Eigen::Matrix3d K) const;

  int GetDisparity(){return disparity_;}

  Eigen::MatrixXd GetFeatureValue();

private:
  double ComputeKernelSAD(cv::Mat block_left, cv::Mat block_right) const;

  double min_disparity_;
  double max_disparity_;

  int kernel_size_;

  double disparity_;

  Eigen::Vector2d point_uv_;
  Eigen::Vector3d point_world_;
};
typedef boost::shared_ptr<CoralFeatureStereoCorrespondence>
    CoralFeatureStereoCorrespondenceSPtr;
typedef std::vector<CoralFeatureStereoCorrespondenceSPtr>
    CoralFeatureStereoCorrespondenceVector;
typedef boost::shared_ptr<CoralFeatureStereoCorrespondenceVector>
    CoralFeatureStereoCorrespondenceVectorSPtr;
} // namespace features
} // namespace coral

#endif // CORAL_FEATURES_STEREO_POINT_H_
