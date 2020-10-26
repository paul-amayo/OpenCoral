#include "../include/models/coral_photo_model.h"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/core/core_c.h>
#include <opencv2/ximgproc/disparity_filter.hpp>

namespace coral {
namespace models {
//------------------------------------------------------------------------------
CoralPhotoModel::CoralPhotoModel(Eigen::Matrix3d K, double baseline) {
  K_ = K;
  baseline_ = baseline;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd
CoralPhotoModel::EvaluateCost(const features::FeatureVectorSPtr &features) {

  Eigen::MatrixXd point_values(features->size(), 1);

  return point_values;
}
//------------------------------------------------------------------------------
void CoralPhotoModel::UpdateModel(const features::FeatureVectorSPtr &features) {

}
//------------------------------------------------------------------------------
int CoralPhotoModel::ModelDegreesOfFreedom() { return 4; }
//------------------------------------------------------------------------------
Eigen::MatrixXd CoralPhotoModel::ModelEquation() {
  return Eigen::MatrixXd::Zero(4, 4);
}
//------------------------------------------------------------------------------
cv::Mat
CoralPhotoModel::TransformImageMotion(cv::Mat image_colour, cv::Mat disparity,
                                      const Eigen::Matrix3d &Rotation,
                                      const Eigen::Vector3d &translation) {

  // find 3d position
  int num_rows = image_colour.rows;
  int num_cols = image_colour.cols;

  cv::Mat disp_vis;
  cv::ximgproc::getDisparityVis(disparity, disp_vis, 1);

  cv::Mat disp;
  disp_vis.convertTo(disp, CV_32F);

  cv::Mat transformed_image =
      cv::Mat(num_rows, num_cols, CV_8UC3, cv::Scalar(80, 80, 80));

  for (int u = 0; u < num_cols; ++u) {
    for (int v = 0; v < num_rows; ++v) {

      // obtain the 3d position
      int curr_disparity = disp.at<float>(v, u);
      Eigen::Vector3d point_world;
      double baseline_over_disparity_1 = baseline_ / curr_disparity;
      point_world(2) = baseline_over_disparity_1 * K_(0, 0);
      point_world(0) = (u - K_(0, 2)) * baseline_over_disparity_1;
      point_world(1) =
          (v - K_(1, 2)) * baseline_over_disparity_1 * (K_(0, 0) / K_(1, 1));


      // Project into camera
      Eigen::Vector3d transformed_position =
          Rotation * point_world + translation;
      Eigen::Vector3d uv = K_*transformed_position;

      // Normalise
      uv = uv / uv(2);

      if(u==150 && v==150){
        LOG(INFO)<<"Current disparity is "<<curr_disparity;
        LOG(INFO)<<"Point 3d is "<<point_world.transpose();
        LOG(INFO)<<"Point transformed is "<<transformed_position.transpose();
        LOG(INFO)<<"Point uv new is "<<uv.transpose();
      }
      if (uv(0) > 0 && uv(0) < num_cols && uv(1) > 0 && uv(1) < num_rows) {
        transformed_image.at<cv::Vec3b>(uv(1), uv(0)) =
            image_colour.at<cv::Vec3b>(v, u);
      }
    }
  }
  return transformed_image;
}
//------------------------------------------------------------------------------
} // namespace models
} // namespace coral
