#include "../include/features/coral_feature_stereo_correspondence.h"
#include <Eigen/Dense>

namespace coral {
namespace features {
//------------------------------------------------------------------------------
CoralFeatureStereoCorrespondence::CoralFeatureStereoCorrespondence(
    const Eigen::Vector2d &point_uv_1, const Eigen::Vector2d &point_uv_2,
    const cv::Mat &image_2_left, const cv::Mat &image_2_right, double baseline,
    Eigen::Matrix3d K) {

  min_disparity_ = 0;
  max_disparity_ = 80;
  kernel_size_ = 3;
  disparity_=0;

  point_uv_ = point_uv_1;
  GetStereoDisparity(point_uv_2, image_2_left, image_2_right);
  point_world_ = GetWorldPoint(point_uv_2, baseline, K);
}
//------------------------------------------------------------------------------
CoralFeatureStereoCorrespondence::CoralFeatureStereoCorrespondence(
    const Eigen::Vector2d &point_uv, const Eigen::Vector3d &point_world) {
  point_uv_ = point_uv;
  point_world_ = point_world;

  max_disparity_ = 50;
  kernel_size_ = 5;
  min_disparity_=0;

}
//------------------------------------------------------------------------------
float CoralFeatureStereoCorrespondence::Compare(
    boost::shared_ptr<CoralFeatureBase> &other_feature) {
  return 0;
}
//------------------------------------------------------------------------------
void CoralFeatureStereoCorrespondence::SetPoint(
    const Eigen::Vector2d &point_uv, const Eigen::Vector3d &point_world) {
  point_uv_ = point_uv;
  point_world_ = point_world;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd CoralFeatureStereoCorrespondence::GetFeatureValue() {
  return point_uv_;
}
//------------------------------------------------------------------------------
void CoralFeatureStereoCorrespondence::GetStereoDisparity(
    const Eigen::Vector2d &point_uv_left, const cv::Mat &image_left,
    const cv::Mat &image_right) {

  double min_sad_score = 1e9;
  disparity_ = min_disparity_;

  int num_cols = image_right.cols;
  int num_rows = image_right.rows;

  cv::Mat final_bloc_left = cv::Mat(10, 10, CV_8U);
  cv::Mat final_bloc_right = cv::Mat(10, 10, CV_8U);

  for (int disp = 1; disp < max_disparity_; ++disp) {
    int u_left = point_uv_left(0);
    int v_left = point_uv_left(1);

    int u_right = u_left - disp;
    int v_right = v_left;

    if ((u_left >= kernel_size_) && (u_left <= num_cols - kernel_size_ - 1) &&
        (u_right >= kernel_size_) && (u_right <= num_cols - kernel_size_ - 1) &&
        (v_left >= kernel_size_) && (v_left <= num_rows - kernel_size_ - 1)) {

      cv::Rect roi_left(u_left - kernel_size_ / 2, v_left - kernel_size_ / 2,
                        kernel_size_, kernel_size_);
      cv::Rect roi_right(u_right - kernel_size_ / 2, v_right - kernel_size_ / 2,
                         kernel_size_, kernel_size_);

      cv::Mat bloc_left = cv::Mat(image_left, roi_left);
      cv::Mat bloc_right = cv::Mat(image_right, roi_right);

      double sad_score = ComputeKernelSAD(bloc_left, bloc_right);

      if (sad_score < min_sad_score) {
        final_bloc_left = bloc_left;
        final_bloc_right = bloc_right;

        min_sad_score = sad_score;
        disparity_ = disp;
      }
    }
  }
}
//------------------------------------------------------------------------------
Eigen::Vector3d CoralFeatureStereoCorrespondence::GetWorldPoint(
    const Eigen::Vector2d &point_uv_left, double baseline,
    Eigen::Matrix3d K) const {
  Eigen::Vector3d point_world;
  double baseline_over_disparity_1 = baseline / disparity_;
  point_world(2) = baseline_over_disparity_1 * K(0, 0);
  point_world(0) = (point_uv_left(0) - K(0, 2)) * baseline_over_disparity_1;
  point_world(1) = (point_uv_left(1) - K(1, 2)) * baseline_over_disparity_1 *
                   (K(0, 0) / K(1, 1));

  return point_world;
}
//------------------------------------------------------------------------------
double
CoralFeatureStereoCorrespondence::ComputeKernelSAD(cv::Mat block_left,
                                                   cv::Mat block_right) const {

  double sad_distance = 0;
  for (int block_col = 0; block_col < kernel_size_; ++block_col) {
    for (int block_row = 0; block_row < kernel_size_; ++block_row) {
      int left_colour = block_left.at<uchar>(block_row, block_col);
      int right_colour = block_right.at<uchar>(block_row, block_col);

      sad_distance += (fabs(left_colour - right_colour));
    }
  }
  return sad_distance;
}
//------------------------------------------------------------------------------
CoralFeatureStereoCorrespondence::CoralFeatureStereoCorrespondence() {
  max_disparity_ = 50;
  kernel_size_ = 5;
}

} // namespace features
} // namespace coral
