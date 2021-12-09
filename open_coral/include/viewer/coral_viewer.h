
#ifndef CORAL_VIEWER_H_
#define CORAL_VIEWER_H_

#include "../features/coral_feature_point.h"
#include "../features/coral_feature_stereo_correspondence.h"
#include "../models/coral_model.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace coral {
namespace viewer {

struct CoralViewerParams {
  Eigen::MatrixXd K_;
  float baseline_;

  CoralViewerParams() {}

  CoralViewerParams(Eigen::MatrixXd K, float baseline)
      : K_(K), baseline_(baseline){};
};

template <typename T> class CoralViewer {
public:
  CoralViewer();

  CoralViewer(CoralViewerParams params);

  ~CoralViewer() = default;

  void VisualiseLabels(const T &input,
                       const features::FeatureVectorSPtr &features,
                       const coral::models::ModelVectorSPtr &models,
                       const Eigen::MatrixXd &labels);

private:
  void CreateColourMap();

  CoralViewerParams params_;
  std::vector<cv::Scalar> colour_map;
};

//------------------------------------------------------------------------------
template <typename InputType> CoralViewer<InputType>::CoralViewer() {
  CreateColourMap();
}
//------------------------------------------------------------------------------
template <typename InputType>
CoralViewer<InputType>::CoralViewer(CoralViewerParams params)
    : params_(params) {
  CreateColourMap();
}
//------------------------------------------------------------------------------
template <typename InputType> void CoralViewer<InputType>::CreateColourMap() {

  colour_map.push_back(cv::Scalar(0, 0, 255));
  colour_map.push_back(cv::Scalar(0, 255, 0));
  colour_map.push_back(cv::Scalar(255, 0, 0));
  colour_map.push_back(cv::Scalar(0, 255, 255));
  colour_map.push_back(cv::Scalar(255, 0, 255));
  colour_map.push_back(cv::Scalar(255, 255, 0));
  colour_map.push_back(cv::Scalar(0, 0, 128));
  colour_map.push_back(cv::Scalar(0, 128, 0));
  colour_map.push_back(cv::Scalar(128, 0, 255));
  colour_map.push_back(cv::Scalar(0, 128, 128));
  colour_map.push_back(cv::Scalar(128, 0, 128));
  colour_map.push_back(cv::Scalar(128, 128, 0));
  colour_map.push_back(cv::Scalar(0, 0, 64));
  colour_map.push_back(cv::Scalar(0, 64, 0));
  colour_map.push_back(cv::Scalar(64, 0, 0));
  colour_map.push_back(cv::Scalar(0, 64, 64));
  colour_map.push_back(cv::Scalar(64, 0, 64));
  colour_map.push_back(cv::Scalar(64, 64, 0));
  colour_map.push_back(cv::Scalar(128, 0, 255));
  colour_map.push_back(cv::Scalar(0, 128, 255));
  colour_map.push_back(cv::Scalar(0, 255, 128));
  colour_map.push_back(cv::Scalar(128, 0, 255));
  colour_map.push_back(cv::Scalar(255, 0, 128));
}
//------------------------------------------------------------------------------
template <typename InputType>
void CoralViewer<InputType>::VisualiseLabels(
    const InputType &input, const features::FeatureVectorSPtr &features,
    const models::ModelVectorSPtr &models, const Eigen::MatrixXd &labels) {
  LOG(ERROR) << "No implementation for current input type";
}
//------------------------------------------------------------------------------
template <>
void CoralViewer<cv::Mat>::VisualiseLabels(
    const cv::Mat &input, const features::FeatureVectorSPtr &features,
    const models::ModelVectorSPtr &models, const Eigen::MatrixXd &labels) {

  cv::Mat image_display = input;

  int feature_no = 0;
  for (auto &feature : *features) {
    cv::KeyPoint curr_keypoint;
    curr_keypoint.size = 1;
    Eigen::Vector2d uv = feature->GetFeatureValue();
    cv::Point2f curr_point(uv(0), uv(1));

    // if (labels(feature_no) < models->size()) {
    cv::circle(image_display, curr_point, 3, colour_map[labels(feature_no)], -1,
               cv::FILLED);
    //}

    feature_no++;
  }

  cv::resize(image_display, image_display, cv::Size(), 0.5, 0.5);
  cv::imshow("Label visualiser", image_display);
  cv::waitKey(0);
}

} // namespace viewer
} // namespace coral

#endif // CORAL_VIEWER_H_