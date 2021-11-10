
// TODO: Fix includes
#include "../coral_gpu/include/coral_cuda_wrapper.h"
#include "../open_coral/include/extractors/coral_curvature_extractor.h"
#include "../open_coral/include/model_initialiser.h"
#include "../open_coral/include/models/coral_pnp_model.h"
#include "../open_coral/include/viewer/coral_viewer.h"
#include "opencv2/optflow.hpp"
#include "opencv2/ximgproc.hpp"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <include/models/coral_photo_model.h>

using namespace coral::features;
using namespace coral;
using namespace coral::models;
using namespace coral::optimiser;
using namespace coral::extractors;
using namespace cv;

cv::Mat DisparityColour(const cv::Mat &disp_grey) {

  cv::Mat disp_vis;
  cv::ximgproc::getDisparityVis(disp_grey, disp_vis, 1);

  double disparity_min, disparity_max;
  cv::minMaxIdx(disp_vis, &disparity_min, &disparity_max);
  double range = disparity_max - disparity_min;
  LOG(INFO) << "Disparity max is " << disparity_max << " with min "
            << disparity_min;
  cv::Mat disparity_normalised;
  disp_vis.convertTo(disparity_normalised, CV_8U, 255 / range,
                     -disparity_min / range);
  cv::Mat disp_colour;
  cv::applyColorMap(disparity_normalised, disp_colour, cv::COLORMAP_JET);
  disp_colour.convertTo(disp_colour, CV_32FC3, 1 / 255.0);
  return disp_colour;
}

cv::Mat FlowColour(const cv::Mat &flow) {

  double flow_min, flow_max;
  cv::minMaxIdx(flow, &flow_min, &flow_max);
  double range = flow_max - flow_min;
  LOG(INFO) << "Flow max is " << flow_max << " with min " << flow_min;
  cv::Mat flow_normalised;
  flow.convertTo(flow_normalised, CV_8U, 255 / range, -flow_min / range);
  cv::Mat flow_colour;
  cv::applyColorMap(flow_normalised, flow_colour, cv::COLORMAP_JET);
  flow_colour.convertTo(flow_colour, CV_32FC3, 1 / 255.0);
  return flow_colour;
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  ExtractorParams params;

  CurvatureExtractor curv_extractor(params);

  LOG(INFO) << "This is an info  message";

  for (int i = 101; i < 800; ++i) {
    std::string directory_left(
        "/home/paulamayo/data/multi_vo/swinging_dynamic/stereo/000");

    std::string directory_right(
        "/home/paulamayo/data/multi_vo/swinging_dynamic/stereo/000");
    std::string zero_pad;

    if (i < 100) {
      zero_pad = "0";
    }

    if (i < 10) {
      zero_pad = "00";
    }

    int delta = 1;
    LOG(INFO) << directory_left + zero_pad + std::to_string(i) + "_left.png";
    cv::Mat image_1_left;
    cv::resize(cv::imread(directory_left + zero_pad + std::to_string(i) +
                          "_left"
                          ".png"),
               image_1_left, cv::Size(), 0.5, 0.5);

    cv::Mat image_1_right;
    cv::resize(cv::imread(directory_right + zero_pad + std::to_string(i) +
                          "_right.png"),
               image_1_right, cv::Size(), 0.5, 0.5);

    cv::Mat image_2_left;
    cv::resize(cv::imread(directory_left + zero_pad +
                          std::to_string(i + delta) +
                          "_left"
                          ".png"),
               image_2_left, cv::Size(), 0.5, 0.5);

    cv::Mat image_2_right;
    cv::resize(cv::imread(directory_right + zero_pad +
                          std::to_string(i + delta) +
                          "_right"
                          ".png"),
               image_2_right, cv::Size(), 0.5, 0.5);

    cv::Mat image_1_left_grey, image_1_right_grey, image_2_left_grey,
        image_2_right_grey;
    cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_RGB2GRAY);
    cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_RGB2GRAY);

    cv::cvtColor(image_2_left, image_2_left_grey, cv::COLOR_RGB2GRAY);
    cv::cvtColor(image_2_right, image_2_right_grey, cv::COLOR_RGB2GRAY);

    FeatureVectorSPtr curv_features =
        curv_extractor.ExtractFeatures(image_1_left_grey);
    std::vector<cv::Point2f> points_0, points_1;
    for (auto feat : *curv_features) {
      points_0.push_back(
          cv::Point2f(feat->GetFeatureValue()(0), feat->GetFeatureValue()(1)));
    }

    cv::Ptr<cv::SparseOpticalFlow> sparse_flow =
        cv::SparsePyrLKOpticalFlow::create();

    std::vector<uchar> status;
    std::vector<float> err;
    TermCriteria criteria =
        TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    sparse_flow->calc(image_1_left_grey, image_2_left_grey, points_0, points_1,
                      status, err);
    // Create some random colors
    std::vector<Scalar> colors;
    RNG rng;
    for(int ii = 0; ii < points_0.size(); ii++)
    {
      int r = rng.uniform(0, 256);
      int g = rng.uniform(0, 256);
      int b = rng.uniform(0, 256);
      colors.push_back(Scalar(r,g,b));
    }

    // Create a mask image
    cv::Mat mask = cv::Mat::zeros(image_1_left.size(), image_1_left
                                                                 .type());

    std::vector<cv::Point2f> good_new;
    for(uint i = 0; i < points_0.size(); i++)
    {
      // Select good points
      if(status[i] == 1) {
        good_new.push_back(points_1[i]);
        // draw the tracks
        cv::line(mask,points_1[i], points_0[i], colors[i], 2);
        //cv::circle(image_2_left, points_1[i], 5, colors[i], -1);
      }
    }
    cv::Mat img;
    cv::add(image_2_left, mask, img);
    cv::imshow("Frame", img);
    cv::waitKey(15);

//    // Get the optical flow
//    cv::Mat flow_deep(image_1_left.size(), CV_32FC2);
//    cv::Ptr<cv::DenseOpticalFlow> dense_flow =
//        cv::optflow::createOptFlow_DeepFlow();
//
//    dense_flow->calc(image_1_left_grey, image_2_left_grey, flow_deep);
//    cv::Mat flow_parts[2];
//    cv::split(flow_deep, flow_parts);
//
//    cv::Mat magnitude, angle, magn_norm, flow_u, flow_v;
//    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
//    flow_parts[0].convertTo(flow_u, CV_32F, 1);
//    flow_parts[1].convertTo(flow_v, CV_32F, 1);
//
//    curv_extractor.ExtractFeatures(image_1_left_grey);
//
//    cv::imshow("Image", image_1_left);
//    cv::imshow("Flow magnitude ", FlowColour(flow_u));
//    cv::waitKey(15);
  }
  return 0;
}
