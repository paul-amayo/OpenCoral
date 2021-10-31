
// TODO: Fix includes
#include "../coral_gpu/include/coral_cuda_wrapper.h"
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

  LOG(INFO) << "This is an info  message";

  for (int i = 50; i < 100; ++i) {
    std::string directory_left(
        "/home/paulamayo/data/multi_vo/swinging_static/stereo/000");

    std::string directory_right(
        "/home/paulamayo/data/multi_vo/swinging_static/stereo/000");
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

    // Open CV SGBM
    int wsize = 7;
    int max_disp = 70;

    cv::Mat left_disp,right_disp;
    Ptr<StereoSGBM> left_matcher  = StereoSGBM::create(0,max_disp,wsize);
    left_matcher->setP1(24*wsize*wsize);
    left_matcher->setP2(96*wsize*wsize);

    left_matcher->setPreFilterCap(63);
    left_matcher->setMode(StereoSGBM::MODE_SGBM_3WAY);
    Ptr<ximgproc::DisparityWLSFilter> wls_filter = ximgproc::createDisparityWLSFilter
        (left_matcher);
    Ptr<StereoMatcher> right_matcher = ximgproc::createRightMatcher(left_matcher);

    left_matcher-> compute(image_1_left_grey,image_1_right_grey,left_disp);
    right_matcher->compute(image_1_right_grey,image_1_left_grey, right_disp);


    wls_filter->setLambda(8000);
    wls_filter->setSigmaColor(1.5);
    cv::Mat filtered_disp;
    wls_filter->filter(left_disp,image_1_left_grey,filtered_disp,right_disp);
    //cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, max_disp, wsize);
    //Ptr<StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(sgbm);

    //cv::Mat disp_1_left, disp_2_left;
    //sgbm->compute(image_1_left_grey, image_1_right_grey, disp_1_left);
    //sgbm->compute(image_2_left_grey, image_2_right_grey, disp_2_left);

    cv::imshow("Disparity left", DisparityColour(left_disp));
    cv::imshow("Disparity right", DisparityColour(right_disp));
    cv::imshow("Disparity right", DisparityColour(filtered_disp));
    cv::waitKey(0);
    //    cv::imshow("Disparity 2 left", DisparityColour(disp_2_left));
    //
    //    cv::imshow("Disp difference", cv::abs(DisparityColour(disp_2_left) -
    //                                          DisparityColour(disp_1_left)));

    // Do a binary filter

    // Get the optical flow

    cv::Mat flow_deep(image_1_left.size(), CV_32FC2);
    cv::Ptr<cv::DenseOpticalFlow> dense_flow =
        cv::optflow::createOptFlow_DeepFlow();

    dense_flow->calc(image_1_left_grey, image_2_left_grey, flow_deep);
    cv::Mat flow_parts[2];
    cv::split(flow_deep, flow_parts);

    cv::Mat magnitude, angle, magn_norm, flow_u, flow_v;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    flow_parts[0].convertTo(flow_u, CV_32F, 1);
    flow_parts[1].convertTo(flow_v, CV_32F, 1);

//    cv::imshow("Flow magnitude ", FlowColour(flow_u));
//    cv::waitKey(0);

    // Filter orb keypoints
    int num_bins = 30;
    int grid_ratio = 1;

    double grid_size_width = (float)num_bins / (float)image_1_left.cols;
    double grid_size_height = (float)num_bins / (float)image_1_left.rows;

    int total_num_bins = num_bins * num_bins * grid_ratio;
    std::vector<bool> bin_occupancy(total_num_bins, false);
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;

    int minHessian = 40;
    cv::Ptr<cv::xfeatures2d::SURF> surf =
        cv::xfeatures2d::SURF::create(minHessian);

    //    orb->setMaxFeatures(10000);
    surf->detect(image_1_left_grey, keypoints_1);

    LOG(INFO) << "Obtain the orb keypoints";

    std::vector<cv::KeyPoint> keypoints_out;

    int flow_threshold=1;
    for (const auto &keypoint : keypoints_1) {

      double u = keypoint.pt.x;
      double v = keypoint.pt.y;

      int new_u = u + flow_u.at<float>(v, u);
      int new_v = v + flow_v.at<float>(v, u);

      cv::KeyPoint flow_point(new_u, new_v, 1);

      int col_bin_index = floor(u * grid_ratio * grid_size_width);
      int row_bin_index = floor(v * grid_size_height);

      int vector_index = row_bin_index * grid_ratio * num_bins + col_bin_index;

      if (vector_index > 0) {
        // if the bin is empty add a feature and then fill it
        if (!bin_occupancy[vector_index] && magnitude.at<float>(v,u)>flow_threshold) {
          keypoints_out.push_back(keypoint);
          keypoints_2.push_back(flow_point);
          bin_occupancy[vector_index] = true;
        }
      }
    }
    keypoints_1=keypoints_out;


    LOG(INFO) << "size of keypoints is " << keypoints_1.size();
    LOG(INFO) << "size of keypoints is " << keypoints_2.size();

    cv::Mat static_image = cv::abs(image_1_left_grey - image_2_left_grey) > 10;

    const double baseline = 0.24;

    const double uc = 643;
    const double vc = 482;

    const double fu = 969.4750;
    const double fv = 969.475;

    Eigen::Matrix3d K;
    K << fu, 0, uc, 0, fv, vc, 0, 0, 1;

    cv::Mat image_drawn_1, image_drawn_2;
    cv::drawKeypoints(image_1_left, keypoints_1, image_drawn_1);
    cv::drawKeypoints(image_2_left, keypoints_2, image_drawn_2);

    cv::Mat image_1_2 = cv::Mat(image_1_left.rows, 2 * image_1_left.cols,
                                CV_8UC3, cv::Scalar(80, 80, 80));
    cv::Mat MatROI =
        image_1_2(cv::Rect(0, 0, image_1_left.cols, image_1_left.rows));

    image_drawn_1.copyTo(MatROI);
    MatROI = image_1_2(
        cv::Rect(image_1_left.cols, 0, image_1_left.cols, image_1_left.rows));
    image_drawn_2.copyTo(MatROI);

    cv::imshow("Image features", image_1_2);
    cv::waitKey(0);

    features::FeatureVectorSPtr image_features =
        boost::make_shared<coral::features::FeatureVector>();

    std::vector<cv::KeyPoint> keypoints_2_left;
    std::vector<cv::KeyPoint> keypoints_2_right;
    int match_no = 0;
    for (int keypoint_no = 0; keypoint_no < keypoints_1.size(); ++keypoint_no) {
      Eigen::Vector2d uv_1_left(keypoints_1[keypoint_no].pt.x,
                                keypoints_1[keypoint_no].pt.y);

      Eigen::Vector2d uv_2_left(keypoints_2[keypoint_no].pt.x,
                                keypoints_2[keypoint_no].pt.y);

      features::CoralFeatureStereoCorrespondenceSPtr new_feature =
          boost::make_shared<coral::features::CoralFeatureStereoCorrespondence>(
              uv_1_left, uv_2_left, image_2_left_grey, image_2_right_grey,
              baseline, K);

      cv::KeyPoint key_left;
      key_left.size = 1;
      key_left.pt.x = uv_2_left(0);
      key_left.pt.y = uv_2_left(1);

      cv::KeyPoint key_right;
      key_right.size = 1;
      key_right.pt.x = uv_2_left(0) - new_feature->GetDisparity();
      key_right.pt.y = uv_2_left(1);

      if (new_feature->GetDisparity() != 0) {
        image_features->push_back(new_feature);
        keypoints_2_left.push_back(key_left);
        keypoints_2_right.push_back(key_right);
      }

      match_no++;
    }

    cv::Mat image_left_drawn, image_right_drawn;
    cv::drawKeypoints(image_2_left, keypoints_2_left, image_left_drawn);
    cv::drawKeypoints(image_2_right, keypoints_2_right, image_right_drawn);

    cv::Mat left_right = cv::Mat(image_1_left.rows, 2 * image_1_left.cols,
                                 CV_8UC3, cv::Scalar(80, 80, 80));
    MatROI = left_right(cv::Rect(0, 0, image_1_left.cols, image_1_left.rows));

    image_drawn_1.copyTo(MatROI);
    MatROI = left_right(
        cv::Rect(image_1_left.cols, 0, image_1_left.cols, image_1_left.rows));
    image_left_drawn.copyTo(MatROI);

    models::ModelInitialiserParams mi_params(100, 0.95);
    models::ModelInitialiser<models::CoralPNPModel> pnp_model_initialiser(
        mi_params);
    pnp_model_initialiser.SetCameraMatrix(K);

    int num_models = 6;
    float threshold = 3.0;

    models::ModelVectorSPtr pnp_models(new models::ModelVector);
    pnp_model_initialiser.Initialise(image_features, num_models, threshold,
                                     pnp_models);

    for (const auto &model : *pnp_models) {
      model->ModelEquation();
    }

    coral::optimiser::CoralOptimiserParams params{};

    params.num_features = image_features->size();
    params.num_neighbours = 2;
    params.outlier_threshold = 3;

    params.num_labels = pnp_models->size() + 1;
    params.num_iterations = 10;
    params.num_loops = 1;

    params.lambda = 0;
    params.nu = 0;

    params.beta = 1;
    params.alpha = 1;
    params.tau = 1;

    CoralOptimiser<CoralPNPModel> optimiser(params);
    LOG(INFO) << "Number of  models is " << pnp_models->size();
    EnergyMinimisationResult result =
        optimiser.EnergyMinimisation(image_features, pnp_models);

    std::vector<cv::Scalar> colour_map;
    colour_map.push_back(cv::Scalar(0, 0, 255));
    colour_map.push_back(cv::Scalar(0, 255, 0));
    colour_map.push_back(cv::Scalar(255, 0, 0));
    colour_map.push_back(cv::Scalar(0, 255, 255));
    colour_map.push_back(cv::Scalar(255, 0, 255));

    int num_rows = image_1_left.rows;
    int num_cols = image_1_left.cols;

    cv::Mat image_label =
        cv::Mat(num_rows, num_cols, CV_8UC3, cv::Scalar(80, 80, 80));

    cv::Mat image_cost = cv::Mat(num_rows, num_cols, CV_32F, cv::Scalar(350));

    int model_no = 0;

    for (const auto &model : *pnp_models) {
      model->ModelEquation();

      boost::shared_ptr<models::CoralPNPModel> pnp_model =
          boost::dynamic_pointer_cast<models::CoralPNPModel>(model);
      boost::shared_ptr<models::CoralPhotoModel> photo_model =
          boost::make_shared<coral::models::CoralPhotoModel>(K, baseline);

      Eigen::Matrix3d R = pnp_model->GetRotation();
      Eigen::Vector3d t = pnp_model->GetTranslation();

      cv::Mat new_image =
          photo_model->TransformImageMotion(image_1_left, left_disp, R, t);

      // Assign pixelwise cost
      for (int u = 0; u < num_cols; ++u) {
        for (int v = 0; v < num_rows; ++v) {

          float prev_cost = image_cost.at<float>(v, u);
          cv::Vec3b new_image_cost = new_image.at<cv::Vec3b>(v, u);

          float curr_cost =
              (new_image_cost(0) + new_image_cost(1) + new_image_cost(2)) / 3.0;
          if (u == 150 && v == 150) {
            LOG(INFO) << "Prev cost is  " << prev_cost;
            LOG(INFO) << "Curr cost is  " << curr_cost;
          }

          if (prev_cost > curr_cost) {

            image_label.at<cv::Vec3b>(v, u) =
                cv::Vec3b(colour_map[model_no](0), colour_map[model_no](1),
                          colour_map[model_no](2));
            image_cost.at<float>(v, u) = curr_cost;
          }
        }
      }

      cv::imshow("Motion model projection", cv::abs(image_2_left - new_image));
      cv::imshow("PhotoModel", image_label);

      cv::waitKey(0);

      model_no++;
    }

    viewer::CoralViewer<cv::Mat> image_viewer;

    image_viewer.VisualiseLabels(image_1_left, image_features, pnp_models,
                                 result.DiscreteLabel);

    LOG(INFO) << "Primal is \n" << result.SoftLabel;
    LOG(INFO) << "Label is \n" << result.DiscreteLabel;

    // cv::waitKey(0);
  }
  return 0;
}
