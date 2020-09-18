#define BOOST_TEST_MODULE My Test
#include "../include/models/coral_pnp_model.h"
#include "../open_coral/include/model_initialiser.h"
#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace coral;

struct PnpModelFixture {

  features::FeatureVectorSPtr stereo_features;

  features::FeatureVectorSPtr image_features;

  boost::shared_ptr<models::CoralPNPModel> stereo_model;

  Eigen::MatrixXd K;

  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  const int n = 1000;
  const double noise = 3;

  const double baseline = 0.5372;

  const double uc = 240;
  const double vc = 320;

  const double fu = 800;
  const double fv = 800;

  cv::Mat image_1_left;
  cv::Mat image_2_left;

  cv::Mat image_1_right;
  cv::Mat image_2_right;

  static double rand(double min, double max) {
    return min + (max - min) * double(std::rand()) / RAND_MAX;
  }

  static void RandomPose(Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    const double range = 1;

    double phi = rand(0, range * 3.14159 * 2);
    double theta = rand(0, range * 3.14159);
    double psi = rand(0, range * 3.14159 * 2);

    R(0, 0) = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
    R(0, 1) = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
    R(0, 2) = sin(psi) * sin(theta);

    R(1, 0) = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
    R(1, 1) = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
    R(1, 2) = cos(psi) * sin(theta);

    R(2, 0) = sin(theta) * sin(phi);
    R(2, 1) = -sin(theta) * cos(phi);
    R(2, 2) = cos(theta);

    t(0) = 5.4f;
    t(1) = -2.0f;
    t(2) = 0.8f;
  }

  Eigen::Vector3d RandomPoint() {
    double theta = rand(0, 3.14159), phi = rand(0, 2 * 3.14159),
           R = rand(0, +2);

    Eigen::Vector3d point;
    point(0) = sin(theta) * sin(phi) * R;
    point(1) = -sin(theta) * cos(phi) * R;
    point(2) = cos(theta) * R;

    return point;
  }

  Eigen::Vector2d ProjectWithNoise(Eigen::Matrix3d R, Eigen::Vector3d t,
                                   Eigen::Vector3d point) {
    double Xc =
        R(0, 0) * point(0) + R(0, 1) * point(1) + R(0, 2) * point(2) + t(0);
    double Yc =
        R(1, 0) * point(0) + R(1, 1) * point(1) + R(1, 2) * point(2) + t(1);
    double Zc =
        R(2, 0) * point(0) + R(2, 1) * point(1) + R(2, 2) * point(2) + t(2);

    double nu = rand(-noise, +noise);
    double nv = rand(-noise, +noise);

    Eigen::Vector2d uv;
    uv(0) = uc + fu * Xc / Zc + nu;
    uv(1) = vc + fv * Yc / Zc + nv;

    return uv;
  }

  PnpModelFixture() {

    K = Eigen::MatrixXd::Zero(3, 3);
    K << fu, 0, uc, 0, fv, vc, 0, 0, 1;

    stereo_model = boost::make_shared<coral::models::CoralPNPModel>();

    stereo_features = boost::make_shared<coral::features::FeatureVector>();

    image_features = boost::make_shared<coral::features::FeatureVector>();

    RandomPose(R, t);

    image_1_left = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_02/data/0000000004.png");
    image_2_left = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_02/data/0000000005.png");

    image_1_right = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_03/data/0000000004.png");
    image_2_right = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_03/data/0000000005.png");
    for (int i = 0; i < n; ++i) {
      Eigen::Vector3d point3d = RandomPoint();
      Eigen::Vector2d pointuv = ProjectWithNoise(R, t, point3d);

      stereo_features->push_back(
          boost::make_shared<coral::features::CoralFeatureStereoCorrespondence>(
              pointuv, point3d));
    }
  }
};

BOOST_FIXTURE_TEST_CASE(camera_params, PnpModelFixture) {

  stereo_model->SetCameraParams(K);
  Eigen::MatrixXd K_out = stereo_model->GetCameraParams();

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      // BOOST_CHECK_EQUAL(K(i,j),K_out(i,j));
    }
  }
}

BOOST_FIXTURE_TEST_CASE(update_model, PnpModelFixture) {
  stereo_model->SetCameraParams(K);

  LOG(INFO) << " R is " << R;
  LOG(INFO) << "T is " << t;

  stereo_model->UpdateModel(stereo_features);

  Eigen::Matrix3d Rot = stereo_model->GetRotation();
  Eigen::Vector3d trans = stereo_model->GetTranslation();

  double rot_error, trans_error;

  stereo_model->RelativeError(rot_error, trans_error, R, t, Rot, trans);

  LOG(INFO) << "Rot error is " << rot_error;
  LOG(INFO) << "Trans error is " << trans_error;
}

BOOST_FIXTURE_TEST_CASE(model_initialiser, PnpModelFixture) {
  models::ModelInitialiserParams mi_params(10, 0.95);
  models::ModelInitialiser<models::CoralPNPModel> pnp_model_initialiser(
      mi_params);

  int num_models = 1;
  float threshold = 3.0;

  models::ModelVectorSPtr pnp_models(new models::ModelVector);
  pnp_model_initialiser.Initialise(stereo_features, num_models, threshold,
                                   pnp_models);

  pnp_models->at(0)->ModelEquation();
}
BOOST_FIXTURE_TEST_CASE(image_test, PnpModelFixture) {

  cv::Mat image_grey_1_left;
  cv::Mat image_grey_2_left;

  cv::Mat image_grey_1_right;
  cv::Mat image_grey_2_right;

  cv::cvtColor(image_1_left, image_grey_1_left, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_2_left, image_grey_2_left, cv::COLOR_BGR2GRAY);

  cv::cvtColor(image_1_right, image_grey_1_right, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_2_right, image_grey_2_right, cv::COLOR_BGR2GRAY);

  std::vector<cv::KeyPoint> keypoints_1;
  cv::Mat descriptors_1;
  std::vector<cv::KeyPoint> keypoints_2;
  cv::Mat descriptors_2;

  cv::ORB orb(1000);
  orb.detect(image_grey_1_left, keypoints_1);

  // Filter orb keypoints
  int num_bins = 20;
  int grid_ratio = 3;

  int total_num_bins = num_bins * num_bins * grid_ratio;
  std::vector<bool> bin_occupancy(total_num_bins, false);
  std::vector<cv::KeyPoint> keypoints_out;
  for (const auto& keypoint : keypoints_1) {

    double u = keypoint.pt.x;
    double v = keypoint.pt.y;

    double grid_size_width = (float)num_bins / (float)image_1_left.cols;
    double grid_size_height = (float)num_bins / (float)image_1_left.rows;

    int col_bin_index = floor(u * grid_ratio * grid_size_width);
    int row_bin_index = floor(v * grid_size_height);

    int vector_index = row_bin_index * grid_ratio * num_bins + col_bin_index;

    if (vector_index > 0) {
      // if the bin is empty andd a feature and then fill it
      if (!bin_occupancy[vector_index]) {
        keypoints_out.push_back(keypoint);
        bin_occupancy[vector_index] = true;
      }
    }
  }

  keypoints_1 = keypoints_out;
  // Get Descriptor
  orb.compute(image_grey_1_left, keypoints_1, descriptors_1);

  orb.detect(image_grey_2_left, keypoints_2);
  orb.compute(image_grey_2_left, keypoints_2, descriptors_2);

  cv::Mat image_drawn_1, image_drawn_2;
  cv::drawKeypoints(image_1_left, keypoints_1, image_drawn_1);
  cv::drawKeypoints(image_2_left, keypoints_2, image_drawn_2);


  // Since SURF is a floating-point descriptor NORM_L2 is used
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create("BruteForce-Hamming");
  std::vector<std::vector<cv::DMatch>> knn_matches;
  matcher->knnMatch(descriptors_1, descriptors_2, knn_matches, 2);
  //-- Filter matches using the Lowe's ratio test
  const float ratio_thresh = 0.7f;
  std::vector<cv::DMatch> good_matches;
  for (auto &knn_match : knn_matches) {
    if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
      good_matches.push_back(knn_match[0]);
    }
  }
  LOG(INFO) << "NUmber good matches is " << good_matches.size();

  int match_no = 0;
  for (auto &good_match : good_matches) {
    Eigen::Vector2d uv_1_left(keypoints_1[good_match.queryIdx].pt.x,
                              keypoints_1[good_match.queryIdx].pt.y);

    Eigen::Vector2d uv_2_left(keypoints_2[good_match.trainIdx].pt.x,
                              keypoints_2[good_match.trainIdx].pt.y);

    features::CoralFeatureStereoCorrespondenceSPtr new_feature =
        boost::make_shared<coral::features::CoralFeatureStereoCorrespondence>(
            uv_1_left, uv_2_left, image_grey_2_left, image_grey_2_right,
            baseline, K);

    if (new_feature->GetDisparity() != 0)
      image_features->push_back(new_feature);

    match_no++;
  }

  models::ModelInitialiserParams mi_params(100, 0.95);
  models::ModelInitialiser<models::CoralPNPModel> pnp_model_initialiser(
      mi_params);

  int num_models = 2;
  float threshold = 3.0;

  models::ModelVectorSPtr line_models(new models::ModelVector);
  pnp_model_initialiser.Initialise(image_features, num_models, threshold,
                                   line_models);

  for (auto model:*line_models){
    model->ModelEquation();
  }

}