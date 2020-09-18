#define BOOST_TEST_MODULE My Test
#include "../include/features/coral_feature_stereo_correspondence.h"
#include "glog/logging.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace coral;

struct StereoCorrespondenceFixture {

  features::CoralFeatureStereoCorrespondenceSPtr stereo_feature;

  const double baseline = 0.5372;
  Eigen::MatrixXd K;

  const double uc = 607.1928;
  const double vc = 185.2157;

  const double fu = 718.856;
  const double fv = 718.856;

  cv::Mat image_1_left;
  cv::Mat image_2_left;

  cv::Mat image_1_right;
  cv::Mat image_2_right;

  StereoCorrespondenceFixture() {

    K = Eigen::MatrixXd::Zero(3, 3);
    K << fu, 0, uc, 0, fv, vc, 0, 0, 1;

    stereo_feature =
        boost::make_shared<coral::features::CoralFeatureStereoCorrespondence>();

    image_1_left = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_02/data/0000000004.png");
    image_2_left = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_02/data/0000000005.png");

    image_1_right = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_03/data/0000000004.png");
    image_2_right = cv::imread(
        "/Users/paulamayo/code/data/kitti/kitti/image_03/data/0000000005.png");
  }
};

BOOST_FIXTURE_TEST_CASE(DisparityComputation, StereoCorrespondenceFixture) {

  cv::Mat image_grey_1_left;
  cv::Mat image_grey_2_left;

  cv::Mat image_grey_1_right;
  cv::Mat image_grey_2_right;

  cv::cvtColor(image_1_left, image_grey_1_left, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_2_left, image_grey_2_left, cv::COLOR_BGR2GRAY);

  cv::cvtColor(image_1_right, image_grey_1_right, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_2_right, image_grey_2_right, cv::COLOR_BGR2GRAY);

  // calculate disparity
  cv::StereoSGBM sgbm;

  int wsize = 3;
  int max_disp = 80;

  cv::StereoSGBM left_matcher(
      0, max_disp, wsize);

  //  = cv::StereoSGBM::create(0,max_disp,wsize);
                           //  left_matcher.setP1(24*wsize*wsize);
                           //  left_matcher.setP2(96*wsize*wsize);
                           //  left_matcher.setPreFilterCap(63);
                           //
                           //  left_matcher.P1=24*wsize*wsize;
                           //  left_matcher.P2=96*wsize*wsize;
                           //  left_matcher.preFilterCap=63;

  cv::Mat left_disp;
  left_matcher.operator()(image_grey_2_left, image_grey_2_right, left_disp);

  double disparity_min, disparity_max;
  cv::minMaxIdx(left_disp, &disparity_min, &disparity_max);
  double range = disparity_max - disparity_min;
  cv::Mat disparity_normalised;
  left_disp.convertTo(disparity_normalised, CV_8U, 255 / range,
                      -disparity_min / range);

  cv::Mat disp_char;
  left_disp.convertTo(disp_char,CV_32F);

  cv::Mat disparity_colour;
  cv::applyColorMap(disparity_normalised, disparity_colour, cv::COLORMAP_JET);
  disparity_colour.convertTo(disparity_colour, CV_32FC3, 1 / 255.0);
  //cv::addWeighted(image_2_left, 0.3, disparity_colour, 0.7, 0,
  //                disparity_colour);

  cv::imshow("Disparity Image", disparity_colour);
  cv::waitKey(0);

  std::vector<cv::KeyPoint> keypoints_1;
  cv::Mat descriptors_1;
  std::vector<cv::KeyPoint> keypoints_2;
  cv::Mat descriptors_2;

  cv::ORB orb(50);
  orb.detect(image_grey_1_left, keypoints_1);
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

  for (auto &good_match : good_matches) {
    Eigen::Vector2d uv_1_left(keypoints_1[good_match.trainIdx].pt.x,
                              keypoints_1[good_match.trainIdx].pt.y);

    Eigen::Vector2d uv_2_left(keypoints_2[good_match.queryIdx].pt.x,
                              keypoints_2[good_match.queryIdx].pt.y);
    features::CoralFeatureStereoCorrespondence new_feature(
        uv_1_left, uv_2_left, image_grey_2_left, image_grey_2_right, baseline,
        K);
    LOG(INFO) << "Point 3d is " << new_feature.GetPoint3d().transpose();
    LOG(INFO) << " Disparity coral is " << new_feature.GetDisparity();
    LOG(INFO) << "Disparity BM is "
              << disp_char.at<float>(keypoints_2[good_match.queryIdx].pt)/16;

    //LOG(INFO)<<"Disp difference is "<<disp_char.at<uchar>(keypoints_2[good_match.queryIdx].pt)/16-new_feature.GetDisparity();
  }
}
