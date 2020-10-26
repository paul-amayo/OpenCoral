#define BOOST_TEST_MODULE My Test
#include "../include/models/coral_photo_model.h"
#include "../open_coral/include/model_initialiser.h"
#include "glog/logging.h"
#include "opencv2/ximgproc.hpp"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>

using namespace coral;

struct PhotoModelFixture {

  boost::shared_ptr<models::CoralPhotoModel> photo_model;

  Eigen::MatrixXd K;

  const double baseline = 0.5372;

  const double uc = 240;
  const double vc = 320;

  const double fu = 800;
  const double fv = 800;

  cv::Mat image_1_left;
  cv::Mat image_2_left;

  cv::Mat image_1_right;
  cv::Mat image_2_right;

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

  PhotoModelFixture() {

    K = Eigen::MatrixXd::Zero(3, 3);
    K << fu, 0, uc, 0, fv, vc, 0, 0, 1;

    photo_model =
        boost::make_shared<coral::models::CoralPhotoModel>(K, baseline);

    image_1_left =
        cv::imread("/home/paulamayo/data/kitti/2011_09_26/"
                   "2011_09_26_drive_0039_sync/image_02/data/0000000005.png");
    image_2_left =
        cv::imread("/home/paulamayo/data/kitti/2011_09_26/"
                   "2011_09_26_drive_0039_sync/image_02/data/0000000006.png");

    image_1_right = cv::imread(
        "//home/paulamayo/data/kitti/2011_09_26/2011_09_26_drive_0039_sync"
        "/image_03/data/0000000005.png");
    image_2_right = cv::imread(
        "/home/paulamayo/data/kitti/2011_09_26/2011_09_26_drive_0039_sync"
        "/image_03/data/0000000006.png");
  }
};

BOOST_FIXTURE_TEST_CASE(image_test, PhotoModelFixture) {

  cv::Mat image_grey_1_left;
  cv::Mat image_grey_2_left;

  cv::Mat image_grey_1_right;
  cv::Mat image_grey_2_right;

  cv::cvtColor(image_1_left, image_grey_1_left, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_2_left, image_grey_2_left, cv::COLOR_BGR2GRAY);

  cv::cvtColor(image_1_right, image_grey_1_right, cv::COLOR_BGR2GRAY);
  cv::cvtColor(image_2_right, image_grey_2_right, cv::COLOR_BGR2GRAY);

  Eigen::Matrix3d R = Eigen::MatrixXd::Identity(3, 3);
  Eigen::Vector3d t(1, 0, 1);

  // Open CV SGBM
  int wsize = 5;
  int max_disp = 80;
  cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, max_disp, wsize);
  sgbm->setP1(24 * wsize * wsize);
  sgbm->setP2(96 * wsize * wsize);
  sgbm->setPreFilterCap(63);

  cv::Mat disp_1_left, disp_2_left;
  sgbm->compute(image_grey_1_left, image_grey_1_right, disp_1_left);
  sgbm->compute(image_grey_2_left, image_grey_2_right, disp_2_left);

  cv::imshow("Disp left ", DisparityColour(disp_1_left));

  cv::Mat new_image =
      photo_model->TransformImageMotion(image_1_left, disp_1_left, R, t);

  cv::imshow("Image 1 ", new_image);
  cv::imshow("Image ", image_2_left);
  cv::waitKey(0);
}