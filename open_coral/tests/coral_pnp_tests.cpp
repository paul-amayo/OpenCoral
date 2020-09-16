#define BOOST_TEST_MODULE My Test
#include "../include/models/coral_pnp_model.h"
#include <boost/make_shared.hpp>
#include <boost/test/included/unit_test.hpp>
#include "glog/logging.h"

using namespace coral;

struct PnpModelFixture {

  features::FeatureVectorSPtr stereo_features;
  boost::shared_ptr<models::CoralPNPModel> stereo_model;
  Eigen::MatrixXd K;

  Eigen::Matrix3d R;
  Eigen::Vector3d t;

  const int n = 100;
  const double noise = 0;

  const double uc = 320;
  const double vc = 240;
  const double fu = 800;
  const double fv = 800;


  static double rand(double min, double max) {
    return min + (max - min) * double(std::rand()) / RAND_MAX;
  }

  static void RandomPose(Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    const double range = 1;

    double phi =rand(0, range * 3.14159 * 2);
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

    t(0) = 0.2f;
    t(1) = -0.1f;
    t(2) = 0.20f;
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
        R(2, 0) * point(1) + R(2, 1) * point(1) + R(2, 2) * point(2) + t(2);

    double nu = rand(-noise, +noise);
    double nv = rand(-noise, +noise);
    Eigen::Vector2d uv;
    uv(0) = uc + fu * Xc / Zc + nu;
    uv(1) =vc + fv * Yc / Zc + nv;

    return uv;
  }

  PnpModelFixture() {

    K = Eigen::MatrixXd::Zero(3, 3);
    K << fu, 0, uc, 0, fv, vc, 0, 0, 1;

    stereo_model = boost::make_shared<coral::models::CoralPNPModel>();

    stereo_features = boost::make_shared<coral::features::FeatureVector>();

    RandomPose(R, t);
    for (int i = 0; i < n; ++i) {
      Eigen::Vector3d point3d = RandomPoint();
      Eigen::Vector2d pointuv = ProjectWithNoise(R, t, point3d);

      stereo_features->push_back(
          boost::make_shared<coral::features::CoralFeatureStereoCorrespondance>(
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

  LOG(INFO)<<" R is "<<R;
  LOG(INFO)<<"T is "<<t;

  stereo_model->UpdateModel(stereo_features);

  Eigen::Matrix3d Rot=stereo_model->GetRotation();
  Eigen::Vector3d trans=stereo_model->GetTranslation();

  double rot_error,trans_error;

  stereo_model->RelativeError(rot_error,trans_error,R,t,Rot,trans);

  LOG(INFO)<<"Rot error is "<<rot_error;
  LOG(INFO)<<"Trans error is "<<trans_error;
}