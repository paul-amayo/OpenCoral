#include "../include/models/coral_pnp_model.h"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include <opencv2/core/eigen.hpp>

namespace coral {
namespace models {
//------------------------------------------------------------------------------
CoralPNPModel::CoralPNPModel(Eigen::Matrix3d K) {
  f_u_ = K(0, 0);
  u_c_ = K(0, 2);
  f_v_ = K(1, 1);
  v_c_ = K(1, 2);
}
//------------------------------------------------------------------------------
void CoralPNPModel::SetCameraParams(Eigen::MatrixXd K) {
  f_u_ = K(0, 0);
  u_c_ = K(0, 2);
  f_v_ = K(1, 1);
  v_c_ = K(1, 2);
}
//------------------------------------------------------------------------------
Eigen::MatrixXd CoralPNPModel::GetCameraParams() const {
  Eigen::Matrix3d K = Eigen::MatrixXd::Zero(3, 3);
  K << f_u_, 0, u_c_, 0, f_v_, v_c_, 0, 0, 1;
  return K;
}
//------------------------------------------------------------------------------
Eigen::MatrixXd
CoralPNPModel::EvaluateCost(const features::FeatureVectorSPtr &features) {

  Eigen::MatrixXd point_values(features->size(), 1);
  int feature_no = 0;

  for (const auto &feature : *features) {
    features::CoralFeatureStereoCorrespondenceSPtr stereo_feature =
        boost::dynamic_pointer_cast<features::CoralFeatureStereoCorrespondence>(
            feature);

    Eigen::Vector3d point_world = stereo_feature->GetPoint3d();
    Eigen::Vector2d point_uv = stereo_feature->GetPointUV();

    Eigen::Vector3d point_projected=R_curr_*point_world+t_curr_;

    double Xc = point_projected(0);
    double Yc = point_projected(1);
    double inv_Zc = 1.0 / point_projected(2);

    double ue = u_c_ + f_u_ * Xc * inv_Zc;
    double ve = v_c_ + f_v_ * Yc * inv_Zc;
    double u = point_uv(0);
    double v = point_uv(1);

    point_values(feature_no) = sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
    feature_no++;
  }

  return point_values;
}
//------------------------------------------------------------------------------
void CoralPNPModel::UpdateModel(const features::FeatureVectorSPtr &features) {

  int num_features = features->size();

  // Cast features to derived class
  features::CoralFeatureStereoCorrespondenceVectorSPtr stereo_features(
      new features::CoralFeatureStereoCorrespondenceVector);

  std::vector<cv::Point3d> points3d;
  std::vector<cv::Point2d> points2d;

  cv::Mat K_matrix =
      cv::Mat::zeros(3, 3, CV_64FC1); // intrinsic camera parameters
  K_matrix.at<double>(0, 0) = f_u_;   //      [ fx   0  cx ]
  K_matrix.at<double>(1, 1) = f_v_;   //      [  0  fy  cy ]
  K_matrix.at<double>(0, 2) = u_c_;   //      [  0   0   1 ]
  K_matrix.at<double>(1, 2) = v_c_;
  K_matrix.at<double>(2, 2) = 1;
  cv::Mat R_matrix = cv::Mat::zeros(3, 3, CV_64FC1); // rotation matrix
  cv::Mat t_matrix = cv::Mat::zeros(3, 1, CV_64FC1);

  for (const auto &feature : *features) {
    features::CoralFeatureStereoCorrespondenceSPtr stereo_feature =
        boost::dynamic_pointer_cast<features::CoralFeatureStereoCorrespondence>(
            feature);
    stereo_features->push_back(stereo_feature);

    points3d.emplace_back(stereo_feature->GetPoint3d()(0),
                                   stereo_feature->GetPoint3d()(1),
                                   stereo_feature->GetPoint3d()(2));

    points2d.emplace_back(stereo_feature->GetPointUV()(0),
                                   stereo_feature->GetPointUV()(1));
  }

  cv::Mat distCoeffs =
      cv::Mat::zeros(4, 1, CV_64FC1); // vector of distortion coefficients
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1); // output rotation vector
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1); // output translation vector
  bool useExtrinsicGuess =
      false; // if true the function uses the provided rvec and tvec values as
  // initial approximations of the rotation and translation vectors

  if (num_features >= 4 ) {
    cv::solvePnP(points3d, points2d, K_matrix, distCoeffs, rvec, tvec,
                 useExtrinsicGuess, cv::SOLVEPNP_EPNP);
    cv::Rodrigues(rvec, R_matrix); // converts Rotation Vector to Matrix
    t_matrix = tvec;               // set translation matrix

    R_curr_=Eigen::MatrixXd::Zero(3,3);
    t_curr_=Eigen::MatrixXd::Zero(3,1);
    cv::cv2eigen(R_matrix,R_curr_);
    cv::cv2eigen(t_matrix,t_curr_);
  }

}

//------------------------------------------------------------------------------
void CoralPNPModel::RelativeError(double &rot_err, double &transl_err,
                                  const Rot& R_true, Trans T_true,
                                  const Rot& R_est, Trans T_est) {
  Eigen::Vector4d q_true, q_est;

  MatToQuat(R_true, q_true);

  MatToQuat(R_est, q_est);

  double rot_err1 = sqrt((q_true(0) - q_est[0]) * (q_true[0] - q_est[0]) +
                         (q_true(1) - q_est[1]) * (q_true[1] - q_est[1]) +
                         (q_true(2) - q_est[2]) * (q_true[2] - q_est[2]) +
                         (q_true(3) - q_est[3]) * (q_true[3] - q_est[3])) /
                    sqrt(q_true(0) * q_true[0] + q_true[1] * q_true[1] +
                         q_true[2] * q_true[2] + q_true[3] * q_true[3]);

  double rot_err2 = sqrt((q_true[0] + q_est[0]) * (q_true[0] + q_est[0]) +
                         (q_true[1] + q_est[1]) * (q_true[1] + q_est[1]) +
                         (q_true[2] + q_est[2]) * (q_true[2] + q_est[2]) +
                         (q_true[3] + q_est[3]) * (q_true[3] + q_est[3])) /
                    sqrt(q_true[0] * q_true[0] + q_true[1] * q_true[1] +
                         q_true[2] * q_true[2] + q_true[3] * q_true[3]);

  rot_err = std::min(rot_err1, rot_err2);

  transl_err = sqrt((T_true[0] - T_est[0]) * (T_true[0] - T_est[0]) +
                    (T_true[1] - T_est[1]) * (T_true[1] - T_est[1]) +
                    (T_true[2] - T_est[2]) * (T_true[2] - T_est[2])) /
               sqrt(T_true[0] * T_true[0] + T_true[1] * T_true[1] +
                    T_true[2] * T_true[2]);
}

//------------------------------------------------------------------------------
void CoralPNPModel::MatToQuat(const Rot R, Eigen::Vector4d &q) {
  double tr = R(0, 0) + R(1, 1) + R(2, 2);
  double n4;

  if (tr > 0.0f) {
    q(0) = R(1, 2) - R(2, 1);
    q(1) = R(2, 0) - R(0, 2);
    q(2) = R(0, 1) - R(1, 0);
    q(3) = tr + 1.0f;
    n4 = q[3];
  } else if ((R(0, 0) > R(1, 1)) && (R(0, 0) > R(2, 2))) {
    q(0) = 1.0f + R(0, 0) - R(1, 1) - R(2, 2);
    q(1) = R(1, 0) + R(0, 1);
    q(2) = R(2, 0) + R(0, 2);
    q(3) = R(1, 2) - R(2, 1);
    n4 = q(0);
  } else if (R(1, 1) > R(2, 2)) {
    q(0) = R(1, 0) + R(0, 1);
    q(1) = 1.0f + R(1, 1) - R(0, 0) - R(2, 2);
    q(2) = R(2, 1) + R(1, 2);
    q(3) = R(2, 0) - R(0, 2);
    n4 = q(1);
  } else {
    q(0) = R(2, 0) + R(0, 2);
    q(1) = R(2, 1) + R(1, 2);
    q(2) = 1.0f + R(2, 2) - R(0, 0) - R(1, 1);
    q(3) = R(0, 1) - R(1, 0);
    n4 = q(2);
  }
  double scale = 0.5f / double(sqrt(n4));

  q(0) *= scale;
  q(1) *= scale;
  q(2) *= scale;
  q(3) *= scale;
}

//------------------------------------------------------------------------------
int CoralPNPModel::ModelDegreesOfFreedom() { return 4; }
//------------------------------------------------------------------------------
Eigen::MatrixXd CoralPNPModel::ModelEquation() {

  LOG(INFO) << " R is \n" << R_curr_;
  LOG(INFO) << " T is \n" << t_curr_;

  return Eigen::MatrixXd::Zero(4, 4);
}

} // namespace models
} // namespace coral
