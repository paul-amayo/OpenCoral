#ifndef CORAL_PNP_MODEL_H_
#define CORAL_PNP_MODEL_H_

#include "../features/coral_feature.h"
#include "../features/coral_feature_stereo_correspondence.h"
#include "coral_model.h"

#include <Eigen/Core>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace coral {

namespace models {

typedef Eigen::Matrix3d Rot;
typedef Eigen::Vector3d Trans;

class CoralPNPModel : public CoralModelBase {
public:
  CoralPNPModel();

  virtual ~CoralPNPModel() = default;

public:
  Eigen::MatrixXd EvaluateCost(const features::FeatureVectorSPtr &features);

  void UpdateModel(const features::FeatureVectorSPtr &features);

  int ModelDegreesOfFreedom();

  Eigen::MatrixXd ModelEquation();

  void SetCameraParams(Eigen::MatrixXd K);

  Eigen::MatrixXd GetCameraParams() const;

  Eigen::Matrix3d GetRotation() { return R_curr_; }

  Eigen::Vector3d GetTranslation() { return t_curr_; }

  void RelativeError(double &rot_err, double &transl_err, const Rot R_true,
                     Trans T_true, const Rot R_est, Trans T_est);

private:

  // Functions from the original EPnP code
  void
  AddCorrespondences(const features::CoralFeatureStereoCorrespondenceVectorSPtr
                         &stereo_features);

  double ComputePose(const Rot &R, Trans T);

  double ReprojError(const Rot R, const Trans t);

  void ChooseCtrlPoints(void);

  void ComputeBarycentricCoordinates(void);

  void FillM(cv::Mat &M, const int row, const Eigen::Vector4d &alphas,
             const double u, const double v) const;
  void ComputeCcs(const Eigen::MatrixXd &betas, const cv::Mat ut);

  void ComputePcs(void);

  void SolveSign(void);

  static void FindBetasApprox1(const cv::Mat &L_6x10, const cv::Mat &Rho,
                               Eigen::Vector4d &betas);

  static void FindBetasApprox2(const cv::Mat &L_6x10, const cv::Mat &Rho,
                               Eigen::Vector4d &betas);

  static void FindBetasApprox3(const cv::Mat &L_6x10, const cv::Mat &Rho,
                        Eigen::Vector4d &betas);

  static void QrSolve(cv::Mat &A, cv::Mat &b, cv::Mat &X);

  double Dot(const Eigen::MatrixXd v1, const Eigen::MatrixXd v2);

  static double DistSquared(const Eigen::MatrixXd p1, const Eigen::MatrixXd p2);

  void ComputeRho(cv::Mat &rho);

  void ComputeL6x10(const cv::Mat &ut, cv::Mat &L_6x10);

  static void GaussNewton(const cv::Mat& L_6x10, const cv::Mat& Rho,
                   Eigen::Vector4d &current_betas);

  static void ComputeAandbGN(const cv::Mat &l_6x10, const cv::Mat rho,
                             const Eigen::MatrixXd cb, cv::Mat& A, cv::Mat& b);

  double ComputeRotTrans(const cv::Mat ut, const Eigen::MatrixXd& betas, Rot &R,
                         Trans &t);

  void EstimateRotTrans(Rot &R, Trans &t);

  void MatToQuat(const Rot R, Eigen::Vector4d &q);

private:
  double u_c_, v_c_, f_u_, f_v_;

  std::vector<Eigen::Vector4d> alphas_;
  std::vector<Eigen::Vector3d> pws_, pcs_;
  std::vector<Eigen::Vector2d> us_;

  int num_correspondences_;

  Eigen::MatrixXd cws_, ccs_;

  // Current Estimation
  Rot R_curr_;
  Trans t_curr_;
};

} // namespace models
} // namespace coral
#endif // CORAL_PNP_MODEL_H_