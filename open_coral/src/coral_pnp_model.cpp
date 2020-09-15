#include "../include/models/coral_pnp_model.h"
#include "glog/logging.h"
#include "opencv2/core/core.hpp"
#include <cxeigen.hpp>
#include <opencv2/core/core_c.h>

namespace coral {
namespace models {
//------------------------------------------------------------------------------
CoralPNPModel::CoralPNPModel() {}
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
  return Eigen::MatrixXd::Zero(3, 3);
}
//------------------------------------------------------------------------------
void CoralPNPModel::UpdateModel(const features::FeatureVectorSPtr &features) {

  num_correspondences_ = features->size();

  // Cast features to derived class
  LOG(INFO) << "Updating the model" << std::endl;
  features::CoralFeatureStereoCorrespondanceVectorSPtr stereo_features(
      new features::CoralFeatureStereoCorrespondanceVector);
  for (const auto &feature : *features) {
    features::CoralFeatureStereoCorrespondanceSPtr stereo_feature =
        boost::dynamic_pointer_cast<features::CoralFeatureStereoCorrespondance>(
            feature);
    stereo_features->push_back(stereo_feature);
  }

  // Add the correspondences
  AddCorrespondences(stereo_features);
  LOG(INFO) << "Added correspondances";

  // compute the pose
  Rot R;
  Trans t;
  double err = ComputePose(R, t);
  LOG(INFO) << "Err is " << err;
}
//------------------------------------------------------------------------------
int CoralPNPModel::ModelDegreesOfFreedom() { return 4; }
//------------------------------------------------------------------------------
Eigen::MatrixXd CoralPNPModel::ModelEquation() {
  return Eigen::MatrixXd::Zero(4, 4);
}
//------------------------------------------------------------------------------
void CoralPNPModel::AddCorrespondences(
    const features::CoralFeatureStereoCorrespondanceVectorSPtr
        &stereo_features) {
  // Clear the previous correspondences

  if (pws != 0)
    delete[] pws;
  if (us != 0)
    delete[] us;
  if (alphas != 0)
    delete[] alphas;
  if (pcs != 0)
    delete[] pcs;

  num_correspondences_ = stereo_features->size();

  pws = new double[3 * num_correspondences_];
  us = new double[2 * num_correspondences_];
  alphas = new double[4 * num_correspondences_];
  pcs = new double[3 * num_correspondences_];

  int correspondance_no = 0;
  for (const auto &feature : *stereo_features) {
    pws_.push_back(feature->GetPoint3d());
    us_.push_back(feature->GetPointUV());

    alphas_.emplace_back(Eigen::MatrixXd::Zero(4, 1));
    pcs_.emplace_back(Eigen::MatrixXd::Zero(3, 1));

    pws[3 * correspondance_no] = feature->GetPoint3d()(0);
    pws[3 * correspondance_no + 1] = feature->GetPoint3d()(1);
    pws[3 * correspondance_no + 2] = feature->GetPoint3d()(2);

    us[2 * correspondance_no] = feature->GetPointUV()(0);
    us[2 * correspondance_no + 1] = feature->GetPointUV()(1);
    correspondance_no++;
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::CheckInliers() {
  mnInliersi = 0;
  for (int i = 0; i < num_correspondences_; i++) {
    cv::Point3f P3Dw = mvP3Dw[i];
    cv::Point2f P2D = mvP2D[i];

    double Xc =
        mRi(0, 0) * P3Dw.x + mRi(0, 1) * P3Dw.y + mRi(0, 2) * P3Dw.z + mti(0);
    double Yc =
        mRi(1, 0) * P3Dw.x + mRi(1, 1) * P3Dw.y + mRi(1, 2) * P3Dw.z + mti(1);
    double invZc = 1 / (mRi(2, 0) * P3Dw.x + mRi(2, 1) * P3Dw.y +
                       mRi(2, 2) * P3Dw.z + mti(2));

    double ue = u_c_ + f_u_ * Xc * invZc;
    double ve = v_c_ + f_v_ * Yc * invZc;

    double distX = P2D.x - ue;
    double distY = P2D.y - ve;

    double error2 = distX * distX + distY * distY;
    mnInliersi++;
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::ChooseCtrlPoints() {
  cws[0][0] = cws[0][1] = cws[0][2] = 0;
  for (int i = 0; i < num_correspondences_; i++)
    for (int j = 0; j < 3; j++)
      cws[0][j] += pws[3 * i + j];

  for (int j = 0; j < 3; j++)
    cws[0][j] /= num_correspondences_;

  cws_ = Eigen::MatrixXd::Zero(4, 3);

  for (int i = 0; i < num_correspondences_; i++)
    for (int j = 0; j < 3; j++)
      cws_(0, j) += pws_[i](j);

  for (int j = 0; j < 3; j++)
    cws_(0, j) /= num_correspondences_;

  // Take C1, C2, and C3 from PCA on the reference points:
  cv::Mat PW0 = cv::Mat(num_correspondences_, 3, CV_64F);

  cv::Mat PW0tPW0; //= cvMat(3, 3, CV_64F, pw0tpw0);
  cv::Mat DC;      // = cvMat(3, 1, CV_64F, dc);
  cv::Mat U, Ut;   //= cvMat(3, 3, CV_64F, uct);
  cv::Mat v;

  for (int i = 0; i < num_correspondences_; i++)
    for (int j = 0; j < 3; j++)
      PW0.at<double>(i, j) = pws_[i](j) - cws_(0, j);

  cv::mulTransposed(PW0, PW0tPW0, 1);

  cv::SVD::compute(PW0tPW0, DC, U, v, cv::SVD::MODIFY_A);

  Ut = U.t();
  for (int i = 1; i < 4; i++) {
    double k = sqrt(DC.at<double>(i - 1) / num_correspondences_);
    for (int j = 0; j < 3; j++)
      cws_(i, j) = cws_(0, j) + k * Ut.at<double>(i - 1, j);
  }

  CvMat *PW0_old = cvCreateMat(num_correspondences_, 3, CV_64F);

  double pw0tpw0_old[3 * 3], dc_old[3], uct_old[3 * 3];
  CvMat PW0tPW0_old = cvMat(3, 3, CV_64F, pw0tpw0_old);
  CvMat DC_old = cvMat(3, 1, CV_64F, dc_old);
  CvMat UCt_old = cvMat(3, 3, CV_64F, uct_old);

  for (int i = 0; i < num_correspondences_; i++)
    for (int j = 0; j < 3; j++)
      PW0_old->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

  cvMulTransposed(PW0_old, &PW0tPW0_old, 1);
  cvSVD(&PW0tPW0_old, &DC_old, &UCt_old, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
  cvReleaseMat(&PW0_old);

  for (int i = 1; i < 4; i++) {
    double k = sqrt(dc_old[i - 1] / num_correspondences_);
    for (int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct_old[3 * (i - 1) + j];
  }

  // COmpare the values of cws
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      LOG(INFO) << "Cws old is " << cws[i][j] << " while new is " << cws_(i, j);
    }
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::SetMaxCorrespondences(const int n) {
  MaxCorrespondences_ = n;
}
//------------------------------------------------------------------------------
void CoralPNPModel::ResetCorrespondences() { num_correspondences_ = 0; }
//------------------------------------------------------------------------------
double CoralPNPModel::ComputePose(const Rot &R, Trans T) {

  ChooseCtrlPoints();
  ComputeBarycentricCoordinates();

  cv::Mat M = cv::Mat(2 * num_correspondences_, 12, CV_64F);
  CvMat *M_old = cvCreateMat(2 * num_correspondences_, 12, CV_64F);

  for (int i = 0; i < num_correspondences_; i++) {
    FillM(M, 2 * i, alphas_[i], us_[i](0), us_[i](1));
    fill_M(M_old, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);
  }

  double mtm_old[12 * 12], d_old[12], ut_old[12 * 12], vt_old[12*12];
  CvMat MtM_old = cvMat(12, 12, CV_64F, mtm_old);
  CvMat D_old = cvMat(12, 1, CV_64F, d_old);
  CvMat Ut_old = cvMat(12, 12, CV_64F, ut_old);
  CvMat Vt_old = cvMat(12, 12, CV_64F, vt_old);

  cvMulTransposed(M_old, &MtM_old, 1);
  cvSVD(&MtM_old, &D_old, &Ut_old, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);
  cvReleaseMat(&M_old);

  cv::Mat MtM = cv::Mat(12, 12, CV_64F);
  cv::Mat D = cv::Mat(12, 1, CV_64F);
  cv::Mat U = cv::Mat(12, 12, CV_64F);
  cv::Mat V;

  cv::mulTransposed(M, MtM, 1);
  cv::SVD::compute(MtM, D, U, V, cv::SVD::MODIFY_A);

  int count = 0;

  cv::Mat L_6x10 = cv::Mat(6, 10, CV_64F);
  cv::Mat Rho = cv::Mat(6, 1, CV_64F);

  cv::Mat Ut = U.t();
  ComputeL6x10(Ut, L_6x10);
  ComputeRho(Rho);

  double l_6x10_old[6 * 10], rho_old[6];
  CvMat L_6x10_old = cvMat(6, 10, CV_64F, l_6x10_old);
  CvMat Rho_old = cvMat(6, 1, CV_64F, rho_old);

  compute_L_6x10(ut_old, l_6x10_old);
  count = 0;

  compute_rho(rho_old);

  cv::Mat l_32;
  cv::Mat l_64 = cv::cvarrToMat(&L_6x10_old);

  l_64.convertTo(l_32, CV_64F);

  cv::Mat rho_32;
  cv::Mat rho_64 = cv::cvarrToMat(&Rho_old);
  rho_64.convertTo(rho_32, CV_64F);
//
//  LOG(INFO) << "L diff is \n" << l_32 - L_6x10;
//
//  LOG(INFO) << "Rho diff is \n" << rho_32 - Rho;

  // Set up vectors for
  std::vector<Eigen::Vector4d> betas_vector(3);
  std::vector<Rot> Rot_vector(3);
  std::vector<Trans> trans_vector(3);
  std::vector<double> error_vector(3);

  double Betas[4][4], rep_errors[4];
  double Rs[4][3][3], ts[4][3];

  FindBetasApprox1(L_6x10, Rho, betas_vector[0]);
  find_betas_approx_1(&L_6x10_old, &Rho_old, Betas[1]);

  GaussNewton(L_6x10, Rho, betas_vector[0]);
  gauss_newton(&L_6x10_old, &Rho_old, Betas[1]);


  ComputeRotTrans(Ut, betas_vector[0], Rot_vector[0], trans_vector[0]);
  compute_R_and_t(ut_old, Betas[1], Rs[1], ts[1]);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      LOG(INFO) << "Rot " << i << "," << j << " init is " << Rs[1][i][j]
                << " and new is " << Rot_vector[0](i, j);
    }
  }

  //    GaussNewton(L_6x10, Rho, betas_vector[0]);
  //    error_vector[0] =
  //        ComputeRotTrans(Ut, betas_vector[0], Rot_vector[0],
  //        trans_vector[0]);

  //  FindBetasApprox2(L_6x10, Rho, betas_vector[1]);
  //  GaussNewton(L_6x10, Rho, betas_vector[1]);
  //  error_vector[1] =
  //      ComputeRotTrans(Ut, betas_vector[1], Rot_vector[1], trans_vector[1]);
  //
  //  FindBetasApprox3(L_6x10, Rho, betas_vector[2]);
  //  GaussNewton(L_6x10, Rho, betas_vector[2]);
  //  error_vector[2] =
  //      ComputeRotTrans(Ut, betas_vector[2], Rot_vector[2], trans_vector[2]);
  //
  //  int N = 0;
  //  if (error_vector[1] < error_vector[0])
  //    N = 1;
  //  if (error_vector[2] < error_vector[N])
  //    N = 2;
  //  LOG(INFO) << " R is " << Rot_vector[N];
  //
  //  // copy_R_and_t(Rs[N], ts[N], R, t);

  return 0;
}
//------------------------------------------------------------------------------
void CoralPNPModel::RelativeError(double &rot_err, double &transl_err,
                                  const Rot R_true, Trans T_true,
                                  const Rot R_est, Trans T_est) {
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
double CoralPNPModel::ReprojError(const Rot R, const Trans t) {
  double sum2 = 0.0;

  for (int i = 0; i < num_correspondences_; i++) {

    double Xc = Dot(R.row(0), pws_[i]) + t[0];
    double Yc = Dot(R.row(1), pws_[i]) + t[1];
    double inv_Zc = 1.0 / (Dot(R.row(2), pws_[i]) + t[2]);

    double ue = u_c_ + f_u_ * Xc * inv_Zc;
    double ve = v_c_ + f_v_ * Yc * inv_Zc;
    double u = us_[i](0);
    double v = us_[i](1);

    sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
  }

  return sum2 / num_correspondences_;
}
//------------------------------------------------------------------------------
void CoralPNPModel::ComputeBarycentricCoordinates() {

  double cc[3 * 3], cc_inv[3 * 3];
  cv::Mat CC = cv::Mat(3, 3, CV_64F);
  cv::Mat CC_inv = cv::Mat(3, 3, CV_64F);

  for (int i = 0; i < 3; i++) {
    for (int j = 1; j < 4; j++) {
      double cws_val = cws_(j, i) - cws_(0, i);
      CC.at<double>(i, j - 1) = cws_val;
    }
  }
  cv::invert(CC, CC_inv, cv::DECOMP_SVD);

  for (int i = 0; i < num_correspondences_; i++) {
    Eigen::Vector3d pi = pws_[i];
    Eigen::Vector4d alpha = alphas_[i];

    for (int j = 0; j < 3; j++)
      alphas_[i](1 + j) = CC_inv.at<double>(j, 0) * (pi(0) - cws_(0, 0)) +
                          CC_inv.at<double>(j, 1) * (pi[1] - cws_(0, 1)) +
                          CC_inv.at<double>(j, 2) * (pi[2] - cws_(0, 2));
    alphas_[i](0) = 1.0f - alphas_[i](1) - alphas_[i](2) - alphas_[i](3);
  }

  CvMat CC_old = cvMat(3, 3, CV_64F, cc);
  CvMat CC_inv_old = cvMat(3, 3, CV_64F, cc_inv);

  for (int i = 0; i < 3; i++)
    for (int j = 1; j < 4; j++)
      cc[3 * i + j - 1] = cws[j][i] - cws[0][i];

  cvInvert(&CC_old, &CC_inv_old, CV_SVD);
  double *ci = cc_inv;
  for (int i = 0; i < num_correspondences_; i++) {
    double *pi = pws + 3 * i;
    double *a = alphas + 4 * i;

    for (int j = 0; j < 3; j++)
      a[1 + j] = ci[3 * j] * (pi[0] - cws[0][0]) +
                 ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                 ci[3 * j + 2] * (pi[2] - cws[0][2]);
    a[0] = 1.0f - a[1] - a[2] - a[3];

    for (int j = 0; j < 4; ++j) {
      LOG(INFO) << "Alpha init is " << a[j] << " while new is "
                << alphas_[i](j);
    }
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::FillM(cv::Mat &M, const int row,
                          const Eigen::Vector4d &alphas, const double u,
                          const double v) const {
  for (int i = 0; i < 4; i++) {
    M.at<double>(row, 3 * i) = alphas(i) * f_u_;
    M.at<double>(row, 3 * i + 1) = 0.0;
    M.at<double>(row, 3 * i + 2) = alphas(i) * (u_c_ - u);

    M.at<double>(row + 1, 3 * i) = 0.0;
    M.at<double>(row + 1, 3 * i + 1) = alphas(i) * f_v_;
    M.at<double>(row + 1, 3 * i + 2) = alphas(i) * (v_c_ - v);
  }
}

void CoralPNPModel::fill_M(CvMat *M, const int row, const double *as,
                           const double u, const double v) {
  double *M1 = M->data.db + row * 12;
  double *M2 = M1 + 12;

  for (int i = 0; i < 4; i++) {
    M1[3 * i] = as[i] * f_u_;
    M1[3 * i + 1] = 0.0;
    M1[3 * i + 2] = as[i] * (u_c_ - u);

    M2[3 * i] = 0.0;
    M2[3 * i + 1] = as[i] * f_v_;
    M2[3 * i + 2] = as[i] * (v_c_ - v);
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::ComputeCcs(const Eigen::MatrixXd &betas, const cv::Mat ut) {

  ccs_ = Eigen::MatrixXd::Zero(4, 3);
  for (int i = 0; i < 4; i++)
    ccs_(i, 0) = ccs_(i, 1) = ccs_(i, 2) = 0.0f;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        ccs_(j, k) += betas(i) * ut.at<double>((11 - i), 3 * j + k);
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::ComputePcs() {
  for (int i = 0; i < num_correspondences_; i++) {
    for (int j = 0; j < 3; j++) {
      pcs_[i](j) = alphas_[i](0) * ccs_(0, j) + alphas_[i](1) * ccs_(1, j) +
                   alphas_[i](2) * ccs_(2, j) + alphas_[i](3) * ccs_(3, j);
    }
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::SolveSign() {
  if (pcs_[0](2) < 0.0) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 3; j++)
        ccs_(i, j) = -ccs_(i, j);

    for (int i = 0; i < num_correspondences_; i++) {
      pcs_[i](0) = -pcs_[i](0);
      pcs_[i](1) = -pcs_[i](1);
      pcs_[i](2) = -pcs_[i](2);
    }
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::FindBetasApprox1(const cv::Mat &L_6x10, const cv::Mat &Rho,
                                     Eigen::Vector4d &betas) {
  cv::Mat L_6x4 = cv::Mat(6, 4, CV_64F);
  cv::Mat B4 = cv::Mat(4, 1, CV_64F);

  for (int i = 0; i < 6; i++) {
    L_6x4.at<double>(i, 0) = L_6x10.at<double>(i, 0);
    L_6x4.at<double>(i, 1) = L_6x10.at<double>(i, 1);
    L_6x4.at<double>(i, 2) = L_6x10.at<double>(i, 3);
    L_6x4.at<double>(i, 3) = L_6x10.at<double>(i, 6);
  }
  cv::solve(L_6x4, Rho, B4, cv::DECOMP_SVD);


  if (B4.at<double>(0) < 0) {
    betas(0) = sqrt(-B4.at<double>(0));
    betas(1) = -B4.at<double>(1) / betas(0);
    betas(2) = -B4.at<double>(2) / betas(0);
    betas(3) = -B4.at<double>(3) / betas(0);
  } else {
    betas(0) = sqrt(B4.at<double>(0));
    betas(1) = B4.at<double>(1) / betas(0);
    betas(2) = B4.at<double>(2) / betas(0);
    betas(3) = B4.at<double>(3) / betas(0);
  }
}

void CoralPNPModel::find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho,
                                        double *betas) {
  double l_6x4[6 * 4], b4[4];
  CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
  CvMat B4 = cvMat(4, 1, CV_64F, b4);

  for (int i = 0; i < 6; i++) {
    cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
  }

  cvSolve(&L_6x4, Rho, &B4, CV_SVD);

  if (b4[0] < 0) {
    betas[0] = sqrt(-b4[0]);
    betas[1] = -b4[1] / betas[0];
    betas[2] = -b4[2] / betas[0];
    betas[3] = -b4[3] / betas[0];
  } else {
    betas[0] = sqrt(b4[0]);
    betas[1] = b4[1] / betas[0];
    betas[2] = b4[2] / betas[0];
    betas[3] = b4[3] / betas[0];
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::FindBetasApprox2(const cv::Mat &L_6x10, const cv::Mat &Rho,
                                     Eigen::Vector4d &betas) {

  cv::Mat L_6x3 = cv::Mat(6, 3, CV_64F);
  cv::Mat B3 = cv::Mat(3, 1, CV_64F);

  for (int i = 0; i < 6; i++) {
    L_6x3.at<double>(i, 0) = L_6x10.at<double>(i, 0);
    L_6x3.at<double>(i, 1) = L_6x10.at<double>(i, 1);
    L_6x3.at<double>(i, 2) = L_6x10.at<double>(i, 2);
  }

  cv::solve(L_6x3, Rho, B3, cv::DECOMP_SVD);

  if (B3.at<double>(0, 0) < 0) {
    betas(0) = sqrt(-B3.at<double>(0, 0));
    betas(1) = (B3.at<double>(2, 0) < 0) ? sqrt(-B3.at<double>(2, 0)) : 0.0;
  } else {
    betas(0) = sqrt(B3.at<double>(0, 0));
    betas(1) = (B3.at<double>(2, 0) > 0) ? sqrt(B3.at<double>(2, 0)) : 0.0;
  }

  if (B3.at<double>(1, 0) < 0)
    betas(0) = -betas(0);

  betas(2) = 0.0;
  betas(3) = 0.0;
}
//------------------------------------------------------------------------------
void CoralPNPModel::FindBetasApprox3(const cv::Mat &L_6x10, const cv::Mat &Rho,
                                     Eigen::Vector4d &betas) {
  cv::Mat L_6x5 = cv::Mat(6, 5, CV_64F);
  cv::Mat B5 = cv::Mat(5, 1, CV_64F);

  for (int i = 0; i < 6; i++) {
    L_6x5.at<double>(i, 0) = L_6x10.at<double>(i, 0);
    L_6x5.at<double>(i, 1) = L_6x10.at<double>(i, 1);
    L_6x5.at<double>(i, 2) = L_6x10.at<double>(i, 2);
    L_6x5.at<double>(i, 0) = L_6x10.at<double>(i, 3);
    L_6x5.at<double>(i, 1) = L_6x10.at<double>(i, 4);
  }

  cv::solve(L_6x5, Rho, B5, cv::DECOMP_SVD);

  if (B5.at<double>(0, 0) < 0) {
    betas(0) = sqrt(-B5.at<double>(0, 0));
    betas(1) = (B5.at<double>(2, 0) < 0) ? sqrt(-B5.at<double>(2, 0)) : 0.0;
  } else {
    betas(0) = sqrt(B5.at<double>(0, 0));
    betas(1) = (B5.at<double>(2, 0) > 0) ? sqrt(B5.at<double>(2, 0)) : 0.0;
  }
  if (B5.at<double>(1, 0) < 0)
    betas(0) = -betas(0);
  betas(2) = B5.at<double>(3, 0) / betas(0);
  betas(3) = 0.0;
}
//------------------------------------------------------------------------------
void CoralPNPModel::QrSolve(cv::Mat &A, cv::Mat &b, cv::Mat &X) {
  static int max_nr = 0;

  const int num_rows = A.rows;
  const int num_cols = A.cols;

  Eigen::MatrixXd A1 = Eigen::VectorXd(num_rows);
  Eigen::MatrixXd A2 = Eigen::VectorXd(num_rows);

  for (int k = 0; k < num_cols; k++) {

    double eta = fabs(A.at<double>(k, k));

    for (int i = k; i < num_rows - 1; i++) {
      double elt = fabs(A.at<double>(i, k));

      if (eta < elt)
        eta = elt;
    }

    if (eta == 0) {
      A1(k) = A2(k) = 0.0;
      std::cerr << "God damnit, A is singular, this shouldn't happen."
                << std::endl;
      return;
    } else {
      double sum = 0.0, inv_eta = 1. / eta;
      for (int i = k; i < num_rows; i++) {
        A.at<double>(i, k) *= inv_eta;

        sum += A.at<double>(i, k) * A.at<double>(i, k);
      }
      double sigma = sqrt(sum);
      if (A.at<double>(k, k) < 0) {
        sigma = -sigma;
      }
      A.at<double>(k, k) += sigma;
      A1(k) = sigma * A.at<double>(k, k);
      A2(k) = -eta * sigma;
      for (int j = k + 1; j < num_cols; j++) {
        sum = 0;
        for (int i = k; i < num_rows; i++) {
          sum += A.at<double>(i, k) * A.at<double>(i, j);
        }
        double tau = sum / A1(k);
        for (int i = k; i < num_rows; i++) {
          A.at<double>(i, j) -= tau * A.at<double>(i, k);
        }
      }
    }
    // LOG(INFO) << "A final is \n" << A;
  }
  //
  //  LOG(INFO) << "A new is \n" << A;

  // b <- Qt b
  for (int j = 0; j < num_cols; j++) {
    double tau = 0;
    for (int i = j; i < num_rows; i++) {
      tau += A.at<double>(i, j) * b.at<double>(i);
    }
    tau /= A1(j);

    for (int i = j; i < num_rows; i++) {
      b.at<double>(i) -= tau * A.at<double>(i, j);
    }
  }

  // X = R-1 b
  X.at<double>(num_cols - 1) = b.at<double>(num_cols - 1) / A2(num_cols - 1);
  for (int i = num_cols - 2; i >= 0; i--) {
    double sum = 0;
    for (int j = i + 1; j < num_cols; j++) {
      sum += A.at<double>(i, j) * X.at<double>(j);
    }
    X.at<double>(i) = (b.at<double>(i) - sum) / A2(i);
  }
}

void CoralPNPModel::qr_solve(CvMat *A, CvMat *b, CvMat *X) {
  static int max_nr = 0;
  static double *A1, *A2;

  const int nr = A->rows;
  const int nc = A->cols;

  if (max_nr != 0 && max_nr < nr) {
    delete[] A1;
    delete[] A2;
  }
  if (max_nr < nr) {
    max_nr = nr;
    A1 = new double[nr];
    A2 = new double[nr];
  }

  double *pA = A->data.db, *ppAkk = pA;
  for (int k = 0; k < nc; k++) {
    double *ppAik = ppAkk, eta = fabs(*ppAik);
    for (int i = k + 1; i < nr; i++) {
      double elt = fabs(*ppAik);
      if (eta < elt)
        eta = elt;
      ppAik += nc;
    }
    if (eta == 0) {
      A1[k] = A2[k] = 0.0;
      std::cerr << "God damnit, A is singular, this shouldn't happen."
                << std::endl;
      return;
    } else {
      double *ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
      for (int i = k; i < nr; i++) {
        *ppAik *= inv_eta;
        sum += *ppAik * *ppAik;
        ppAik += nc;
      }
      double sigma = sqrt(sum);
      if (*ppAkk < 0)
        sigma = -sigma;
      *ppAkk += sigma;
      A1[k] = sigma * *ppAkk;
      A2[k] = -eta * sigma;
      for (int j = k + 1; j < nc; j++) {
        double *ppAik = ppAkk, sum = 0;
        for (int i = k; i < nr; i++) {
          sum += *ppAik * ppAik[j - k];
          ppAik += nc;
        }
        double tau = sum / A1[k];
        ppAik = ppAkk;
        for (int i = k; i < nr; i++) {
          ppAik[j - k] -= tau * *ppAik;
          ppAik += nc;
        }
      }
    }
    ppAkk += nc + 1;
  }
  // LOG(INFO) << "A init is \n" << cv::cvarrToMat(A);

  // b <- Qt b
  double *ppAjj = pA, *pb = b->data.db;
  for (int j = 0; j < nc; j++) {
    double *ppAij = ppAjj, tau = 0;
    for (int i = j; i < nr; i++) {
      tau += *ppAij * pb[i];
      ppAij += nc;
    }
    tau /= A1[j];
    ppAij = ppAjj;
    for (int i = j; i < nr; i++) {
      pb[i] -= tau * *ppAij;
      ppAij += nc;
    }
    ppAjj += nc + 1;
  }

  // X = R-1 b
  double *pX = X->data.db;
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
  for (int i = nc - 2; i >= 0; i--) {
    double *ppAij = pA + i * nc + (i + 1), sum = 0;

    for (int j = i + 1; j < nc; j++) {
      sum += *ppAij * pX[j];
      ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];
  }

}
//------------------------------------------------------------------------------
double CoralPNPModel::Dot(const Eigen::MatrixXd v1, const Eigen::MatrixXd v2) {
  return v1(0) * v2(0) + v1(1) * v2(1) + v1(2) * v2(2);
}
//------------------------------------------------------------------------------
double CoralPNPModel::DistSquared(const Eigen::MatrixXd p1,
                                  const Eigen::MatrixXd p2) {
  return (p1(0) - p2(0)) * (p1(0) - p2(0)) + (p1(1) - p2(1)) * (p1(1) - p2(1)) +
         (p1(2) - p2(2)) * (p1(2) - p2(2));
}
//------------------------------------------------------------------------------
void CoralPNPModel::ComputeRho(cv::Mat &rho) {
  rho.at<double>(0) = DistSquared(cws_.row(0), cws_.row(1));
  rho.at<double>(1) = DistSquared(cws_.row(0), cws_.row(2));
  rho.at<double>(2) = DistSquared(cws_.row(0), cws_.row(3));
  rho.at<double>(3) = DistSquared(cws_.row(1), cws_.row(2));
  rho.at<double>(4) = DistSquared(cws_.row(1), cws_.row(3));
  rho.at<double>(5) = DistSquared(cws_.row(2), cws_.row(3));
}

void CoralPNPModel::compute_rho(double *rho) {
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}
//------------------------------------------------------------------------------
void CoralPNPModel::ComputeL6x10(const cv::Mat &ut, cv::Mat &L_6x10) {

  // Ut is a 12 by 12 matrix

  std::vector<cv::Mat> v;
  v.push_back(ut.row(11));
  v.push_back(ut.row(10));
  v.push_back(ut.row(9));
  v.push_back(ut.row(8));

  std::vector<Eigen::MatrixXd> dv;

  for (int i = 0; i < 4; i++) {
    int a = 0, b = 1;
    Eigen::MatrixXd dv_curr = Eigen::MatrixXd(6, 3);
    for (int j = 0; j < 6; j++) {

      dv_curr(j, 0) = v[i].at<double>(3 * a) - v[i].at<double>(3 * b);
      dv_curr(j, 1) = v[i].at<double>(3 * a + 1) - v[i].at<double>(3 * b + 1);
      dv_curr(j, 2) = v[i].at<double>(3 * a + 2) - v[i].at<double>(3 * b + 2);

      b++;
      if (b > 3) {
        a++;
        b = a + 1;
      }
    }
    dv.push_back(dv_curr);
  }

  for (int i = 0; i < 6; i++) {
    L_6x10.at<double>(i, 0) = Dot(dv[0].row(i), dv[0].row(i));
    L_6x10.at<double>(i, 1) = 2.0f * Dot(dv[0].row(i), dv[1].row(i));
    L_6x10.at<double>(i, 2) = Dot(dv[1].row(i), dv[1].row(i));
    L_6x10.at<double>(i, 3) = 2.0f * Dot(dv[0].row(i), dv[2].row(i));
    L_6x10.at<double>(i, 4) = 2.0f * Dot(dv[1].row(i), dv[2].row(i));
    L_6x10.at<double>(i, 5) = Dot(dv[2].row(i), dv[2].row(i));
    L_6x10.at<double>(i, 6) = 2.0f * Dot(dv[0].row(i), dv[3].row(i));
    L_6x10.at<double>(i, 7) = 2.0f * Dot(dv[1].row(i), dv[3].row(i));
    L_6x10.at<double>(i, 8) = 2.0f * Dot(dv[2].row(i), dv[3].row(i));
    L_6x10.at<double>(i, 9) = Dot(dv[3].row(i), dv[3].row(i));
  }
}

void CoralPNPModel::compute_L_6x10(const double *ut, double *l_6x10) {
  const double *v[4];

  v[0] = ut + 12 * 11;
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 * 9;
  v[3] = ut + 12 * 8;

  double dv[4][6][3];

  for (int i = 0; i < 4; i++) {
    int a = 0, b = 1;
    for (int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
        a++;
        b = a + 1;
      }
    }
  }

  for (int i = 0; i < 6; i++) {
    double *row = l_6x10 + 10 * i;

    row[0] = dot(dv[0][i], dv[0][i]);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
    row[2] = dot(dv[1][i], dv[1][i]);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
    row[5] = dot(dv[2][i], dv[2][i]);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
    row[9] = dot(dv[3][i], dv[3][i]);
  }
}
double CoralPNPModel::dist2(const double *p1, const double *p2) {
  return (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) +
         (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double CoralPNPModel::dot(const double *v1, const double *v2) {
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}
//------------------------------------------------------------------------------
void CoralPNPModel::GaussNewton(const cv::Mat L_6x10, const cv::Mat Rho,
                                Eigen::Vector4d &current_betas) {
  const int iterations_number = 1;

  cv::Mat A = cv::Mat(6, 4, CV_64F);
  cv::Mat B = cv::Mat(6, 1, CV_64F);
  cv::Mat X = cv::Mat(4, 1, CV_64F);

  for (int k = 0; k < iterations_number; k++) {
    ComputeAandbGN(L_6x10, Rho, current_betas, A, B);
    QrSolve(A, B, X);

    for (int i = 0; i < 4; i++)
      current_betas(i) += X.at<double>(i);
  }
}

void CoralPNPModel::gauss_newton(const CvMat *L_6x10, const CvMat *Rho,
                                 double current_betas[4]) {
  const int iterations_number = 1;

  double a[6 * 4], b[6], x[4];
  CvMat A = cvMat(6, 4, CV_64F, a);
  CvMat B = cvMat(6, 1, CV_64F, b);
  CvMat X = cvMat(4, 1, CV_64F, x);

  for (int k = 0; k < iterations_number; k++) {
    compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db, current_betas,
                                 &A, &B);
    qr_solve(&A, &B, &X);
    for (int i = 0; i < 4; i++) {
      current_betas[i] += x[i];
    }
  }
}
//------------------------------------------------------------------------------
void CoralPNPModel::ComputeAandbGN(const cv::Mat &l_6x10, const cv::Mat rho,
                                   Eigen::MatrixXd cb, cv::Mat A, cv::Mat b) {
  for (int i = 0; i < 6; i++) {
    A.at<double>(i, 0) =
        2 * l_6x10.at<double>(i, 0) * cb(0) + l_6x10.at<double>(i, 1) * cb(1) +
        l_6x10.at<double>(i, 3) * cb(2) + l_6x10.at<double>(i, 6) * cb(3);
    A.at<double>(i, 1) =
        l_6x10.at<double>(i, 1) * cb(0) + 2 * l_6x10.at<double>(i, 2) * cb(1) +
        l_6x10.at<double>(i, 4) * cb(2) + l_6x10.at<double>(i, 7) * cb(3);
    A.at<double>(i, 2) =
        l_6x10.at<double>(i, 3) * cb(0) + l_6x10.at<double>(i, 4) * cb(1) +
        2 * l_6x10.at<double>(i, 5) * cb(2) + l_6x10.at<double>(i, 8) * cb(3);
    A.at<double>(i, 3) =
        l_6x10.at<double>(i, 6) * cb(0) + l_6x10.at<double>(i, 7) * cb(1) +
        l_6x10.at<double>(i, 8) * cb(2) + 2 * l_6x10.at<double>(i, 9) * cb(3);

    b.at<double>(i, 0) =
        (rho.at<double>(i) - (l_6x10.at<double>(i, 0) * cb(0) * cb(0) +
                             l_6x10.at<double>(i, 1) * cb(0) * cb(1) +
                             l_6x10.at<double>(i, 2) * cb(1) * cb(1) +
                             l_6x10.at<double>(i, 3) * cb(0) * cb(2) +
                             l_6x10.at<double>(i, 4) * cb(1) * cb(2) +
                             l_6x10.at<double>(i, 5) * cb(2) * cb(2) +
                             l_6x10.at<double>(i, 6) * cb(0) * cb(3) +
                             l_6x10.at<double>(i, 7) * cb(1) * cb(3) +
                             l_6x10.at<double>(i, 8) * cb(2) * cb(3) +
                             l_6x10.at<double>(i, 9) * cb(3) * cb(3)));
  }
}

void CoralPNPModel::compute_A_and_b_gauss_newton(const double *l_6x10,
                                                 const double *rho,
                                                 double betas[4], CvMat *A,
                                                 CvMat *b) {
  for (int i = 0; i < 6; i++) {
    const double *rowL = l_6x10 + i * 10;
    double *rowA = A->data.db + i * 4;

    rowA[0] = 2 * rowL[0] * betas[0] + rowL[1] * betas[1] + rowL[3] * betas[2] +
              rowL[6] * betas[3];
    rowA[1] = rowL[1] * betas[0] + 2 * rowL[2] * betas[1] + rowL[4] * betas[2] +
              rowL[7] * betas[3];
    rowA[2] = rowL[3] * betas[0] + rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +
              rowL[8] * betas[3];
    rowA[3] = rowL[6] * betas[0] + rowL[7] * betas[1] + rowL[8] * betas[2] +
              2 * rowL[9] * betas[3];

    cvmSet(b, i, 0,
           rho[i] -
               (rowL[0] * betas[0] * betas[0] + rowL[1] * betas[0] * betas[1] +
                rowL[2] * betas[1] * betas[1] + rowL[3] * betas[0] * betas[2] +
                rowL[4] * betas[1] * betas[2] + rowL[5] * betas[2] * betas[2] +
                rowL[6] * betas[0] * betas[3] + rowL[7] * betas[1] * betas[3] +
                rowL[8] * betas[2] * betas[3] + rowL[9] * betas[3] * betas[3]));
  }
}
//------------------------------------------------------------------------------
double CoralPNPModel::ComputeRotTrans(const cv::Mat ut,
                                      const Eigen::MatrixXd betas, Rot &R,
                                      Trans &t) {
  ComputeCcs(betas, ut);
  LOG(INFO) << "Compute ccs";
  ComputePcs();
  LOG(INFO) << "Compute pcs";
  SolveSign();
  LOG(INFO) << "Compute sign";

  EstimateRotTrans(R, t);
  return ReprojError(R, t);
}

double CoralPNPModel::compute_R_and_t(const double *ut, const double *betas,
                                      double (*R)[3], double *t) {
  compute_ccs(betas, ut);
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
}

void CoralPNPModel::estimate_R_and_t(double (*R)[3], double *t) {
  double pc0[3], pw0[3];

  pc0[0] = pc0[1] = pc0[2] = 0.0;
  pw0[0] = pw0[1] = pw0[2] = 0.0;

  for (int i = 0; i < num_correspondences_; i++) {
    const double *pc = pcs + 3 * i;
    const double *pw = pws + 3 * i;

    for (int j = 0; j < 3; j++) {
      pc0[j] += pc[j];
      pw0[j] += pw[j];
    }
  }

  for (int j = 0; j < 3; j++) {
    pc0[j] /= num_correspondences_;
    pw0[j] /= num_correspondences_;
  }

  double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
  CvMat ABt = cvMat(3, 3, CV_64F, abt);
  CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
  CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
  CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

  cvSetZero(&ABt);
  for (int i = 0; i < num_correspondences_; i++) {
    double *pc = pcs + 3 * i;
    double *pw = pws + 3 * i;

    for (int j = 0; j < 3; j++) {
      abt[3 * j] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);
    }

  const double det = R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] +
                     R[0][2] * R[1][0] * R[2][1] - R[0][2] * R[1][1] * R[2][0] -
                     R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

  if (det < 0) {
    R[2][0] = -R[2][0];
    R[2][1] = -R[2][1];
    R[2][2] = -R[2][2];
  }

  t[0] = pc0[0] - dot(R[0], pw0);
  t[1] = pc0[1] - dot(R[1], pw0);
  t[2] = pc0[2] - dot(R[2], pw0);
}

void CoralPNPModel::compute_ccs(const double *betas, const double *ut) {
  for (int i = 0; i < 4; i++)
    ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

  for (int i = 0; i < 4; i++) {
    const double *v = ut + 12 * (11 - i);
    for (int j = 0; j < 4; j++)
      for (int k = 0; k < 3; k++)
        ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

void CoralPNPModel::compute_pcs(void) {
  for (int i = 0; i < num_correspondences_; i++) {
    double *a = alphas + 4 * i;
    double *pc = pcs + 3 * i;

    for (int j = 0; j < 3; j++) {
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] +
              a[3] * ccs[3][j];
    }
  }
}

double CoralPNPModel::reprojection_error(const double (*R)[3],
                                         const double *t) {
  double sum2 = 0.0;

  for (int i = 0; i < num_correspondences_; i++) {
    double *pw = pws + 3 * i;
    double Xc = dot(R[0], pw) + t[0];
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
    double ue = u_c_ + f_u_ * Xc * inv_Zc;
    double ve = v_c_ + f_v_ * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];

    sum2 += sqrt((u - ue) * (u - ue) + (v - ve) * (v - ve));
  }
  return sum2 / num_correspondences_;
}

void CoralPNPModel::solve_for_sign() {
  if (pcs[2] < 0.0) {
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 3; j++)
        ccs[i][j] = -ccs[i][j];

    for (int i = 0; i < num_correspondences_; i++) {
      pcs[3 * i] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
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
void CoralPNPModel::EstimateRotTrans(Rot &R, Trans &t) {

  Eigen::Vector3d pc0, pw0;
  pc0 = Eigen::MatrixXd::Zero(3, 1);
  pw0 = Eigen::MatrixXd::Zero(3, 1);

  for (int i = 0; i < num_correspondences_; i++) {

    for (int j = 0; j < 3; j++) {
      pc0(j) += pcs_[i](j);
      pw0(j) += pws_[i](j);
    }
  }

  for (int j = 0; j < 3; j++) {
    pc0(j) /= num_correspondences_;
    pw0(j) /= num_correspondences_;
  }

  cv::Mat ABt = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
  cv::Mat ABt_D = cv::Mat(3, 1, CV_64F, cv::Scalar(0));
  cv::Mat ABt_U = cv::Mat(3, 3, CV_64F, cv::Scalar(0));
  cv::Mat AB_V = cv::Mat(3, 3, CV_64F, cv::Scalar(0));

  for (int i = 0; i < num_correspondences_; i++) {
    for (int j = 0; j < 3; j++) {
      ABt.at<double>(j, 0) += (pcs_[i](j) - pc0(j)) * (pws_[i](0) - pw0(0));
      ABt.at<double>(j, 1) += (pcs_[i](j) - pc0(j)) * (pws_[i](1) - pw0(1));
      ABt.at<double>(j, 2) += (pcs_[i](j) - pc0(j)) * (pws_[i](2) - pw0(2));
    }
  }

  // LOG(INFO) << "Abt new is \n" << ABt;

  cv::SVD::compute(ABt, ABt_D, ABt_U, AB_V, cv::SVD::MODIFY_A);

  cv::Mat ABt_V = AB_V.t();
  // LOG(INFO)<<"Abt V is \n"<<ABt_V;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      Eigen::Vector3d abt_u_row = Eigen::Vector3d(
          ABt_U.at<double>(i, 0), ABt_U.at<double>(i, 1), ABt_U.at<double>(i, 2));
      Eigen::Vector3d abt_v_row = Eigen::Vector3d(
          ABt_V.at<double>(j, 0), ABt_V.at<double>(j, 1), ABt_V.at<double>(j, 2));
      R(i, j) = Dot(abt_v_row, abt_u_row);
    }
  }

  const double det = R(0, 0) * R(1, 1) * R(2, 2) + R(0, 1) * R(1, 2) * R(2, 0) +
                     R(0, 2) * R(1, 0) * R(2, 1) - R(0, 2) * R(1, 1) * R(2, 0) -
                     R(0, 1) * R(1, 0) * R(2, 2) - R(0, 0) * R(1, 2) * R(2, 1);

  if (det < 0) {
    R(2, 0) = -R(2, 0);
    R(2, 1) = -R(2, 1);
    R(2, 2) = -R(2, 2);
  }

  t(0) = pc0(0) - Dot(R.row(0), pw0);
  t(1) = pc0(1) - Dot(R.row(1), pw0);
  t(2) = pc0(2) - Dot(R.row(2), pw0);
}
} // namespace models
} // namespace coral
