#ifndef CORAL_PNP_MODEL_H_
#define CORAL_PNP_MODEL_H_

#include "../features/coral_feature.h"
#include "../features/coral_feature_stereo_correspondance.h"
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

private:
  void CheckInliers();
  bool Refine();

  // Functions from the original EPnP code
  void SetMaxCorrespondences(const int n);

  void ResetCorrespondences();

  void
  AddCorrespondences(const features::CoralFeatureStereoCorrespondanceVectorSPtr
                         &stereo_features);

  double ComputePose(const Rot &R, Trans T);

  void RelativeError(double &rot_err, double &transl_err, const Rot R_true,
                     Trans T_true, const Rot R_est, Trans T_est);

  void print_pose(const Rot R, const Trans t);

  double ReprojError(const Rot R, const Trans t);

  void ChooseCtrlPoints(void);

  void ComputeBarycentricCoordinates(void);

  void FillM(cv::Mat &M, const int row, const Eigen::Vector4d &alphas,
             const double u, const double v) const;

  void fill_M(CvMat *M, const int row, const double *as, const double u,
              const double v);

  void ComputeCcs(const Eigen::MatrixXd &betas, const cv::Mat ut);

  void ComputePcs(void);

  void SolveSign(void);

  static void FindBetasApprox1(const cv::Mat &L_6x10, const cv::Mat &Rho,
                               Eigen::Vector4d &betas);

  static void FindBetasApprox2(const cv::Mat &L_6x10, const cv::Mat &Rho,
                               Eigen::Vector4d &betas);

  void FindBetasApprox3(const cv::Mat &L_6x10, const cv::Mat &Rho,
                        Eigen::Vector4d &betas);

  static void QrSolve(cv::Mat &A, cv::Mat &b, cv::Mat &X);

  double Dot(const Eigen::MatrixXd v1, const Eigen::MatrixXd v2);

  static double DistSquared(const Eigen::MatrixXd p1, const Eigen::MatrixXd p2);

  void ComputeRho(cv::Mat &rho);

  void compute_rho(double *rho);
  void compute_L_6x10(const double *ut, double *l_6x10);

  double dot(const double *v1, const double *v2);
  double dist2(const double *p1, const double *p2);

  void ComputeL6x10(const cv::Mat &ut, cv::Mat &L_6x10);

  void GaussNewton(const cv::Mat L_6x10, const cv::Mat Rho,
                   Eigen::Vector4d& current_betas);

  static void ComputeAandbGN(const cv::Mat &l_6x10, const cv::Mat rho,
                             Eigen::MatrixXd cb, cv::Mat A, cv::Mat b);

  double ComputeRotTrans(const cv::Mat ut, const Eigen::MatrixXd betas, Rot &R,
                         Trans &t);

  double compute_R_and_t(const double * ut, const double * betas,
                         double R[3][3], double t[3]);
  void estimate_R_and_t(double R[3][3], double t[3]);

  void compute_ccs(const double * betas, const double * ut);
  void compute_pcs(void);

  void solve_for_sign(void);

  double reprojection_error(const double R[3][3], const double t[3]);


  void EstimateRotTrans(Rot &R, Trans &t);

  void MatToQuat(const Rot R, Eigen::Vector4d &q);

  void find_betas_approx_1(const CvMat *L_6x10, const CvMat *Rho,
                           double *betas);
  void find_betas_approx_2(const CvMat *L_6x10, const CvMat *Rho,
                           double *betas);
  void find_betas_approx_3(const CvMat *L_6x10, const CvMat *Rho,
                           double *betas);
  void qr_solve(CvMat *A, CvMat *b, CvMat *X);

  void gauss_newton(const CvMat *L_6x10, const CvMat *Rho,
                    double current_betas[4]);
  void compute_A_and_b_gauss_newton(const double *l_6x10, const double *rho,
                                    double cb[4], CvMat *A, CvMat *b);

private:
  double u_c_, v_c_, f_u_, f_v_;

  std::vector<Eigen::Vector4d> alphas_;
  std::vector<Eigen::Vector3d> pws_, pcs_;
  std::vector<Eigen::Vector2d> us_;

  // Use for comparison with previous values
  double *pws, *us, *alphas, *pcs;
  double cws[4][3], ccs[4][3];
  double cws_determinant;

  int MaxCorrespondences_;
  int num_correspondences_;

  Eigen::MatrixXd cws_, ccs_;

  // 2D Points
  std::vector<cv::Point2f> mvP2D;
  std::vector<float> mvSigma2;

  // 3D Points
  std::vector<cv::Point3f> mvP3Dw;

  // Index in Frame
  std::vector<size_t> mvKeyPointIndices;

  // Current Estimation
  Rot mRi;
  Trans mti;

  cv::Mat mTcwi;
  std::vector<bool> mvbInliersi;
  int mnInliersi;

  // Current Ransac State
  int mnIterations;
  std::vector<bool> mvbBestInliers;
  int mnBestInliers;
  cv::Mat mBestTcw;

  // Refined
  cv::Mat mRefinedTcw;
  std::vector<bool> mvbRefinedInliers;
  int mnRefinedInliers;

  // Number of Correspondences
  int N;

  // Indices for random selection [0 .. N-1]
  std::vector<size_t> mvAllIndices;

  // RANSAC probability
  double mRansacProb;

  // RANSAC min inliers
  int mRansacMinInliers;

  // RANSAC max iterations
  int mRansacMaxIts;

  // RANSAC expected inliers/total ratio
  float mRansacEpsilon;

  // RANSAC Threshold inlier/outlier. Max error e = dist(P1,T_12*P2)^2
  float mRansacTh;

  // RANSAC Minimun Set used at each iteration
  int mRansacMinSet;

  // Max square error associated with scale level. Max error =
  // th*th*sigma(level)*sigma(level)
  std::vector<float> mvMaxError;
};

} // namespace models
} // namespace coral
#endif // CORAL_PNP_MODEL_H_