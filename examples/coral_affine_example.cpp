
// TODO: Fix includes
#include "../coral_gpu/include/coral_cuda_wrapper.h"
#include "../open_coral/include/extractors/coral_curvature_extractor.h"
#include "../open_coral/include/model_initialiser.h"
#include "../open_coral/include/models/coral_affine_model.h"
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
// using namespace cv;

double Rho(const double x, const double sigma) {
  // really should be `rho`, but using p anyway
  // Geman-McClure Kernel
  double xsq = pow(x, 2);
  double ssq = pow(sigma, 2);
  return (xsq / (xsq + ssq));
}
double Weight(const double x, const double sigma) {
  return (1.0 - Rho(x, sigma));
}
void HillClimb(const cv::Mat &curvature, std::vector<Eigen::Vector2i> &points1,
               std::vector<Eigen::Vector2d> points1_precise, double lmd) {
  // kappa_pad = np.pad(kappa, ((1,1),(1,1)),
  //     mode='constant', constant_values=-np.inf)

  size_t num_points = points1.size();
  std::vector<bool> msk(num_points, true);

  cv::Mat curvature_pad;

  copyMakeBorder(curvature, curvature_pad, 1, 1, 1, 1, cv::BORDER_DEFAULT,
                 cv::Scalar(0));

  std::vector<int> patch{-1, 0, 1};
  for (int k = 0; k < points1.size(); k++) {
    if (msk[k]) {
      auto F = static_cast<double>(
          curvature.at<uchar>(points1[k].y(), points1[k].x()));
      while (true) {
        std::vector<double> Fs;
        std::vector<Eigen::Vector2i> ds;
        for (int i = 0; i < patch.size(); i++) {
          int di = patch[i];
          for (int dj : patch) {
            ds.emplace_back(Eigen::Vector2i(di, dj));
            if (di == 0 && dj == 0) {
              Fs.push_back(F);
              continue;
            }

            // double f;

            Eigen::Vector2d pt = points1[k].cast<double>() +
                                 Eigen::Vector2d(di, dj) - points1_precise[k];
            double d_pt = sqrt(pow(pt.x(), 2) + pow(pt.y(), 2));

            double f =
                static_cast<double>(curvature_pad.at<uchar>(
                    points1[k].y() + (1 + dj), points1[k].x() + (1 + di))) +
                lmd * Rho(d_pt, 0.1);
            // f = f_;
            Fs.push_back(f);
          }
        }
        // colwise argmax
        unsigned int sel;
        double FsMax = -100000;

        for (int j = 0; j < Fs.size(); j++) {
          if (FsMax < Fs[j]) {
            FsMax = Fs[j];
            sel = j;
          }
        }

        F = FsMax;

        // update pt
        points1[k] = points1[k] + ds[sel];

        // recalc msk
        if (sel == 4) {
          msk[k] = false;
          break;
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);

  ExtractorParams params{};
  params.scale_factor = 0.25;
  params.inverse_scale_factor = 4;

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
               image_1_left, cv::Size(), 1, 1);

    cv::Mat image_1_right;
    cv::resize(cv::imread(directory_right + zero_pad + std::to_string(i) +
                          "_right.png"),
               image_1_right, cv::Size(), 1, 1);

    cv::Mat image_2_left;
    cv::resize(cv::imread(directory_left + zero_pad +
                          std::to_string(i + delta) +
                          "_left"
                          ".png"),
               image_2_left, cv::Size(), 1, 1);

    cv::Mat image_2_right;
    cv::resize(cv::imread(directory_right + zero_pad +
                          std::to_string(i + delta) +
                          "_right"
                          ".png"),
               image_2_right, cv::Size(), 1, 1);

    cv::Mat image_1_left_grey, image_1_right_grey, image_2_left_grey,
        image_2_right_grey;
    cv::cvtColor(image_1_left, image_1_left_grey, cv::COLOR_RGB2GRAY);
    cv::cvtColor(image_1_right, image_1_right_grey, cv::COLOR_RGB2GRAY);

    cv::cvtColor(image_2_left, image_2_left_grey, cv::COLOR_RGB2GRAY);
    cv::cvtColor(image_2_right, image_2_right_grey, cv::COLOR_RGB2GRAY);

    //    FeatureVectorSPtr curv_features_1 =
    //        curv_extractor.ExtractFeatures(image_1_left_grey);
    //    FeatureVectorSPtr curv_features_2 =
    //        curv_extractor.ExtractFeatures(image_2_left_grey);

    FeatureVectorSPtr matched_features =
        curv_extractor.ExtractAndMatchFeatures(image_1_left, image_2_left);

    ModelInitialiserParams mi_params(100, 0.95);

    ModelInitialiser<CoralModelAffine> affine_model_initialiser(mi_params);

    int num_models = 1;
    float threshold = 3;

    ModelVectorSPtr affine_models(new ModelVector);
    affine_model_initialiser.Initialise(matched_features, num_models, threshold,
                                        affine_models);

    LOG(INFO) << "Number of models is " << affine_models->size();

    coral::optimiser::CoralOptimiserParams coral_params{};

    coral_params.num_features = matched_features->size();
    coral_params.num_neighbours = 2;
    coral_params.outlier_threshold = 3;

    coral_params.num_labels = affine_models->size() + 1;
    coral_params.num_iterations = 10;
    coral_params.num_loops = 1;

    coral_params.lambda = 10.0;
    coral_params.beta = 0;
    coral_params.nu = 0.125;
    coral_params.alpha = 0.0125;
    coral_params.tau = 0.0125;
    coral_params.update_models = false;

    cv::Mat neighbour_index, neighbour_transpose;
    cuda::coral_wrapper::CoralCudaWrapper<CoralModelAffine> cuda_optimiser(
        coral_params);

    LOG(INFO) << "Number of  models is " << affine_models->size();
    LOG(INFO) << "Number of features is " << matched_features->size();
    EnergyMinimisationResult result =
        cuda_optimiser.EnergyMinimisation(matched_features, affine_models);

    viewer::CoralViewer<cv::Mat> image_viewer;

    image_viewer.VisualiseLabels(image_1_left, matched_features, affine_models,
                                 result.DiscreteLabel);

    //    cv::Mat descriptors_1 =
    //        curv_extractor.FeatureDescriptors(image_1_left_grey,
    //        curv_features_1);
    //    cv::Mat descriptors_2 =
    //        curv_extractor.FeatureDescriptors(image_2_left_grey,
    //        curv_features_2);
    //
    //    //-- Step 2: Matching descriptor vectors with a brute force matcher
    //    // Since SURF is a floating-point descriptor NORM_L2 is used
    //    cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING,
    //    true); std::vector<cv::DMatch> matches; bf->match(descriptors_1,
    //    descriptors_2, matches);
    //
    //    //-- Get matched keypoints
    //    std::vector<cv::Point2f> points_1, points_2;
    //    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    //    std::vector<Eigen::Vector2d> initial_pos;
    //    std::vector<Eigen::Vector2d> actual_pos;
    //    for (const auto &match : matches) {
    //      features::FeatureSPtr feat_1 = curv_features_1->at(match.queryIdx);
    //      initial_pos.emplace_back(feat_1->GetFeatureValue());
    //      points_1.emplace_back(cv::Point2f(feat_1->GetFeatureValue()(0),
    //                                        feat_1->GetFeatureValue()(1)));
    //
    //      keypoints_1.emplace_back(cv::Point2f(feat_1->GetFeatureValue()(0),
    //                                           feat_1->GetFeatureValue()(1)),
    //                               1);
    //      features::FeatureSPtr feat_2 = curv_features_2->at(match.trainIdx);
    //      actual_pos.emplace_back(feat_2->GetFeatureValue());
    //      points_2.emplace_back(cv::Point2f(feat_2->GetFeatureValue()(0),
    //                                        feat_2->GetFeatureValue()(1)));
    //      keypoints_2.emplace_back(cv::Point2f(feat_2->GetFeatureValue()(0),
    //                                           feat_2->GetFeatureValue()(1)),
    //                               1);
    //    }
    //
    //    // Estimate the affine transform
    //    cv::Mat mat_pts_1(points_1);
    //    cv::Mat mat_pts_2(points_2);
    //
    //    cv::Mat M;
    //    M = cv::estimateAffine2D(points_1, points_2);
    //    Eigen::MatrixXd affine = Eigen::MatrixXd::Zero(2, 3);
    //    cv::cv2eigen(M, affine);
    //    Eigen::Matrix2d A_ = affine.block(0, 0, 2, 2);
    //    Eigen::Vector2d b_ = affine.block(0, 2, 2, 1);
    //
    //    // Predict the next positions
    //    std::vector<Eigen::Vector2d> predicted_pos_precision;
    //    std::vector<Eigen::Vector2i> predicted_pos;
    //    // vector<int> predictedP2KeyPoint_idx;
    //
    //    int feat_no = 0;
    //    Eigen::MatrixXd affine_false = Eigen::MatrixXd::Zero(2, 3);
    //    affine_false << 1, 0, 0, 0, 1, 0;
    //    A_ = affine.block(0, 0, 2, 2);
    //    b_ = affine.block(0, 2, 2, 1);
    //    for (const auto &initial : initial_pos) {
    //      Eigen::Vector2d pred = A_ * initial + b_;
    //      Eigen::Vector2d actual = actual_pos[feat_no];
    //      if (pred(0) > 0 && pred(1) > 0 && pred(1) < 960 && pred(0) < 1280) {
    //        predicted_pos.emplace_back(pred.cast<int>());
    //        predicted_pos_precision.emplace_back(pred);
    //      }
    //      //      if((actual-pred).norm()<30) {
    //      //        LOG(INFO) << "Initial position is " <<
    //      initial.transpose();
    //      //        LOG(INFO) << "Final position   is " << actual.transpose();
    //      //        LOG(INFO) << "Pred position    is " << pred.transpose();
    //      //        LOG(INFO) << "Difference is " << (actual - pred).norm();
    //      //        predicted_pos.emplace_back(pred.cast<int>());
    //      //        predicted_pos_precision.emplace_back(pred);
    //      //      }
    //      feat_no++;
    //    }
    //
    //    double lmd = 5;
    //    HillClimb(
    //        coral::extractors::CurvatureExtractor::GetCurvature(image_2_left_grey),
    //        predicted_pos, predicted_pos_precision, lmd);
    //
    //    std::vector<cv::KeyPoint> tracked_features;
    //    for (auto pred : predicted_pos) {
    //      tracked_features.push_back(
    //          cv::KeyPoint(cv::Point2f(pred(0), pred(1)), 1));
    //    }
    //
    //    LOG(INFO) << "M is " << M;
    //    cv::Mat img;
    //    cv::drawKeypoints(image_1_left, keypoints_1, image_1_left);
    //    cv::drawKeypoints(image_2_left, tracked_features, image_2_left);
    //    //-- Show detected matches
    //    cv::imshow("Matched keypoints", image_1_left);
    //    cv::imshow("Matched keypoints 2", image_2_left);
    //    cv::waitKey(0);
  }
  return 0;
}
