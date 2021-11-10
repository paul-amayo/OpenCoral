//
// Created by paulamayo on 2021/10/31.
//

#ifndef SRC_CORAL_CURVATURE_EXTRACTOR_H
#define SRC_CORAL_CURVATURE_EXTRACTOR_H

#include "../features/coral_feature_point.h"
#include <opencv2/core/mat.hpp>
#include <vector>

namespace coral {
namespace extractors {

struct ExtractorParams {
  int num_features;
  int num_levels;
  float scale_factor;
  int initial_fast_threshold;
  int minimum_fast_threshold;
  int patch_size;
  int half_patch_size;
  int edge_threshold;
};

class CurvatureExtractor {
public:
  explicit CurvatureExtractor(ExtractorParams extractor_params);

  ~CurvatureExtractor() = default;

  std::vector<cv::Mat> mvImagePyramid;

  features::FeatureVectorSPtr ExtractFeatures(const cv::Mat &image);

  cv::Mat FeatureDescriptors(const cv::Mat &image,
                             features::FeatureVectorSPtr point_features);

  cv::Mat NonMaximalSuppression(cv::Mat image, bool remove_plateaus);

  cv::Mat GetCurvature(cv::Mat image);
  cv::Mat GetLocalMaxima(cv::Mat image);

private:
  ExtractorParams extractor_params_;
  std::vector<int> mnFeaturesPerLevel_;
  std::vector<float> mvScaleFactor_;
  std::vector<float> mvInvScaleFactor_;
  std::vector<float> mvLevelSigma2_;
  std::vector<float> mvInvLevelSigma2_;
};
} // namespace extractors
} // namespace coral

#endif // SRC_CORAL_CURVATURE_EXTRACTOR_H
