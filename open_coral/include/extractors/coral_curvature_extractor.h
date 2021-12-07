//
// Created by paulamayo on 2021/10/31.
//

#ifndef SRC_CORAL_CURVATURE_EXTRACTOR_H
#define SRC_CORAL_CURVATURE_EXTRACTOR_H

#include "../features/coral_feature_curvature.h"
#include "../features/coral_feature_point.h"
#include <opencv2/core/mat.hpp>
#include <vector>

namespace coral {
namespace extractors {

struct ExtractorParams {
  int num_features;
  int num_levels;
  float scale_factor;
  float inverse_scale_factor;
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

  features::FeatureVectorSPtr ExtractAndMatchFeatures(const cv::Mat &image_1,
                                                      const cv::Mat &image_2);

  features::FeatureVectorSPtr
  MatchFeatures(cv::Mat image_1, cv::Mat image_2,
                features::FeatureVectorSPtr &feat_image_1,
                features::FeatureVectorSPtr &feat_image_2);

  features::FeatureVectorSPtr ExtractFeatures(const cv::Mat &image_1);

  static cv::Mat
  FeatureDescriptors(const cv::Mat &image,
                     features::FeatureVectorSPtr &point_features);

  cv::Mat NonMaximalSuppression(cv::Mat image, bool remove_plateaus);

  static cv::Mat GetCurvature(const cv::Mat &image);
  static cv::Mat GetLocalMaxima(const cv::Mat &image, int w_size,
                                bool remove_plateaus);

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
