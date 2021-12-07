#include "../include/extractors/coral_curvature_extractor.h"
#include <Eigen/Dense>
#include <boost/make_shared.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

namespace coral {
namespace extractors {
//------------------------------------------------------------------------------
CurvatureExtractor::CurvatureExtractor(ExtractorParams extractor_params)
    : extractor_params_(extractor_params) {}
//------------------------------------------------------------------------------
cv::Mat CurvatureExtractor::GetCurvature(const cv::Mat &image) {
  // const cv::Mat& blur = image;
  //  cv::GaussianBlur(image, blur, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
  const int scale = 1;
  const int delta = 0;
  const int ddepth = CV_8U; // CV_64F
  const int ksize = 1;
  // GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
  cv::Mat fx, fy, fxx, fyy, fxy;
  cv::Sobel(image, fx, ddepth, 1, 0, ksize);
  cv::Sobel(image, fy, ddepth, 0, 1, ksize);
  cv::Sobel(image, fxx, ddepth, 2, 0, ksize);
  cv::Sobel(image, fyy, ddepth, 0, 2, ksize);
  cv::Sobel(image, fxy, ddepth, 1, 1, ksize);

  cv::Mat k =
      (fy.mul(fy)).mul(fxx) - 2 * (fx.mul(fy)).mul(fxy) + (fx.mul(fx)).mul(fyy);

  return k;
}
//------------------------------------------------------------------------------
cv::Mat CurvatureExtractor::NonMaximalSuppression(cv::Mat image,
                                                  bool remove_plateaus) {
  cv::Mat mask;
  cv::dilate(image, mask, cv::Mat());
  cv::compare(image, mask, mask, cv::CMP_GE);

  // optionally filter out pixels that are equal to the local minimum
  // ('plateaus')
  if (remove_plateaus) {
    cv::Mat non_plateau_mask;
    cv::erode(image, non_plateau_mask, cv::Mat());
    cv::compare(image, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
    cv::bitwise_and(mask, non_plateau_mask, mask);
  }
  return mask;
}
//------------------------------------------------------------------------------
cv::Mat CurvatureExtractor::GetLocalMaxima(const cv::Mat &image, int w_size,
                                           bool remove_plateaus) {

  cv::Mat thresh_image;
  cv::threshold(image, thresh_image, 200, 255, cv::THRESH_TOZERO);

  cv::Mat thresh_float;
  thresh_image.convertTo(thresh_float, CV_32FC1);
  thresh_float = thresh_float / 255.0f;

  // Get kernel
  cv::Mat element =
      cv::getStructuringElement(cv::MORPH_RECT, cv::Size(w_size, w_size));

  cv::Mat mask;
  cv::dilate(thresh_float, mask, element);

  cv::compare(thresh_float, mask, mask, cv::CMP_GE);

  if (remove_plateaus) {
    cv::Mat non_plateau_mask;
    cv::erode(thresh_float, non_plateau_mask, element);

    cv::compare(thresh_float, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
    cv::bitwise_and(mask, non_plateau_mask, mask);
  }

  // mask = NonMaximalSuppression(th, 1);

  return mask;
}
//------------------------------------------------------------------------------
cv::Mat CurvatureExtractor::FeatureDescriptors(
    const cv::Mat &image, features::FeatureVectorSPtr &point_features) {
  cv::Mat descriptors_;
  std::vector<cv::KeyPoint> keypoints;
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief =
      cv::xfeatures2d::BriefDescriptorExtractor::create();

  features::FeatureVectorSPtr valid_features =
      boost::make_shared<features::FeatureVector>();
  for (const auto &feature : *point_features) {
    float x = feature->GetFeatureValue()(0);
    float y = feature->GetFeatureValue()(1);
    keypoints.emplace_back(cv::Point2f(x, y), 1);
    std::vector<cv::KeyPoint> keypoints_temp;
    keypoints_temp.emplace_back(cv::Point2f(x, y), 1);
    cv::Mat descriptor;
    brief->compute(image, keypoints_temp, descriptor);
    if (!descriptor.empty()) {
      descriptors_.push_back(descriptor);
      valid_features->push_back(feature);
    }
  }
  LOG(INFO) << "Number of keypoints is " << valid_features->size();

  point_features = valid_features;
  return descriptors_;
}
//------------------------------------------------------------------------------
features::FeatureVectorSPtr
CurvatureExtractor::ExtractFeatures(const cv::Mat &image) {
  // Clear features
  features::FeatureVectorSPtr curv_features =
      boost::make_shared<features::FeatureVector>();
  LOG(INFO) << "Getting curvature";
  cv::Mat kappa = GetCurvature(image); // Done
                                       //  cv::imshow("Curvature", kappa);
                                       //  cv::waitKey(0);

  cv::Mat localPoints = GetLocalMaxima(kappa, 3, true);
  //  cv::imshow("Local maxima", localPoints);
  //  cv::waitKey(0);

  cv::Mat image_show = image.clone();
  cv::Size imgsize = image.size();
  std::vector<const float *> tmpCorners;
  std::vector<cv::KeyPoint> keypoints;
  // collect list of pointers to features - put them into temporary image
  // Mat mask = _mask.getMat();

  localPoints.convertTo(localPoints, CV_32FC1);
  for (int y = 10; y < imgsize.height - 10; y++) {
    for (int x = 10; x < imgsize.width - 10; x++) {
      float val = localPoints.at<float>(y, x);
      if (val != 0) {
        features::FeatureSPtr curr_feature =
            boost::make_shared<features::CoralFeaturePoint>(
                Eigen::Vector2d((float)x, (float)y));

        curv_features->push_back(curr_feature);
        keypoints.emplace_back(x, y, 1);
      }
    }
  }

  //    const auto *eig_data = (const float *)localPoints.ptr(y);
  //
  //    for (int x = 10; x < imgsize.width - 10; x++) {
  //      float val = eig_data[x];
  //      if (val != 0) {
  //        features::FeatureSPtr curr_feature =
  //            boost::make_shared<features::CoralFeaturePoint>(
  //                Eigen::Vector2d((float)x, (float)y));
  //
  //        curv_features->push_back(curr_feature);
  //        keypoints.emplace_back(y, x, 1);
  //      } //&& val == tmp_data[x] && (!mask_data || mask_data[x]) )
  //        //        tmpCorners.push_back(eig_data + x);
  //    }
  //  }

  //  size_t i, j, total = tmpCorners.size(), ncorners = 0;
  //
  //  int minDistance = 10;
  //
  //  // Partition the image into larger grids
  //  int w = image.cols;
  //  int h = image.rows;
  //
  //  const int cell_size = cvRound(minDistance);
  //  const int grid_width = (w + cell_size) / cell_size;
  //  const int grid_height = (h + cell_size) / cell_size;
  //
  //  std::vector<std::vector<cv::Point2f>> grid(grid_width * grid_height);
  //
  //  std::vector<cv::KeyPoint> keypoints;
  //
  //  minDistance *= minDistance;
  //
  //  for (i = 0; i < total; i++) {
  //
  //    int ofs = (int)((const uchar *)tmpCorners[i] - localPoints.ptr());
  //    int y = (int)(ofs / localPoints.step);
  //    int x = (int)((ofs - y * localPoints.step) / sizeof(uchar));
  //
  //    bool good = true;
  //
  //    int x_cell = x / cell_size;
  //    int y_cell = y / cell_size;
  //
  //    int x1 = x_cell - 1;
  //    int y1 = y_cell - 1;
  //    int x2 = x_cell + 1;
  //    int y2 = y_cell + 1;
  //
  //    // boundary check
  //    x1 = std::max(0, x1);
  //    y1 = std::max(0, y1);
  //    x2 = std::min(grid_width - 1, x2);
  //    y2 = std::min(grid_height - 1, y2);
  //
  //    for (int yy = y1; yy <= y2; yy++) {
  //      for (int xx = x1; xx <= x2; xx++) {
  //        std::vector<cv::Point2f> &m = grid[yy * grid_width + xx];
  //
  //        if (m.size()) {
  //          for (j = 0; j < m.size(); j++) {
  //            float dx = x - m[j].x;
  //            float dy = y - m[j].y;
  //
  //            if (dx * dx + dy * dy < minDistance) {
  //              good = false;
  //              break;
  //            }
  //          }
  //        }
  //        if (!good)
  //          break;
  //      }
  //      if (!good)
  //        break;
  //    }
  //    if (good) {
  //
  //      grid[y_cell * grid_width + x_cell].push_back(
  //          cv::Point2f((float)x, (float)y));
  //
  //      features::FeatureSPtr curr_feature =
  //          boost::make_shared<features::CoralFeaturePoint>(
  //              Eigen::Vector2d((float)x, (float)y));
  //
  //      curv_features->push_back(curr_feature);
  //      keypoints.emplace_back(x, y, 1);
  //      ++ncorners;
  //    }
  //  }
  //  cv::drawKeypoints(image_show, keypoints, image_show);
  //  cv::imshow("Image keypoints", image_show);
  //  cv::waitKey(0);
  return curv_features;
}
//------------------------------------------------------------------------------
features::FeatureVectorSPtr
CurvatureExtractor::ExtractAndMatchFeatures(const cv::Mat &image_1,
                                            const cv::Mat &image_2) {

  cv::Mat image_1_resized;
  cv::resize(image_1, image_1_resized, cv::Size(),
             extractor_params_.scale_factor, extractor_params_.scale_factor);

  cv::Mat image_2_resized;
  cv::resize(image_2, image_2_resized, cv::Size(),
             extractor_params_.scale_factor, extractor_params_.scale_factor);

  cv::Mat image_1_grey, image_2_grey;
  cv::cvtColor(image_1_resized, image_1_grey, cv::COLOR_RGB2GRAY);
  cv::cvtColor(image_2_resized, image_2_grey, cv::COLOR_RGB2GRAY);

  features::FeatureVectorSPtr feat_image_1 = ExtractFeatures(image_1_grey);
  features::FeatureVectorSPtr feat_image_2 = ExtractFeatures(image_2_grey);

  features::FeatureVectorSPtr matched_features =
      boost::make_shared<features::FeatureVector>();
  cv::Mat descriptors_1 = FeatureDescriptors(image_1_resized, feat_image_1);
  cv::Mat descriptors_2 = FeatureDescriptors(image_2_resized, feat_image_2);

  std::vector<cv::KeyPoint> keypoints_1;
  std::vector<cv::KeyPoint> keypoints_2;

  //-- Step 2: Matching descriptor vectors with a brute force matcher
  // Since SURF is a floating-point descriptor NORM_L2 is used
  cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  bf->match(descriptors_1, descriptors_2, matches);
  //-- Get matched keypoints

  for (const auto &match : matches) {
    Eigen::Vector2d point_1 =
        feat_image_1->at(match.queryIdx)->GetFeatureValue() *
        extractor_params_.inverse_scale_factor;
    Eigen::Vector2d point_2 =
        feat_image_2->at(match.trainIdx)->GetFeatureValue() *
        extractor_params_.inverse_scale_factor;

    features::FeatureSPtr curr_feature =
        boost::make_shared<features::CoralFeatureCurvature>(point_1, point_2);
    keypoints_1.emplace_back(point_1(0), point_1(1), 1);

    matched_features->push_back(curr_feature);
  }
  cv::Mat image_1_show = image_1.clone();
  cv::drawKeypoints(image_1, keypoints_1, image_1_show);
//  cv::imshow("Image Keypoints", image_1_show);
//  cv::waitKey(0);

  return matched_features;
}
//------------------------------------------------------------------------------
features::FeatureVectorSPtr
CurvatureExtractor::MatchFeatures(cv::Mat image_1, cv::Mat image_2,
                                  features::FeatureVectorSPtr &feat_image_1,
                                  features::FeatureVectorSPtr &feat_image_2) {

  features::FeatureVectorSPtr matched_features =
      boost::make_shared<features::FeatureVector>();
  cv::Mat descriptors_1 = FeatureDescriptors(image_1, feat_image_1);
  cv::Mat descriptors_2 = FeatureDescriptors(image_2, feat_image_2);

  //-- Step 2: Matching descriptor vectors with a brute force matcher
  // Since SURF is a floating-point descriptor NORM_L2 is used
  cv::Ptr<cv::BFMatcher> bf = cv::BFMatcher::create(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  bf->match(descriptors_1, descriptors_2, matches);

  //-- Get matched keypoints
  for (const auto &match : matches) {
    Eigen::Vector2d point_1 =
        feat_image_1->at(match.queryIdx)->GetFeatureValue();
    Eigen::Vector2d point_2 =
        feat_image_2->at(match.trainIdx)->GetFeatureValue();

    features::FeatureSPtr curr_feature =
        boost::make_shared<features::CoralFeatureCurvature>(point_1, point_2);

    matched_features->push_back(curr_feature);
  }

  return matched_features;
}
} // namespace extractors
} // namespace coral
