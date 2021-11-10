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
cv::Mat CurvatureExtractor::GetCurvature(cv::Mat image) {
  cv::Mat blur;
  cv::GaussianBlur(image, blur, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
  const int scale = 1;
  const int delta = 0;
  const int ddepth = CV_8U; // CV_64F
  const int ksize = 1;
  // GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
  cv::Mat fx, fy, fxx, fyy, fxy;
  cv::Sobel(blur, fx, ddepth, 1, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(blur, fy, ddepth, 0, 1, ksize, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(blur, fxx, ddepth, 2, 0, ksize, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(blur, fyy, ddepth, 0, 2, ksize, scale, delta, cv::BORDER_DEFAULT);
  cv::Sobel(blur, fxy, ddepth, 1, 1, ksize, scale, delta, cv::BORDER_DEFAULT);

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
cv::Mat CurvatureExtractor::GetLocalMaxima(cv::Mat image) {
  //#pragma omp parallel for
  cv::Mat th, mask;
  cv::threshold(image, th, 200, 255, cv::THRESH_TOZERO);
  mask = NonMaximalSuppression(th, 1);

  return mask;
}
//------------------------------------------------------------------------------
cv::Mat CurvatureExtractor::FeatureDescriptors(
    const cv::Mat &image, features::FeatureVectorSPtr point_features) {
  cv::Mat descriptors_;
  std::vector<cv::KeyPoint> keypoints;
  for (auto feature : *point_features) {
    cv::xfeatures2d::BriefDescriptorExtractor brief;
    float x = feature->GetFeatureValue()(0);
    float y = feature->GetFeatureValue()(1);
    keypoints.push_back(cv::KeyPoint(cv::Point2f(x, y), 1));
  }

  cv::xfeatures2d::BriefDescriptorExtractor brief;
  brief.compute(image, keypoints, descriptors_);
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
  // cv::waitKey(0);

  cv::Mat localPoints = GetLocalMaxima(kappa);
  cv::Mat image_show = image.clone();
  cv::Size imgsize = image.size();
  std::vector<const float *> tmpCorners;
  // collect list of pointers to features - put them into temporary image
  // Mat mask = _mask.getMat();
  for (int y = 10; y < imgsize.height - 10; y++) {
    const auto *eig_data = (const float *)localPoints.ptr(y);

    for (int x = 10; x < imgsize.width - 10; x++) {
      float val = eig_data[x];
      if (val != 0) //&& val == tmp_data[x] && (!mask_data || mask_data[x]) )
        tmpCorners.push_back(eig_data + x);
    }
  }

  size_t i, j, total = tmpCorners.size(), ncorners = 0;

  int minDistance = 10;

  // Partition the image into larger grids
  int w = image.cols;
  int h = image.rows;

  const int cell_size = cvRound(minDistance);
  const int grid_width = (w + cell_size) / cell_size;
  const int grid_height = (h + cell_size) / cell_size;

  std::vector<std::vector<cv::Point2f>> grid(grid_width * grid_height);

  std::vector<cv::KeyPoint> keypoints;

  minDistance *= minDistance;

  for (i = 0; i < total; i++) {

    int ofs = (int)((const uchar *)tmpCorners[i] - localPoints.ptr());
    int y = (int)(ofs / localPoints.step);
    int x = (int)((ofs - y * localPoints.step) / sizeof(uchar));

    bool good = true;

    int x_cell = x / cell_size;
    int y_cell = y / cell_size;

    int x1 = x_cell - 1;
    int y1 = y_cell - 1;
    int x2 = x_cell + 1;
    int y2 = y_cell + 1;

    // boundary check
    x1 = std::max(0, x1);
    y1 = std::max(0, y1);
    x2 = std::min(grid_width - 1, x2);
    y2 = std::min(grid_height - 1, y2);

    for (int yy = y1; yy <= y2; yy++) {
      for (int xx = x1; xx <= x2; xx++) {
        std::vector<cv::Point2f> &m = grid[yy * grid_width + xx];

        if (m.size()) {
          for (j = 0; j < m.size(); j++) {
            float dx = x - m[j].x;
            float dy = y - m[j].y;

            if (dx * dx + dy * dy < minDistance) {
              good = false;
              break;
            }
          }
        }
        if (!good)
          break;
      }
      if (!good)
        break;
    }
    if (good) {

      grid[y_cell * grid_width + x_cell].push_back(
          cv::Point2f((float)x, (float)y));

      features::FeatureSPtr curr_feature =
          boost::make_shared<features::CoralFeaturePoint>(
              Eigen::Vector2d((float)x, (float)y));

      curv_features->push_back(curr_feature);
      keypoints.emplace_back(x, y, 1);
      ++ncorners;
    }
  }
  cv::drawKeypoints(image_show, keypoints, image_show);
  cv::imshow("Image keypoints", image_show);
  cv::waitKey(15);
  return curv_features;
}

} // namespace extractors
} // namespace coral
