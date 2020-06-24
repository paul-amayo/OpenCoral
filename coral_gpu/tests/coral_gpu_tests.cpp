#define BOOST_TEST_MODULE My Test
#include "../include/coral_cuda_wrapper.h"
#include "../open_coral/include/features/coral_feature_point.h"
#include <boost/test/included/unit_test.hpp>
#include <boost/make_shared.hpp>

#include <glog/logging.h>

struct CoralCudaWrapperFixture{

};
BOOST_AUTO_TEST_CASE(simplex_test) {

  int num_points = 10;
  int num_dimensions = 2;

  Eigen::VectorXd x_values(num_points);
  Eigen::Matrix3d K;

  K<<0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9;

  Eigen::Matrix3d K_simplex=cuda::coral_wrapper::CoralCudaWrapper<int
      >::SimplexProjectionVector(K);


  BOOST_CHECK_CLOSE(K_simplex(0,0)+K_simplex(0,1)+K_simplex(0,2),1,1);
  BOOST_CHECK_CLOSE(K_simplex(1,0)+K_simplex(1,1)+K_simplex(1,2),1,1);
  BOOST_CHECK_CLOSE(K_simplex(2,0)+K_simplex(2,1)+K_simplex(2,2),1,1);
}

BOOST_AUTO_TEST_CASE(neighbour_wrapper){

int num_points=9;
int num_dimensions=2;

coral::features::FeatureVectorSPtr  point_features(new coral::features::FeatureVector);

point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(0,0.01)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(0,1.02)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(0.03,2)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(1.04,0)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(1,1.06)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(1,2.07)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(2,0.02)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(2,1.08)));
point_features->push_back(boost::make_shared<coral::features::CoralFeaturePoint>(Eigen::Vector2d(2.10,2)));


coral::optimiser::CoralOptimiserParams params{};
params.num_features=num_points;
params.num_neighbours=2;
params.max_neighbours=3;

cuda::coral_wrapper::CoralCudaWrapper<int> cuda_optimiser(params);

coral::optimiser::CoralOptimiser<int> optimiser(params);

optimiser.FindNearestNeighbours(point_features);

cv::Mat neighbour_index, neighbour_transpose;
cuda_optimiser.FindNearestNeighbours(point_features,neighbour_index, neighbour_transpose);
Eigen::SparseMatrix<double> nabla=optimiser.GetGradient();
cv::Mat wrapped_neighbour_index,wrapped_neighbour_transpose;
cuda_optimiser.WrapNeighbourHood(nabla, wrapped_neighbour_index, wrapped_neighbour_transpose);

for(int i=0;i<num_points;++i){
	for(int j=0;j<num_dimensions;++j){
	BOOST_CHECK_CLOSE(neighbour_index.at<float>(j,i), wrapped_neighbour_index.at<float>(j,i),1);
	}
	for(int j=0;j<params.max_neighbours;++j){
	BOOST_CHECK_CLOSE(neighbour_transpose.at<float>(j,i),wrapped_neighbour_transpose.at<float>(j,i),1);
	}
}

}
