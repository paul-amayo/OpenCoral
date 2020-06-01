#define BOOST_TEST_MODULE My Test
#include "../include/coral.h"
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_CASE(simplex_test) {

  int num_points = 10;
  int num_dimensions = 2;

  Eigen::VectorXd x_values(num_points);
  Eigen::Matrix3d K;

  K<<0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9;

  std::cout<<"K is "<<K;

  Eigen::Matrix3d K_simplex=coral::optimiser::CoralOptimiser<int
      >::SimplexProjectionVector(K);

  std::cout<<"K out is "<<K_simplex;


  BOOST_CHECK_CLOSE(K_simplex(0,0)+K_simplex(0,1)+K_simplex(0,2),1,1);
  BOOST_CHECK_CLOSE(K_simplex(1,0)+K_simplex(1,1)+K_simplex(1,2),1,1);
  BOOST_CHECK_CLOSE(K_simplex(2,0)+K_simplex(2,1)+K_simplex(2,2),1,1);
}