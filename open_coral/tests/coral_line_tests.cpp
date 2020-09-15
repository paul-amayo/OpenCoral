#define BOOST_TEST_MODULE My Test
#include "../include/models/coral_model_line.h"
#include <boost/test/included/unit_test.hpp>

struct LineFixture {

  boost::shared_ptr<coral::models::CoralModelLine> line_model;


  LineFixture() {
    line_model=boost::make_shared<coral::models::CoralModelLine>();

  }
};
BOOST_AUTO_TEST_CASE(line_test) {

  int num_points = 10;
  int num_dimensions = 2;

  Eigen::VectorXd x_values(num_points);
  Eigen::VectorXd y_values(num_points);

  int feat_no = 0;
  // y=2x+3
  for (int i = 0; i < num_points; ++i) {
    x_values(i) = i;
    y_values(i) = 2 * i + 3;
  }

  Eigen::VectorXd line_params =
      coral::models::CoralModelLine ::CalculateLeastSquaresModel(x_values,
                                                                 y_values);
  std::cout << "Line params is " << line_params;

  BOOST_CHECK_CLOSE(line_params(0), -0.894, 1);
  BOOST_CHECK_CLOSE(line_params(1), 0.447, 1);
  BOOST_CHECK_CLOSE(line_params(2), -1.34, 1);
}