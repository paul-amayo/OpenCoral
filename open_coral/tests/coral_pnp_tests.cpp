#define BOOST_TEST_MODULE My Test
#include "../include/models/coral_pnp_model.h"
#include <boost/test/included/unit_test.hpp>
#include <boost/make_shared.hpp>


using namespace coral;

struct PnpModelFixture{

  features::FeatureVectorSPtr stereo_features;
  boost::shared_ptr<models::CoralPNPModel> stereo_model;
  Eigen::MatrixXd K;

PnpModelFixture(){

  stereo_model=boost::make_shared<coral::models::CoralPNPModel>();

  stereo_features = boost::make_shared<coral::features::FeatureVector>();
  stereo_features->push_back(
      boost::make_shared<coral::features::CoralFeatureStereoCorrespondance>(
          Eigen::Vector2d(2, 9.01),Eigen::Vector3d(1, 5,9)));
  stereo_features->push_back(
      boost::make_shared<coral::features::CoralFeatureStereoCorrespondance>(
          Eigen::Vector2d(15, 17.01),Eigen::Vector3d(5, 26,19)));
  stereo_features->push_back(
      boost::make_shared<coral::features::CoralFeatureStereoCorrespondance>(
          Eigen::Vector2d(23, 20.1),Eigen::Vector3d(3, 17,31)));
  stereo_features->push_back(
      boost::make_shared<coral::features::CoralFeatureStereoCorrespondance>(
          Eigen::Vector2d(8, 24.01),Eigen::Vector3d(21, 8,5)));


  K=Eigen::MatrixXd::Zero(3,3);
  K << 0.1, 0, 0.3, 0, 0.5, 0.6, 0, 0, 1;
}
};

BOOST_FIXTURE_TEST_CASE(camera_params, PnpModelFixture){

  stereo_model->SetCameraParams(K);
  Eigen::MatrixXd K_out=stereo_model->GetCameraParams();

  for (int i=0;i<3;++i){
    for (int j=0;j<3;++j){
      //BOOST_CHECK_EQUAL(K(i,j),K_out(i,j));
    }
  }
}

BOOST_FIXTURE_TEST_CASE(update_model, PnpModelFixture){
  stereo_model->SetCameraParams(K);
  stereo_model->UpdateModel(stereo_features);
}