project(open_coral)
enable_testing()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

#OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" ON)

find_package(OpenMP REQUIRED)

find_package(Ceres REQUIRED)

find_package(GLog REQUIRED)
include_directories(${GLog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
set(CORAL_UTIL_SRCS
        src/coral_feature_stereo_correspondence.cpp
        src/coral_feature_curvature.cpp
        src/coral_pnp_model.cpp
        src/coral_model_line.cpp
        src/coral_feature_point.cpp
        src/coral_photo_model.cpp
        src/coral_curvature_extractor.cpp
        src/coral_affine_model.cpp
        include/coral.h
        include/model_initialiser.h
        include/viewer/coral_viewer.h
        )


add_library(open_coral ${CORAL_UTIL_SRCS})
target_link_libraries(open_coral ${OpenCV_LIBS} Eigen3::Eigen
        OpenMP::OpenMP_CXX gflags glog::glog Ceres::ceres)

target_include_directories(open_coral
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        )

# Cmake logic for creating the boost tests
# creates the executable
add_executable(test_executable tests/coral_line_tests.cpp)
# indicates the include paths
target_include_directories(test_executable PRIVATE ${Boost_INCLUDE_DIRS})
# indicates the shared library variant
target_compile_definitions(test_executable PRIVATE "BOOST_TEST_DYN_LINK=1")
# indicates the link paths
target_link_libraries(test_executable open_coral ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
# declares a test with our executable
add_test(NAME test1 COMMAND test_executable)


# creates the executable
add_executable(pnp_model_test tests/coral_pnp_tests.cpp)
# indicates the include paths
target_include_directories(pnp_model_test PRIVATE ${Boost_INCLUDE_DIRS})
# indicates the shared library variant
target_compile_definitions(pnp_model_test PRIVATE "BOOST_TEST_DYN_LINK=1")
# indicates the link paths
target_link_libraries(pnp_model_test open_coral ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
# declares a test with our executable
add_test(NAME coral_pnp_model_test COMMAND pnp_model_test)

# creates the executable
add_executable(coral_test tests/coral_tests.cpp)
# indicates the include paths
target_include_directories(coral_test PRIVATE ${Boost_INCLUDE_DIRS})
# indicates the shared library variant
target_compile_definitions(coral_test PRIVATE "BOOST_TEST_DYN_LINK=1")
# indicates the link paths
target_link_libraries(coral_test open_coral ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
# declares a test with our executable
add_test(NAME coral_library_test COMMAND coral_test)

# creates the executable
add_executable(stereo_test tests/coral_stereo_tests.cpp)
# indicates the include paths
target_include_directories(stereo_test PRIVATE ${Boost_INCLUDE_DIRS})
# indicates the shared library variant
target_compile_definitions(stereo_test PRIVATE "BOOST_TEST_DYN_LINK=1")
# indicates the link paths
target_link_libraries(stereo_test open_coral ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
# declares a test with our executable
add_test(NAME coral_stereo_test COMMAND stereo_test)

# creates the executable
add_executable(photo_test tests/coral_photo_tests.cpp)
# indicates the include paths
target_include_directories(photo_test PRIVATE ${Boost_INCLUDE_DIRS})
# indicates the shared library variant
target_compile_definitions(photo_test PRIVATE "BOOST_TEST_DYN_LINK=1")
# indicates the link paths
target_link_libraries(photo_test open_coral ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})
# declares a test with our executable
add_test(NAME coral_photo_test COMMAND photo_test)
