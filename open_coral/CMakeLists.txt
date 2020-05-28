project(open_coral)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})


find_package(OpenMP REQUIRED)

find_package(GLog REQUIRED)
include_directories(${GLog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
set(CORAL_UTIL_SRCS
        src/coral_model_line.cpp
        src/coral_feature_point.cpp
        include/coral.h
        include/model_initialiser.h
        )


add_library(open_coral ${CORAL_UTIL_SRCS})
target_link_libraries(open_coral ${OpenCV_LIBS} Eigen3::Eigen
        OpenMP::OpenMP_CXX gflags glog::glog)

target_include_directories(open_coral
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        )