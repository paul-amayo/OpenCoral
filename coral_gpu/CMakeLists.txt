project(coral_cuda)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
find_package(OpenMP REQUIRED)

find_package(Open3D REQUIRED)
include_directories(${Open3D_INCLUDE_DIRS})

#find_package(Boost REQUIRED 1.72.0)
find_package(Boost 1.58 REQUIRED COMPONENTS
        date_time filesystem system iostreams)
include_directories(${Boost_INCLUDE_DIRS})

find_package(GLog REQUIRED)
include_directories(${GLog_INCLUDE_DIRS})

find_package(gflags REQUIRED)
include_directories(${gflags_INCLUDE_DIRS})

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_path(CUDA_SDK_INCLUDE_DIR
        helper_math.h
        PATHS ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES "common/inc" "samples/common/inc"
        DOC "Location of helper_math.h"
        NO_DEFAULT_PATH)
if (NOT CUDA_SDK_INCLUDE_DIR)
    message(FATAL_ERROR "CUDA SDK DIRECTORY NOT FOUND")
endif (NOT CUDA_SDK_INCLUDE_DIR)
include_directories(${CUDA_SDK_INCLUDE_DIR})
set(CUDA_SEPARABLE_COMPILATION ON)


set(CMAKE_CXX_STANDARD 14)
set(CORAL_UTIL_SRCS

        include/coral_cuda_wrapper.h
        include/cuda_matrix.h
        include/cuda_memory.h
        include/cuda_optimiser.h

        src/cuda_optimiser.cu
        src/cuda_memory.cpp


        )


cuda_add_library(coral_cuda ${CORAL_UTIL_SRCS})

target_compile_features(coral_cuda PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the particle
# library to be built with -dc as the member functions could be called by
# other libraries and executables
set_target_properties(coral_cuda
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON
        )

target_link_libraries(coral_cuda open_coral ${CUDA_LIBRARIES} ${CUDA_nppc_LIBRARY} ${CUDA_nppial_LIBRARY}
        ${CUDA_nppicc_LIBRARY} ${CUDA_nppicom_LIBRARY} ${CUDA_nppidei_LIBRARY} ${CUDA_nppif_LIBRARY}
        ${CUDA_nppig_LIBRARY} ${CUDA_nppim_LIBRARY} ${CUDA_nppist_LIBRARY}
        ${CUDA_nppisu_LIBRARY} ${CUDA_nppitc_LIBRARY} ${CUDA_npps_LIBRARY}
        ${CUDA_nppc_LIBRARY} ${Boost_SYSTEM_LIBRARY}
        gflags glog::glog)

target_include_directories(coral_cuda
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
        )

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach (dir ${dirs})
    message(STATUS "include dir = '${dir}'")
endforeach ()