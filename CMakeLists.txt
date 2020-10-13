cmake_minimum_required(VERSION 3.5)

#project(src LANGUAGES CXX CUDA)
project(src)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")

# TODO: Remove this from top-level add some logic so that this is a search, allowing you to build on non-gpu devices
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language(CUDA)


add_subdirectory(open_coral)
add_subdirectory(coral_gpu)
add_subdirectory(examples)




