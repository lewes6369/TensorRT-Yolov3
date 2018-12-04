cmake_minimum_required(VERSION 2.8)
project(trtNet)

set(CMAKE_BUILD_TYPE Release)

#include
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#src
set(PLUGIN_SOURCES
  src/EntroyCalibrator.cpp
  src/UpsampleLayer.cpp
  src/UpsampleLayer.cu
  src/TrtNet.cpp
)

#
# CUDA Configuration
#
find_package(CUDA REQUIRED)

set(CUDA_VERBOSE_BUILD ON)

# list(APPEND GPU_ARCHS
#     35
#     53
#     61
#     70
#     75
#     )

# # Generate SASS for each architecture
# foreach(arch ${GPU_ARCHS})
#     set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
# endforeach()
# # Generate PTX for the last architecture
# list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
# set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
#     -cudart static \
#     -lineinfo \
#     -g \
#     --expt-extended-lambda \
#     ${GENCODES} \
# ")

# Specify the cuda host compiler to use the same compiler as cmake.
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# # CUDNN
# set(CUDNN_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA cuDNN")
# find_path(CUDNN_INCLUDE_DIR cudnn.h
#     HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
#     PATH_SUFFIXES cuda/include include)
# find_library(CUDNN_LIBRARY cudnn
#     HINTS ${CUDNN_ROOT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
#     PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)
# find_package_handle_standard_args(
#     CUDNN DEFAULT_MSG CUDNN_INCLUDE_DIR CUDNN_LIBRARY)

# if(NOT CUDNN_FOUND)
#   message(WARNING
#       "Cudnn cannot be found. TensorRT depends explicitly "
#       "on cudnn so you should consider installing it.")
#   return()
# endif()

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
  find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS  ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
  TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)
if(NOT TENSORRT_FOUND)
  message(ERROR
    "Cannot find TensorRT library.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")	# -std=gnu++11
set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")

# if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
#   list(APPEND CUDA_NVCC_FLAGS -std=c++11)
# endif()
list(APPEND CUDA_NVCC_FLAGS "-D_FORCE_INLINES -Xcompiler -fPIC")
CUDA_INCLUDE_DIRECTORIES(${CUDNN_INCLUDE_DIR} ${TENSORRT_INCLUDE_DIR})
CUDA_ADD_LIBRARY(TrtNet STATIC ${PLUGIN_SOURCES})

target_include_directories(TrtNet PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR})
target_link_libraries(TrtNet ${TENSORRT_LIBRARY})