cmake_minimum_required(VERSION 2.8)
project(demo)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors")

#add lib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/wrapper)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/wrapper/include)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#add opencv
FIND_PACKAGE(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES_DIRS})

#build runYolov3
add_executable(runYolov3 main.cpp YoloLayer.cpp box.cpp)
target_link_libraries(runYolov3 TrtNet ${OpenCV_LIBS})

install(TARGETS runYolov3 DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/install/)