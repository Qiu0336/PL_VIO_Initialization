cmake_minimum_required(VERSION 3.5)

project(VISLAM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wpedantic -Wall -Wextra")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
else(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif(CMAKE_BUILD_TYPE STREQUAL "Release")
message(STATUS "Flags: " ${CMAKE_CXX_FLAGS})


find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(Ceres REQUIRED)
#find_package(OpenCV 3.4 QUIET)
#find_package(OpenCV REQUIRED PATHS /home/qjy/lib/opencv-4.5.2/build NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED PATHS /home/qjy/lib/opencv-3.4.5/build NO_DEFAULT_PATH)

#Set(OpenCV_DIR "/home/qjy/lib/opencv-3.4.5/build")


find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)


include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
#    ${GFLAGS_INCLUDE_DIR}
#    ${GLOG_INCLUDE_DIRS}
#    ${GTEST_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
)

file(GLOB_RECURSE CAMERA src/camera/*.h src/camera/*.cc)
file(GLOB_RECURSE FEATURE src/feature/*.h src/feature/*.cc)
file(GLOB_RECURSE INITIALIZATION src/initialization/*.h src/initialization/*.cc)
file(GLOB_RECURSE SIMULATION src/simulation/*.h src/simulation/*.cc)
file(GLOB_RECURSE UTILS src/util/*.h src/util/*.cc)
file(GLOB_RECURSE VISUALIZATION src/visualization/*.h src/visualization/*.cc)


add_executable(evaluate evaluate.cc)

add_executable(feature ${UTILS} ${FEATURE} ${CAMERA})
target_link_libraries(feature
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    )

add_executable(initialization ${UTILS} ${INITIALIZATION})
target_link_libraries(initialization
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Boost_LIBRARIES}
    ${CERES_LIBRARIES}
    #${Pangolin_LIBRARIES}
    )

add_executable(simulation ${UTILS} ${SIMULATION} ${VISUALIZATION})
target_link_libraries(simulation
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    )
