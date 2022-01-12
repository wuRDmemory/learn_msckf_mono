cmake_minimum_required(VERSION 3.0.2)
project(learn_msckf)

option(BUILD_DEBUG ON)
option(BUILD_TEST  ON)
if (BUILD_DEBUG)
  set(CMAKE_BUILD_TYPE Debug)
  message("-- Build debug version.")
else()
  set(CMAKE_BUILD_TYPE Release)
  message("-- Build release version.")
endif()

include(${PROJECT_SOURCE_DIR}/cmake/functions.cmake)

## setup project
init_project(PROJECT_SOURCE_DIR)

set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  cv_bridge
  image_transport
  geometry_msgs
  nav_msgs
  sensor_msgs
  nodelet
  tf2
  tf2_ros
)

catkin_package()

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(glog REQUIRED)
find_package(gflags REQUIRED)
find_package(absl REQUIRED)
find_package(GTest REQUIRED)
find_package(Ceres REQUIRED)
find_package(SuiteSparse REQUIRED)
set(ABSL_LIBS
  absl::algorithm
  absl::base
  absl::flat_hash_map
  absl::memory
  absl::synchronization
)

include_directories(
  /usr/include/eigen3/
  ${SRC_DIR}
  ${catkin_INCLUDE_DIRS}
  ${BOOST_INCLUDE_DIRS}
  ${GLOG_INCLUDE_DIR}
  ${GFLAGS_INCLUDE_DIR}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${GTEST_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

set(ALL_LIBS
  ${catkin_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${ABSL_LIBS}
  ${OpenCV_LIBS}
  ${GTEST_BOTH_LIBRARIES}
  ${CERES_LIBRARIES}
  glog
  gflags
  pthread
)


file(GLOB_RECURSE ALL_SRCS ${SRC_DIR}/*.cc)
# file(GLOB_RECURSE ALL_NODE ${SRC_DIR}/*_node.cc)
# list(REMOVE_ITEM ALL_SRCS ${ALL_NODE})

message("-- All Source file list")
foreach(SRC_FILE ${ALL_SRCS})
  file(RELATIVE_PATH SRC_NAME ${PROJECT_SOURCE_DIR} ${SRC_FILE})
  message("-- |-- ${SRC_NAME}")
endforeach(SRC_FILE ${ALL_SRCS})

# node 
add_executable(msckf_node msckf_node.cc ${ALL_SRCS})
target_link_libraries(msckf_node ${ALL_LIBS})

# test
if (BUILD_TEST) 
  enable_testing()
  set(TEST_DIR ${PROJECT_SOURCE_DIR}/testing)

  include_directories(${TEST_DIR})

  file(GLOB_RECURSE ALL_TEST ${TEST_DIR}/*_test.cc)
  file(GLOB_RECURSE ALL_TEST_SRCS ${TEST_DIR}/*.cc)
  file(GLOB_RECURSE ALL_BACK_TEST ${TEST_DIR}/*_test1.cc)
  list(REMOVE_ITEM ALL_TEST_SRCS ${ALL_TEST} ${ALL_BACK_TEST})

  message("-- Build TESTING")
  message("-- All Test source file")
  foreach(SRC_FILE ${ALL_TEST_SRCS})
    file(RELATIVE_PATH SRC_NAME ${TEST_DIR} ${SRC_FILE})
    message("-- |-- ${SRC_NAME}")
  endforeach(SRC_FILE ${ALL_TEST_SRCS})

  message("-- All Test execute file")
  foreach(TEST_FILE ${ALL_TEST})
    file(RELATIVE_PATH TEST_NAME ${PROJECT_SOURCE_DIR} ${TEST_FILE})
    get_filename_component(EXE_NAME ${TEST_FILE} NAME_WE)
    message("-- |-- ${TEST_NAME}: ${EXE_NAME}")   

    add_executable(${EXE_NAME} ${TEST_FILE} ${ALL_SRCS} ${ALL_TEST_SRCS})
    target_link_libraries(${EXE_NAME} ${ALL_LIBS})

  endforeach(TEST_FILE ${ALL_TEST})
  
endif(BUILD_TEST)
