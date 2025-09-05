cmake_minimum_required(VERSION 3.0)
project(KalmanFilterExample)

# Use FetchContent to manage Eigen
include(FetchContent)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0 # Use a specific version tag for reproducibility
  # Use CMAKE_BINARY_DIR to ensure a safe build directory
  SOURCE_DIR ${CMAKE_BINARY_DIR}/eigen-src
  BINARY_DIR ${CMAKE_BINARY_DIR}/eigen-build
)
FetchContent_MakeAvailable(Eigen)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

set(KALMAN_SOURCES
    src/kalman_filter.cpp
    src/extended_kalman_filter.cpp
    src/unscented_kalman_filter.cpp
)

set(EXECUTABLES
  main.cpp
  )

add_library(kalman_filters SHARED ${KALMAN_SOURCES})
target_link_libraries(kalman_filters PRIVATE Eigen3::Eigen)
target_include_directories(kalman_filters PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kalman_example ${EXECUTABLES})
target_link_libraries(kalman_example PRIVATE kalman_filters Eigen3::Eigen)
target_include_directories(kalman_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(kalman_filter_tests test/test_kalman_filters.cpp)
target_link_libraries(kalman_filter_tests PRIVATE kalman_filters Eigen3::Eigen gtest_main)
target_include_directories(kalman_filter_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_test(NAME KalmanFilterTests COMMAND kalman_filter_tests)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)