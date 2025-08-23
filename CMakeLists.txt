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

set(KALMAN_SOURCES
    main.cpp
    src/kalman_filter.cpp
)

add_executable(kalman_example ${KALMAN_SOURCES})

target_link_libraries(kalman_example PRIVATE Eigen3::Eigen)

target_include_directories(kalman_example PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)