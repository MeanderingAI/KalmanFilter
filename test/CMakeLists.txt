enable_testing()

set(TEST_SOURCES
    test_extended_kalman_filter.cpp
    test_kalman_filter.cpp
    test_particle_filter.cpp
    test_unscented_kalman_filter.cpp
)

add_executable(kalman_filter_tests ${TEST_SOURCES})
target_link_libraries(kalman_filter_tests PRIVATE kalman_filters Eigen3::Eigen gtest_main)
target_include_directories(kalman_filter_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME KalmanFilterTests COMMAND kalman_filter_tests)