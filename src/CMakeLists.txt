add_library(kalman_filters SHARED
    kalman_filter.cpp
    extended_kalman_filter.cpp
    unscented_kalman_filter.cpp
    particle_filter.cpp
)
target_link_libraries(kalman_filters PRIVATE Eigen3::Eigen)
target_include_directories(kalman_filters PUBLIC ${CMAKE_SOURCE_DIR}/include)