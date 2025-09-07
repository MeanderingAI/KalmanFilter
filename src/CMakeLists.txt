add_library(filters SHARED
    kalman_filter.cpp
    extended_kalman_filter.cpp
    unscented_kalman_filter.cpp
    sequential_monte_carlo.cpp
)
target_link_libraries(filters PRIVATE Eigen3::Eigen)
target_include_directories(filters PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(hidden_markov_model SHARED 
    hidden_markov_model.cpp
)
target_link_libraries(hidden_markov_model PRIVATE Eigen3::Eigen)
target_include_directories(hidden_markov_model PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(bayesian_network SHARED
    bayesian_network.cpp
)
target_link_libraries(bayesian_network PRIVATE Eigen3::Eigen)
target_include_directories(bayesian_network PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(decision_tree SHARED
    decision_tree.cpp
    rule_set.cpp
    random_forest.cpp
    boost_tree.cpp
)
target_link_libraries(decision_tree PRIVATE Eigen3::Eigen)
target_include_directories(decision_tree PUBLIC ${CMAKE_SOURCE_DIR}/include)


add_library(generalized_linear_model SHARED
    generalized_linear_model.cpp
    linear_regression.cpp
)
target_link_libraries(generalized_linear_model PRIVATE Eigen3::Eigen)
target_include_directories(generalized_linear_model PUBLIC ${CMAKE_SOURCE_DIR}/include)
