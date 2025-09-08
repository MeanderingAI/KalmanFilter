add_library(text_user_interface SHARED
    Command.cpp
    DistributionCommand.cpp
    Terminal.cpp
)

target_include_directories(text_user_interface PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(text_user_interface PUBLIC probability_distribution)