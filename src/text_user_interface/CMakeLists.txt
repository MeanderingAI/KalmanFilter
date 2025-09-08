add_library(text_user_interface SHARED
    Command.cpp
    Terminal.cpp
)
target_include_directories(text_user_interface PUBLIC ${CMAKE_SOURCE_DIR}/include/text_user_interface)