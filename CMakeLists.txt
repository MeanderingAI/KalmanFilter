cmake_minimum_required(VERSION 3.0)
project(Models)

# Include the external dependencies from the sub-directory
include(cmake/ExternalDependencies.cmake)

# The main executable for the example application
add_executable(tui main.cpp)

# Add subdirectories for the shared library and tests
add_subdirectory(src)
add_subdirectory(test)

target_link_libraries(tui PRIVATE text_user_interface)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)