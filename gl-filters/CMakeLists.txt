cmake_minimum_required(VERSION 3.24)
project(gl_filters LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- vcpkg dependencies ---
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)

# --- add stb manually (header-only) ---
# Assumes you put stb_image.h inside a local "include" folder in your project root.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- build target ---
add_executable(gl_window
    src/main_window.cpp
)

target_link_libraries(gl_window PRIVATE
    glfw
    glad::glad
)