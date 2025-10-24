cmake_minimum_required(VERSION 3.24)
project(cuda_filters LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(cuda_gray src/main_gray.cu)
set_target_properties(cuda_gray PROPERTIES CUDA_ARCHITECTURES "75;86;89")
