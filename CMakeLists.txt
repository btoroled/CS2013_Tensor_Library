cmake_minimum_required(VERSION 4.1)
project(CS2013_Tensor_Library)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(CS2013_Tensor_Library
        main.cpp
        src/Tensor.cpp
        include/Tensor.h
)
