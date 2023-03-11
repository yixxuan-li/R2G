cmake_minimum_required(VERSION 3.5.1)
project(relation)
#这一句一定要，编译要用release模式，不设的话，默认是debug，速度慢很多 
set(CMAKE_BUILD_TYPE "Release")
add_subdirectory(pybind11)
pybind11_add_module(relation relation.cpp)
 
 
 