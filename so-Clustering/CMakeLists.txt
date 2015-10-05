
set(CMAKE_CXX_FLAGS "-fPIC")

# In order to use C++ 11
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
# In order to use C++ 14
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y")

cmake_minimum_required(VERSION 2.8)

# For Release compilation
set(CMAKE_BUILD_TYPE Release)

# OpenMP for paralelizing code
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


# About this project
#
# SO-Clustering - Single Objective Optimisation for solving the clustering problem
project(SO-Clustering)

# ParadisEO
include_directories(${SO_SOURCE_DIR}/../ext/include/ParadisEO-2.0/eo/src/)
include_directories(${SO_SOURCE_DIR}/../ext/include/ParadisEO-2.0/mo/src/)
include_directories(${SO_SOURCE_DIR}/../ext/include/ParadisEO-2.0/moeo/src/)

# libpca
include_directories(${SO_SOURCE_DIR}/../ext/include/libpca-1.3.3/src/lib/)

include_directories(exes)
include_directories(lib)
include_directories(lib/chromosome) 

#
# Add subfolders
#
add_subdirectory(lib)
add_subdirectory(exes)
