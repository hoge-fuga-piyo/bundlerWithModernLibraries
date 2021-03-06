###############################
# Compiler setting
###############################

cmake_minimum_required(VERSION 2.8)

###############################
# External libraries
###############################
# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres Solver
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# yaml-cpp
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIR})

# Exiv2
find_path(Exiv2_INCLUDE_DIR NAMES exiv2.hpp)
find_library(Exiv2_LIBRARY NAMES exiv2.lib)
include_directories(${Exiv2_INCLUDE_DIR})

###############################
# For unit test
###############################
# Google Test
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

###############################
# Sources
###############################
set(PROJ_SHARED_DIR ${CMAKE_SOURCE_DIR}/src)

# core
set(PROJ_SHARED_CORE_DIR ${PROJ_SHARED_DIR}/core)
file(GLOB PROJ_SHARED_CORE_SOURCE ${PROJ_SHARED_CORE_DIR}/*.cpp)
file(GLOB PROJ_SHARED_CORE_INCLUDE ${PROJ_SHARED_CORE_DIR}/*.h ${PROJ_SHARED_CORE_DIR}/*.hpp)

# utility
set(PROJ_SHARED_UTIL_DIR ${PROJ_SHARED_DIR}/utility/src)
file(GLOB PROJ_SHARED_UTIL_SOURCE ${PROJ_SHARED_UTIL_DIR}/*.cpp)
file(GLOB PROJ_SHARED_UTIL_INCLUDE ${PROJ_SHARED_UTIL_DIR}/*.h ${PROJ_SHARED_UTIL_DIR}/*.hpp)

# costFunc
set(PROJ_SHARED_COSTFUNC_DIR ${PROJ_SHARED_DIR}/costFunc)
file(GLOB PROJ_SHARED_COSTFUNC_SOURCE ${PROJ_SHARED_COSTFUNC_DIR}/*.cpp)
file(GLOB PROJ_SHARED_COSTFUNC_INCLUDE ${PROJ_SHARED_COSTFUNC_DIR}/*.h ${PROJ_SHARED_COSTFUNC_DIR}/*.hpp)

###############################
# Project
###############################
set(PROJ_SUB_DIR ${CMAKE_SOURCE_DIR}/proj)
add_subdirectory(${PROJ_SUB_DIR})

###############################
# Test
###############################
#set(PROJ_TEST_DIR ${CMAKE_SOURCE_DIR}/test)
#add_subdirectory(${PROJ_TEST_DIR})
