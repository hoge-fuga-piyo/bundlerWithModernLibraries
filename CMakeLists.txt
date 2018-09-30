###############################
# Compiler setting
###############################

cmake_minimum_required(VERSION 2.8)

###############################
# External libraries
###############################
# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Ceres Solver
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

###############################
# Sources
###############################
set(PROJ_SHARED_DIR ${CMAKE_SOURCE_DIR}/src)

# core
set(PROJ_SHARED_CORE_DIR ${PROJ_SHARED_DIR}/core)
file(GLOB PROJ_SHARED_CORE_SOURCE ${PROJ_SHARED_CORE_DIR}/*.cpp)
file(GLOB PROJ_SHARED_CORE_INCLUDE ${PROJ_SHARED_CORE_DIR}/*.h ${PROJ_SHARED_CORE_DIR}/*.hpp)

# utility
set(PROJ_SHARED_UTIL_DIR ${PROJ_SHARED_DIR}/utility)
file(GLOB PROJ_SHARED_UTIL_SOURCE ${PROJ_SHARED_UTIL_DIR}/*.cpp)
file(GLOB PROJ_SHARED_UTIL_INCLUDE ${PROJ_SHARED_UTIL_DIR}/*.h ${PROJ_SHARED_UTIL_DIR}/*.hpp)

###############################
# Project
###############################
set(PROJ_SUB_DIR ${CMAKE_SOURCE_DIR}/proj)
add_subdirectory(${PROJ_SUB_DIR})
