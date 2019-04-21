###############################
# SfM
###############################
set(PROJ_NAME SfM)
project(${PROJ_NAME})

# internal
file(GLOB PROJ_SOURCE *.cpp)
file(GLOB PROJ_INCLUDE *.h *.hpp)

# shared
include_directories(${PROJ_SHARED_CORE_DIR})
include_directories(${PROJ_SHARED_UTIL_DIR})
include_directories(${PROJ_SHARED_COSTFUNC_DIR})

#target
add_executable(${PROJ_NAME}
  ${PROJ_SOURCE} ${PROJ_INCLUDE}
  ${PROJ_SHARED_CORE_SOURCE} ${PROJ_SHARED_CORE_INCLUDE}
  ${PROJ_SHARED_UTIL_SOURCE} ${PROJ_SHARED_UTIL_INCLUDE}
  ${PROJ_SHARED_COSTFUNC_SOURCE} ${PROJ_SHARED_COSTFUNC_INCLUDE}
)

target_link_libraries(${PROJ_NAME}
  ${OpenCV_LIBS}
  ${CERES_LIBRARIES}
)
