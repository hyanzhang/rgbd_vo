list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)
find_package(PCL 1.7 REQUIRED)
find_package(OpenCV REQUIRED)
add_definitions(${PCL_DEFINITIONS})

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories( "/usr/include/eigen3" )
include_directories(${G2O_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})

add_library(slambase slamBase.cpp)
target_link_libraries(slambase ${OpenCV_LIBS} ${PCL_LIBRARIES})

# add_executable(rgbdVO visualOdometry.cpp)
# target_link_libraries(rgbdVO slambase)

add_executable(fullslam slam.cpp)
target_link_libraries(fullslam slambase g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension ${CSPARSE_LIBRARY})