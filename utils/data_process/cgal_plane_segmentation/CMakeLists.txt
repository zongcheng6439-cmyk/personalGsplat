cmake_minimum_required(VERSION 3.16)
project(cgal_plane_segmentation LANGUAGES CXX)

find_package(CGAL REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(cgal_plane_segmentation cgal_plane_segmentation.cpp)
target_link_libraries(cgal_plane_segmentation PRIVATE CGAL::CGAL)
target_include_directories(cgal_plane_segmentation PRIVATE ${EIGEN3_INCLUDE_DIR})
target_compile_features(cgal_plane_segmentation PRIVATE cxx_std_17)
