cmake_minimum_required(VERSION 3.14)
project(pkl_reader VERSION 1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

FILE(GLOB_RECURSE sources RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
FILE(GLOB_RECURSE headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.hpp")

list(FILTER sources EXCLUDE REGEX "^mujoco_utils/")

add_library(rpc-util ${sources} ${headers})
target_link_libraries(rpc-util PUBLIC rpc-yaml)

install(TARGETS rpc-util DESTINATION "${INSTALL_LIB_DIR}")
install(FILES ${headers} DESTINATION "${INSTALL_INCLUDE_DIR}/util")

add_subdirectory(mujoco_utils)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g") ## unset for debug on segfault or memleaks

add_library(pkl_reader pkl_reader.cpp)
target_include_directories(pkl_reader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(pkl_reader PRIVATE pybind11::embed Eigen3::Eigen)
add_library(pkl_reader::pkl_reader ALIAS pkl_reader)

if(BUILD_WITH_GTESTS)
    add_executable(pkl_example pkl_reader_example.cpp)
    target_link_libraries(pkl_example PRIVATE pybind11::embed pkl_reader::pkl_reader)
endif()