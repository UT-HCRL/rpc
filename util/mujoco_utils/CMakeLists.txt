cmake_minimum_required(VERSION 3.14)
project(mujoco_utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(sources
    src/mujoco_utils.cpp
)
set(headers
    include/mujoco_utils.hpp
)

add_library(mujoco-utils ${sources} ${headers})

target_include_directories(mujoco-utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mujoco-utils PUBLIC ${MUJOCO_LIBRARY})

install(TARGETS mujoco-utils DESTINATION "${INSTALL_LIB_DIR}")
install(FILES ${headers} DESTINATION "${INSTALL_INCLUDE_DIR}/util")