file(GLOB simulate_sources "simulate/*.cc" "simulate/*.cpp")
file(GLOB simulate_headers "simulate/*.h")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(run_g1 main.cc ${simulate_sources} ${simulate_headers})
target_link_libraries(run_g1 ${mujoco_LIBRARIES}
                                ${glfw3_LIBRARIES}
                                glfw
                                mujoco-utils
                                Threads::Threads)
target_link_libraries(run_g1 rpc-g1-controller mujoco-utils)
