cmake_minimum_required(VERSION 3.10)

project(multicontact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(crocoddyl REQUIRED)

set(sources src/humanoid_multicontact_tracker.cpp)

add_library(${PROJECT_NAME} SHARED)
target_sources(${PROJECT_NAME} PRIVATE ${sources})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${crocoddyl_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} 
    PUBLIC crocoddyl::crocoddyl
    PUBLIC rpc-util
)

add_executable(${PROJECT_NAME}_test src/model_example.cpp)
target_link_libraries(${PROJECT_NAME}_test
    PRIVATE ${PROJECT_NAME}
)
target_include_directories(${PROJECT_NAME}_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(${PROJECT_NAME}_plot_time src/plot_solvetime.cpp)
target_link_libraries(${PROJECT_NAME}_plot_time
    PRIVATE ${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
)