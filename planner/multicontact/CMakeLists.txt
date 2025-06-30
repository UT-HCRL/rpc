cmake_minimum_required(VERSION 3.10)

project(multicontact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(crocoddyl REQUIRED)
find_package(OpenMP REQUIRED)

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
    PUBLIC OpenMP::OpenMP_CXX
    PUBLIC rpc-util
    PUBLIC contact_switching
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