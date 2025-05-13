file(GLOB_RECURSE sources "*.cpp" "*.c")
file(GLOB controller_headers "*.hpp")
file(GLOB state_machines_headers "g1_state_machines/*.hpp")
file(GLOB state_machines_headers_wbic "g1_state_machines_wbic/*.hpp")
file(GLOB g1_task_headers "g1_task/*.hpp" "g1_task/*.h")

list(REMOVE_ITEM sources "${CMAKE_CURRENT_SOURCE_DIR}/g1_data_manager.cpp")
list(REMOVE_ITEM controller_headers "${CMAKE_CURRENT_SOURCE_DIR}/g1_data_manager.hpp")

if(BUILD_WITH_ZMQ_PROTOBUF)
    list(APPEND sources "${CMAKE_CURRENT_SOURCE_DIR}/g1_data_manager.cpp")
    list(APPEND controller_headers "${CMAKE_CURRENT_SOURCE_DIR}/g1_data_manager.hpp")
endif()

add_library(rpc-g1-controller SHARED ${sources} ${controller_headers}
    ${state_machines_headers} ${state_machines_headers_wbic} ${g1_task_headers} ${g1_crbi_headers})


target_link_libraries(rpc-g1-controller PUBLIC rpc-pin-robot-system
                                           PUBLIC rpc-wbc
                                           PUBLIC rpc-util
                                           PUBLIC rpc-dcm-planner
                                           PUBLIC rpc-filter
                                           PUBLIC rpc-state-estimator
                                           PUBLIC rpc-model
                                           )

if(BUILD_WITH_ZMQ_PROTOBUF)
    target_link_libraries(rpc-g1-controller PUBLIC ${Protobuf_LIBRARIES}
                                               PUBLIC ${ZMQ_LIBRARIES}
                                               PUBLIC rpc-g1-msg)
endif()

if(BUILD_WITH_HPIPM)
    target_link_libraries(rpc-g1-controller PUBLIC convex-mpc)
endif()

if(BUILD_WITH_MATLOGGER)
    target_link_libraries(rpc-g1-controller PUBLIC matlogger2::matlogger2)
endif()

if(BUILD_WITH_FOXGLOVE)
    target_link_libraries(rpc-g1-controller PUBLIC foxglove-parameter-subscriber)
endif()


install(TARGETS rpc-g1-controller DESTINATION "${INSTALL_LIB_DIR}")
install(FILES ${controller_headers} DESTINATION
    "${INSTALL_INCLUDE_DIR}/controller/g1_controller")
install(FILES ${state_machines_headers} DESTINATION
    "${INSTALL_INCLUDE_DIR}/controller/g1_controller/g1_state_machines")
install(FILES ${state_machines_headers_wbic} DESTINATION
    "${INSTALL_INCLUDE_DIR}/controller/g1_controller/g1_state_machines_wbic")
install(FILES ${g1_task_headers} DESTINATION
    "${INSTALL_INCLUDE_DIR}/controller/g1_controller/g1_task")
