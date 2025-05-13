# Draco3 messages
PROTOBUF_GENERATE_CPP(draco-rpc-msg-src draco-rpc-msg-header draco.proto)
add_library(rpc-draco-msg ${draco-rpc-msg-src} ${draco-rpc-msg-header})
target_include_directories(rpc-draco-msg PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

PROTOBUF_GENERATE_PYTHON(draco-rpc-msg-py draco.proto)
add_custom_target(draco-rpc-msg-py-target ALL DEPENDS ${draco-rpc-msg-py})

# G1 messages
PROTOBUF_GENERATE_CPP(g1-rpc-msg-src g1-rpc-msg-header g1.proto)
add_library(rpc-g1-msg ${g1-rpc-msg-src} ${g1-rpc-msg-header})
target_include_directories(rpc-g1-msg PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

PROTOBUF_GENERATE_PYTHON(g1-rpc-msg-py g1.proto)
add_custom_target(g1-rpc-msg-py-target ALL DEPENDS ${g1-rpc-msg-py})


install(TARGETS rpc-draco-msg DESTINATION "${INSTALL_LIB_DIR}")
install(FILES ${PROJECT_SOURCE_DIR}/build/messages/draco.pb.h DESTINATION
    "${INSTALL_INCLUDE_DIR}/messages")

install(TARGETS rpc-g1-msg DESTINATION "${INSTALL_LIB_DIR}")
install(FILES ${PROJECT_SOURCE_DIR}/build/messages/g1.pb.h DESTINATION
        "${INSTALL_INCLUDE_DIR}/messages")