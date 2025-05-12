find_package(hpp-fcl REQUIRED)

pybind11_add_module(draco_interface_py draco_interface_pybind_pin.cpp)
target_link_libraries(draco_interface_py PUBLIC rpc-draco-controller hpp-fcl::hpp-fcl)
