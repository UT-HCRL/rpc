find_package(hpp-fcl REQUIRED)

pybind11_add_module(optimo_interface_py optimo_interface_pybind_pin.cpp)
target_link_libraries(optimo_interface_py PUBLIC rpc-optimo-controller hpp-fcl::hpp-fcl)