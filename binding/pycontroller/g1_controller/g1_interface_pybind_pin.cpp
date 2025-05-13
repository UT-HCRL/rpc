#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "controller/g1_controller/g1_interface.hpp"
#include "controller/interrupt_handler.hpp"

class PyInterface : public Interface {
public:
  using Interface::Interface;
  void GetCommand(void *sensor_data, void *command_data) override {
    PYBIND11_OVERRIDE_PURE(void, Interface, GetCommand, sensor_data,
                           command_data);
  }

protected:
  void _SetParameters() override {}
};

namespace py = pybind11;

PYBIND11_MODULE(g1_interface_py, m) {
  py::module::import("interrupt_py");

  py::class_<Interface, PyInterface>(m, "Interface")
      .def(py::init<>())
      .def("GetCommand", &Interface::GetCommand);

  py::class_<G1Interface, Interface>(m, "G1Interface")
      .def(py::init<>())
      .def_readwrite("interrupt_", &G1Interface::interrupt_handler_);

  py::class_<G1SensorData>(m, "G1SensorData")
      .def(py::init<>())
      .def_readwrite("imu_frame_quat_", &G1SensorData::imu_frame_quat_)
      .def_readwrite("imu_ang_vel_", &G1SensorData::imu_ang_vel_)
      .def_readwrite("joint_pos_", &G1SensorData::joint_pos_)
      .def_readwrite("joint_vel_", &G1SensorData::joint_vel_)
      .def_readwrite("b_lf_contact_", &G1SensorData::b_lf_contact_)
      .def_readwrite("b_rf_contact_", &G1SensorData::b_rf_contact_)
      .def_readwrite("lf_contact_normal_", &G1SensorData::lf_contact_normal_)
      .def_readwrite("rf_contact_normal_", &G1SensorData::rf_contact_normal_)
      .def_readwrite("imu_dvel_", &G1SensorData::imu_dvel_)
      .def_readwrite("imu_lin_acc_", &G1SensorData::imu_lin_acc_)

      // Debug
      .def_readwrite("base_joint_pos_", &G1SensorData::base_joint_pos_)
      .def_readwrite("base_joint_quat_", &G1SensorData::base_joint_quat_)
      .def_readwrite("base_joint_lin_vel_",
                     &G1SensorData::base_joint_lin_vel_)
      .def_readwrite("base_joint_ang_vel_",
                     &G1SensorData::base_joint_ang_vel_);

  py::class_<G1Command>(m, "G1Command")
      .def(py::init<>())
      .def_readwrite("joint_pos_cmd_", &G1Command::joint_pos_cmd_)
      .def_readwrite("joint_vel_cmd_", &G1Command::joint_vel_cmd_)
      .def_readwrite("joint_trq_cmd_", &G1Command::joint_trq_cmd_);
}
