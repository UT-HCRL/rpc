#include "configuration.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"

#include "controller/g1_controller/g1_kf_state_estimator.hpp"
#include "controller/g1_controller/g1_state_estimator.hpp"

#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_control_architecture_wbic.hpp"
#include "controller/g1_controller/g1_interface.hpp"
#include "controller/g1_controller/g1_interrupt_handler.hpp"
#include "controller/g1_controller/g1_state_provider.hpp"
#include "controller/g1_controller/g1_task_gain_handler.hpp"

#include "controller/g1_controller/g1_definition.hpp"

#if B_USE_ZMQ
#include "controller/g1_controller/g1_data_manager.hpp"
#endif

G1Interface::G1Interface() : Interface() {
  std::string border = "=";
  for (unsigned int i = 0; i < 79; ++i)
    border += "=";
  util::ColorPrint(color::kBoldRed, border);
  util::PrettyConstructor(0, "G1Interface");

  sp_ = G1StateProvider::GetStateProvider();

  // initialize robot model
  robot_ = new PinocchioRobotSystem(
      THIS_COM "robot_model/g1/g1_29dof_lock_waist.urdf",
      THIS_COM "robot_model/g1", false, false);

  // set locomotion control point
  robot_->SetFeetControlPoint("l_foot_contact", "r_foot_contact");

  // set control parameters
  this->_SetParameters();

  // initialize state estimator
  if (state_estimator_type_ == "default")
    se_ = new G1StateEstimator(robot_, cfg_);
  else if (state_estimator_type_ == "kf")
    se_ = new G1KFStateEstimator(robot_, cfg_);
  else {
    std::cout
        << "[G1Interface] Please check the state estimator type in pnc.yaml"
        << '\n';
    assert(false);
  }

  // initialize controller
  if (wbc_type_ == "ihwbc") {
    std::cout << "[G1Interface] Initializing ihwbc controller..." << std::endl;
    ctrl_arch_ = new G1ControlArchitecture(robot_, cfg_);
    interrupt_handler_ = new G1InterruptHandler(
        static_cast<G1ControlArchitecture *>(ctrl_arch_));
    task_gain_handler_ = new G1TaskGainHandler(
        static_cast<G1ControlArchitecture *>(ctrl_arch_));
  } else if (wbc_type_ == "wbic") {
    ctrl_arch_ = new G1ControlArchitecture_WBIC(robot_, cfg_);
    interrupt_handler_ = new G1InterruptHandler(
        static_cast<G1ControlArchitecture_WBIC *>(ctrl_arch_));
    task_gain_handler_ = new G1TaskGainHandler(
        static_cast<G1ControlArchitecture_WBIC *>(ctrl_arch_));
  } else {
    assert(false);
  }
}

G1Interface::~G1Interface() {
  delete robot_;
  delete se_;
  delete ctrl_arch_;
  delete interrupt_handler_;
  delete task_gain_handler_;
}

void G1Interface::GetCommand(void *sensor_data, void *command_data) {
  sp_->count_ = count_;
  sp_->current_time_ = static_cast<double>(count_) * sp_->servo_dt_;
  sp_->state_ = ctrl_arch_->locostate();
  sp_->prev_state_ = ctrl_arch_->prev_locostate();

  G1SensorData *g1_sensor_data =
      static_cast<G1SensorData *>(sensor_data);
  G1Command *g1_command = static_cast<G1Command *>(command_data);

  // estimate states
  if (b_cheater_mode_)
    se_->UpdateGroundTruthSensorData(g1_sensor_data);
  else {
    sp_->state_ == g1_states::kInitialize
        ? se_->Initialize(g1_sensor_data)
        : se_->Update(g1_sensor_data);
  }

  // process interrupt & task gains
  if (interrupt_handler_->IsSignalReceived())
    interrupt_handler_->Process();
  if (task_gain_handler_->IsSignalReceived())
    task_gain_handler_->Process();

  // get control command
  ctrl_arch_->GetCommand(g1_command);
  // get gripper command
  if (sp_->b_recv_gripper_cmd_)
    g1_command->gripper_pos_cmd_ = sp_->gripper_pos_cmd_;

#if B_USE_ZMQ
  if (sp_->count_ % sp_->data_save_freq_ == 0) {
    G1DataManager *dm = G1DataManager::GetDataManager();
    dm->data_->time_ = sp_->current_time_;
    dm->data_->phase_ = sp_->state_;
    dm->SendData();
  }
#endif

  // step control count
  count_++;
}

void G1Interface::_SafeCommand(G1SensorData *data,
                                  G1Command *command) {
  command->joint_pos_cmd_ = data->joint_pos_;
  command->joint_vel_cmd_.setZero();
  command->joint_trq_cmd_.setZero();
}

void G1Interface::_SetParameters() {
  try {
    // select test environment
    YAML::Node interface_cfg =
        YAML::LoadFile(THIS_COM "config/g1/INTERFACE.yaml");
    std::string test_env_name =
        util::ReadParameter<std::string>(interface_cfg, "test_env_name");
    std::cout << "============================================" << '\n';
    std::cout << "TEST ENV: " << test_env_name << '\n';
    std::cout << "============================================" << '\n';

    // select whole body controller type
    wbc_type_ = util::ReadParameter<std::string>(interface_cfg,
                                                 "whole_body_controller");
    std::cout << "============================================" << '\n';
    std::cout << "WHOLE BODY CONTROLLER: " << wbc_type_ << '\n';
    std::cout << "============================================" << '\n';

    if (test_env_name == "mujoco") {
      if (wbc_type_ == "ihwbc")
        cfg_ =
            YAML::LoadFile(THIS_COM "config/g1/sim/mujoco/ihwbc/pnc.yaml");
      else if (wbc_type_ == "wbic")
        cfg_ = YAML::LoadFile(THIS_COM "config/g1/sim/mujoco/wbic/pnc.yaml");
    } else if (test_env_name == "pybullet") {
      if (wbc_type_ == "ihwbc")
        cfg_ =
            YAML::LoadFile(THIS_COM "config/g1/sim/pybullet/ihwbc/pnc.yaml");
      if (wbc_type_ == "wbic")
        cfg_ =
            YAML::LoadFile(THIS_COM "config/g1/sim/pybullet/wbic/pnc.yaml");
    } else if (test_env_name == "hw") {
      if (wbc_type_ == "ihwbc")
        cfg_ = YAML::LoadFile(THIS_COM "config/g1/hw/ihwbc/pnc.yaml");
      if (wbc_type_ == "wbic")
        cfg_ = YAML::LoadFile(THIS_COM "config/g1/hw/wbic/pnc.yaml");
    } else {
      assert(false);
    }

    // WBC controller frequency
    sp_->servo_dt_ = util::ReadParameter<double>(cfg_, "servo_dt");
    sp_->data_save_freq_ = util::ReadParameter<int>(cfg_, "data_save_freq");

    // select state estimator
    state_estimator_type_ = util::ReadParameter<std::string>(
        cfg_["state_estimator"], "state_estimator_type");
    b_cheater_mode_ =
        util::ReadParameter<bool>(cfg_["state_estimator"], "b_cheater_mode");

    // select stance foot side
    int stance_foot = util::ReadParameter<int>(cfg_, "stance_foot");
    if (stance_foot == 0) {
      sp_->stance_foot_ = g1_link::l_foot_contact;
      sp_->prev_stance_foot_ = g1_link::l_foot_contact;
    } else if (stance_foot == 1) {
      sp_->stance_foot_ = g1_link::r_foot_contact;
      sp_->prev_stance_foot_ = g1_link::r_foot_contact;
    } else {
      assert(false);
    }

#if B_USE_ZMQ
    // ZMQ socket communication
    if (!G1DataManager::GetDataManager()->IsInitialized()) {
      std::string socket_address =
          util::ReadParameter<std::string>(cfg_, "ip_address");
      G1DataManager::GetDataManager()->InitializeSocket(socket_address);
    }
#endif

  } catch (const std::runtime_error &ex) {
    std::cerr << "Error Reading Parameter [" << ex.what() << "] at file: ["
              << __FILE__ << "]" << std::endl;
  }
}
