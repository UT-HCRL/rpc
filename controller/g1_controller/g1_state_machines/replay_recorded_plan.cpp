#include "controller/g1_controller/g1_state_machines/replay_recorded_plan.hpp"
#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_definition.hpp"
#include "controller/g1_controller/g1_state_provider.hpp"
#include "controller/g1_controller/g1_tci_container.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "configuration.hpp"
#include "interpolation.hpp"


ReplayRecordedPlan::ReplayRecordedPlan(const StateId state_id,
                                       PinocchioRobotSystem *robot,
                                       G1ControlArchitecture *ctrl_arch)
    : StateMachine(state_id, robot), ctrl_arch_(ctrl_arch),
      planner_counter_(0) {
  util::PrettyConstructor(2, "ReplayRecordedPlan");

  sp_ = G1StateProvider::GetStateProvider();

  // load recorded plan
  std::string file_path = THIS_COM "experiment_data/g1_29dof_knee_knocker_full_step_over.pkl";
  // reader_ = new PickleReader(file_path, PickleType::LIST);
  // // reader_ = std::make_unique<PickleReader>(file_path, PickleType::LIST);
  // if (!reader_->isReady()) {
  //   std::cerr << "Failed to open the file." << std::endl;
  //   return;
  // }
  // reader_->parse();
  // pkl_joint_pos_ = reader_->getJointPosDes();
  // pkl_joint_vel_ = reader_->getJointVelDes();
  // pkl_joint_tau_ = reader_->getJointTauDes();
  // pkl_time_ = reader_->getTimeVec();
  k_interp_method_ = kLinear;
}

ReplayRecordedPlan::~ReplayRecordedPlan() {
  delete sp_;
  delete ctrl_arch_;
  delete reader_;
}

void ReplayRecordedPlan::FirstVisit() {
  state_machine_start_time_ = sp_->current_time_;
}

void ReplayRecordedPlan::OneStep() {
  state_machine_time_ = sp_->current_time_ - state_machine_start_time_;

  Matrix<double, 44, 1> des_joint_pos;
  Matrix<double, 43, 1> des_joint_vel;
  Matrix<double, 37, 1> des_joint_trq;
  des_joint_trq.setZero();

  // get corresponding time index
  if (state_machine_time_ >= pkl_time_[planner_counter_ + 1]) {
    // increase planner counter
    planner_counter_++;
  }

  // send commands (ZOH version)
  double alpha = 0.;
  switch (k_interp_method_) {
    case kZOH:
      // apply desired joint position/velocity/torque at corresponding time
      des_joint_pos = pkl_joint_pos_[planner_counter_];
      des_joint_vel = pkl_joint_vel_[planner_counter_];
      // des_joint_trq = pkl_joint_tau_[planner_counter_];
      break;
    case kLinear:
      alpha = (state_machine_time_ - pkl_time_[planner_counter_]) /
        (pkl_time_[planner_counter_ + 1] - pkl_time_[planner_counter_]);
      des_joint_pos = Lerp<Matrix<double, 44, 1>, double>(
        pkl_joint_pos_[planner_counter_], pkl_joint_pos_[planner_counter_ + 1],
        alpha);
      des_joint_vel = Lerp<Matrix<double, 43, 1>, double>(
        pkl_joint_vel_[planner_counter_], pkl_joint_vel_[planner_counter_ + 1],
        alpha);
      des_joint_trq = Lerp<Matrix<double, 37, 1>, double>(
        pkl_joint_tau_[planner_counter_], pkl_joint_tau_[planner_counter_ + 1],
        alpha);
      break;
    default:
      std::cout << "[ReplayRecordedPlan] - Invalid interpolation method" << std::endl;
      break;
  }

  // update commands
  ctrl_arch_->tci_container_->robot_commands_->UpdateDesired(
    des_joint_pos.tail(37), des_joint_vel.tail(37), des_joint_trq);
}

bool ReplayRecordedPlan::EndOfState() {

  if (state_machine_time_ >= pkl_time_.back()) {
    std::cout << "End of replay recorded plan" << std::endl;
    return true;
  }

  return false;
}

void ReplayRecordedPlan::LastVisit() {
  state_machine_time_ = 0.;
}

StateId ReplayRecordedPlan::GetNextState() {
  return g1_states::kDoubleSupportBalance;
}

void ReplayRecordedPlan::SetParameters(const YAML::Node &node) {}
