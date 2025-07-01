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
                                       std::vector<Matrix <double, 34, 1>> joint_pos,
                                       std::vector<Matrix <double, 33, 1>> joint_vel,
                                       std::vector<Matrix <double, 27, 1>> joint_tau,
                                       std::vector<double> time_vec,
                                       G1ControlArchitecture *ctrl_arch)
    : StateMachine(state_id, robot), ctrl_arch_(ctrl_arch),
      planner_counter_(0) {
  util::PrettyConstructor(2, "ReplayRecordedPlan");

  sp_ = G1StateProvider::GetStateProvider();

  // load recorded plan
  pkl_joint_pos_ = joint_pos;
  pkl_joint_vel_ = joint_vel;
  pkl_joint_tau_ = joint_tau;
  pkl_time_ = time_vec;

  k_interp_method_ = kLinear;
}

ReplayRecordedPlan::~ReplayRecordedPlan() {
  delete sp_;
  delete ctrl_arch_;
}

void ReplayRecordedPlan::FirstVisit() {
  state_machine_start_time_ = sp_->current_time_;
}

void ReplayRecordedPlan::OneStep() {
  state_machine_time_ = sp_->current_time_ - state_machine_start_time_;

  Matrix<double, 34, 1> des_joint_pos;
  Matrix<double, 33, 1> des_joint_vel;
  Matrix<double, 27, 1> des_joint_trq;
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
      des_joint_pos = Lerp<Matrix<double, 34, 1>, double>(
        pkl_joint_pos_[planner_counter_], pkl_joint_pos_[planner_counter_ + 1],
        alpha);
      des_joint_vel = Lerp<Matrix<double, 33, 1>, double>(
        pkl_joint_vel_[planner_counter_], pkl_joint_vel_[planner_counter_ + 1],
        alpha);
      // des_joint_trq = Lerp<Matrix<double, 27, 1>, double>(
      //   pkl_joint_tau_[planner_counter_], pkl_joint_tau_[planner_counter_ + 1],
      //   alpha);
      break;
    default:
      std::cout << "[ReplayRecordedPlan] - Invalid interpolation method" << std::endl;
      break;
  }

  // update commands
  ctrl_arch_->tci_container_->robot_commands_->UpdateDesired(
    des_joint_pos.tail(27), des_joint_vel.tail(27), des_joint_trq);
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
