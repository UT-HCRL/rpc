#include "controller/g1_controller/g1_state_machines/replay_recorded_plan.hpp"
#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_definition.hpp"
#include "controller/g1_controller/g1_state_provider.hpp"
#include "controller/g1_controller/g1_tci_container.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "configuration.hpp"


ReplayRecordedPlan::ReplayRecordedPlan(const StateId state_id,
                                           PinocchioRobotSystem *robot,
                                           G1ControlArchitecture *ctrl_arch)
    : StateMachine(state_id, robot), ctrl_arch_(ctrl_arch),
      planner_counter_(0) {
  util::PrettyConstructor(2, "ReplayRecordedPlan");

  sp_ = G1StateProvider::GetStateProvider();

  // load recorded plan
  // std::string file_path = THIS_COM "experiment_data/g1_knee_knocker_sca_on.pkl";
  // reader_ = new PickleReader(file_path, PickleType::LIST);
  // reader_->parse();
  // pkl_joint_pos_ = reader_->getJointPosDes();
  // pkl_joint_vel_ = reader_->getJointVelDes();
  // pkl_joint_tau_ = reader_->getJointTauDes();
  // pkl_time_ = reader_->getTimeVec();
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

  // send commands (ZOH version)
  if (state_machine_time_ >= pkl_time_[planner_counter_ + 1]) {
      // increase planner counter
      planner_counter_++;
  }
  Matrix<double, 44, 1> des_joint_pos = pkl_joint_pos_[planner_counter_];
  Matrix<double, 43, 1> des_joint_vel = pkl_joint_vel_[planner_counter_];
  Matrix<double, 37, 1> des_joint_trq = pkl_joint_tau_[planner_counter_];

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
