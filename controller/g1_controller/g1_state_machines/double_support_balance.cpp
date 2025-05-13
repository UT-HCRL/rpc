#include "controller/g1_controller/g1_state_machines/double_support_balance.hpp"
#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_definition.hpp"
#include "controller/g1_controller/g1_state_provider.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "controller/whole_body_controller/managers/dcm_trajectory_manager.hpp"
#include "controller/whole_body_controller/managers/end_effector_trajectory_manager.hpp"

DoubleSupportBalance::DoubleSupportBalance(const StateId state_id,
                                           PinocchioRobotSystem *robot,
                                           G1ControlArchitecture *ctrl_arch)
    : StateMachine(state_id, robot), ctrl_arch_(ctrl_arch),
      b_com_swaying_(false), b_dcm_walking_(false), b_static_walking_(false) {
  util::PrettyConstructor(2, "DoubleSupportBalance");

  sp_ = G1StateProvider::GetStateProvider();
  nominal_lfoot_iso_.setIdentity();
  nominal_rfoot_iso_.setIdentity();
}

void DoubleSupportBalance::FirstVisit() {
  std::cout << "g1_states: kDoubleSupportBalance" << std::endl;
  state_machine_start_time_ = sp_->current_time_;

  // reset flags
  b_com_swaying_ = false;
  b_dcm_walking_ = false;
  b_static_walking_ = false;

  // set current foot position as nominal (desired) for rest of this state
  nominal_lfoot_iso_ = robot_->GetLinkIsometry(g1_link::l_foot_contact);
  nominal_rfoot_iso_ = robot_->GetLinkIsometry(g1_link::r_foot_contact);
  FootStep::MakeHorizontal(nominal_lfoot_iso_);
  FootStep::MakeHorizontal(nominal_rfoot_iso_);
}

void DoubleSupportBalance::OneStep() {
  state_machine_time_ = sp_->current_time_ - state_machine_start_time_;

  // update foot pose task update
  if (b_use_fixed_foot_pos_) {
    ctrl_arch_->lf_SE3_tm_->UseNominal(nominal_lfoot_iso_);
    ctrl_arch_->rf_SE3_tm_->UseNominal(nominal_rfoot_iso_);
  } else {
    ctrl_arch_->lf_SE3_tm_->UseCurrent();
    ctrl_arch_->rf_SE3_tm_->UseCurrent();
  }
}

bool DoubleSupportBalance::EndOfState() {
  if (b_com_swaying_ || b_static_walking_)
    return true;

  if (b_dcm_walking_ && ctrl_arch_->dcm_tm_->GetFootStepList().size() > 0 &&
      !ctrl_arch_->dcm_tm_->NoRemainingSteps())
    return true;

  return false;
}

void DoubleSupportBalance::LastVisit() {
  state_machine_time_ = 0.;

  if (sp_->b_use_base_height_)
    sp_->des_com_height_ = robot_->GetRobotComPos()[2];

  std::cout << "-----------------------------------------" << std::endl;
  std::cout << "des com height: " << sp_->des_com_height_ << std::endl;
  std::cout << "-----------------------------------------" << std::endl;

  Eigen::Isometry3d torso_iso =
      robot_->GetLinkIsometry(g1_link::torso_com_link);
  FootStep::MakeHorizontal(torso_iso);
  sp_->rot_world_local_ = torso_iso.linear();
}

StateId DoubleSupportBalance::GetNextState() {
  if (b_com_swaying_)
    return g1_states::kDoubleSupportSwaying;

  if (b_dcm_walking_) {
    if (ctrl_arch_->dcm_tm_->GetSwingLeg() == end_effector::LFoot) {
      b_dcm_walking_ = false;
      return g1_states::kLFContactTransitionStart;
    } else if (ctrl_arch_->dcm_tm_->GetSwingLeg() == end_effector::RFoot) {
      b_dcm_walking_ = false;
      return g1_states::kRFContactTransitionStart;
    }
  }

  // if (b_static_walking_) {
  // if (true) {
  // return g1_states::kMoveComToLFoot;
  //} else {
  // return g1_states::kMoveComToRFoot;
  //}
  //}
}

void DoubleSupportBalance::SetParameters(const YAML::Node &node) {
  try {
    b_use_fixed_foot_pos_ = util::ReadParameter<bool>(
        node["state_machine"], "b_use_const_desired_foot_pos");
  } catch (const std::runtime_error &e) {
    std::cerr << "Error reading parameter [" << e.what() << "] at file: ["
              << __FILE__ << "]" << std::endl;
  }
}
