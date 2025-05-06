#pragma once
#include "controller/state_machine.hpp"

class PinocchioRobotSystem;
class G1ControlArchitecture;
class G1StateProvider;

class DoubleSupportBalance : public StateMachine {
public:
  DoubleSupportBalance(const StateId state_id, PinocchioRobotSystem *robot,
                       G1ControlArchitecture *ctrl_arch);
  ~DoubleSupportBalance() = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

  // boolean setter
  void DoComSwaying() { b_com_swaying_ = true; }

  void DoDcmWalking() { b_dcm_walking_ = true; }

  void DoStaticWalking() { b_static_walking_ = true; }

  void DoReplayRecordedPlan() { b_replay_recorded_plan_ = true; }

  void DoTrackPlan() { b_track_plan_ = true; }

private:
  G1ControlArchitecture *ctrl_arch_;

  G1StateProvider *sp_;

  bool b_com_swaying_;

  bool b_dcm_walking_;

  bool b_static_walking_;

  bool b_replay_recorded_plan_;

  bool b_track_plan_;

  // set nominal desired position/orientation (e.g., for zero acceleration cmd)
  bool b_use_fixed_foot_pos_;
  Eigen::Isometry3d nominal_lfoot_iso_;
  Eigen::Isometry3d nominal_rfoot_iso_;
};
