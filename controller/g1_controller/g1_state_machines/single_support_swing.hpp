#pragma once

#include "controller/state_machine.hpp"

class PinocchioRobotSystem;
class G1ControlArchitecture;
class G1StateProvider;

class SingleSupportSwing : public StateMachine {
public:
  SingleSupportSwing(StateId state_id, PinocchioRobotSystem *robot,
                     G1ControlArchitecture *ctrl_arch);
  ~SingleSupportSwing() = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;
  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture *ctrl_arch_;
  G1StateProvider *sp_;

  double swing_height_;

  // set nominal desired position/orientation (e.g., for zero acceleration cmd)
  bool b_use_fixed_foot_pos_;
  Eigen::Isometry3d nominal_lfoot_iso_;
  Eigen::Isometry3d nominal_rfoot_iso_;
};
