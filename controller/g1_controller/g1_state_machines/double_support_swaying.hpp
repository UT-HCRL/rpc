#pragma once
#include "controller/state_machine.hpp"

class PinocchioRobotSystem;
class G1ControlArchitecture;
class G1StateProvider;

class DoubleSupportSwaying : public StateMachine {
public:
  DoubleSupportSwaying(const StateId state_id, PinocchioRobotSystem *robot,
                       G1ControlArchitecture *ctrl_arch);
  ~DoubleSupportSwaying() = default;

  void FirstVisit() override;
  void OneStep() override;
  bool EndOfState() override;
  void LastVisit() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture *ctrl_arch_;
  G1StateProvider *sp_;

  Eigen::Vector3d amp_;
  Eigen::Vector3d freq_;

  // set nominal desired position/orientation (e.g., for zero acceleration cmd)
  bool b_use_fixed_foot_pos_;
  Eigen::Isometry3d nominal_lfoot_iso_;
  Eigen::Isometry3d nominal_rfoot_iso_;
};
