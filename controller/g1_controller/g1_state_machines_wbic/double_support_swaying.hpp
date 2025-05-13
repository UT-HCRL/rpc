#pragma once
#include "controller/state_machine.hpp"

class PinocchioRobotSystem;
class G1ControlArchitecture_WBIC;
class G1StateProvider;

class DoubleSupportSwaying_WBIC : public StateMachine {
public:
  DoubleSupportSwaying_WBIC(const StateId state_id, PinocchioRobotSystem *robot,
                            G1ControlArchitecture_WBIC *ctrl_arch);
  ~DoubleSupportSwaying_WBIC() = default;

  void FirstVisit() override;
  void OneStep() override;
  bool EndOfState() override;
  void LastVisit() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture_WBIC *ctrl_arch_;
  G1StateProvider *sp_;

  Eigen::Vector3d amp_;
  Eigen::Vector3d freq_;
};
