#pragma once

#include "controller/state_machine.hpp"

class PinocchioRobotSystem;
class G1ControlArchitecture_WBIC;
class G1StateProvider;

class SingleSupportSwing_WBIC : public StateMachine {
public:
  SingleSupportSwing_WBIC(StateId state_id, PinocchioRobotSystem *robot,
                          G1ControlArchitecture_WBIC *ctrl_arch);
  ~SingleSupportSwing_WBIC() = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;
  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture_WBIC *ctrl_arch_;
  G1StateProvider *sp_;

  double swing_height_;
};
