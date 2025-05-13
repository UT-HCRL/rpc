#pragma once
#include "controller/state_machine.hpp"

class G1StateProvider;
class G1ControlArchitecture;
class MinJerkCurveVec;

class Initialize : public StateMachine {
public:
  Initialize(const StateId state_id, PinocchioRobotSystem *robot,
             G1ControlArchitecture *ctrl_arch);
  ~Initialize();

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture *ctrl_arch_;
  G1StateProvider *sp_;

  Eigen::VectorXd target_joint_pos_;
  Eigen::VectorXd init_joint_pos_;

  bool b_stay_here_;
  double wait_time_;

  MinJerkCurveVec *min_jerk_curves_;
};
