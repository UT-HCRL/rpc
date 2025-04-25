#pragma once
#include "util/pkl_reader.hpp"
#include "controller/state_machine.hpp"

using namespace pkl_utils;

class PinocchioRobotSystem;
class G1ControlArchitecture;
class G1StateProvider;

class ReplayRecordedPlan : public StateMachine {
public:
  ReplayRecordedPlan(const StateId state_id, PinocchioRobotSystem *robot,
                       G1ControlArchitecture *ctrl_arch);
  ~ReplayRecordedPlan();

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

private:
  G1ControlArchitecture *ctrl_arch_;

  G1StateProvider *sp_;

  PickleReader* reader_;
  std::vector<Matrix<double, 44, 1>> pkl_joint_pos_;
  std::vector<Matrix<double, 43, 1>> pkl_joint_vel_;
  std::vector<Matrix<double, 37, 1>> pkl_joint_tau_;
  std::vector<double> pkl_time_;
  unsigned int planner_counter_;

};
