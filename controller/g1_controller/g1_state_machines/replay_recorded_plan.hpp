#pragma once
#include "util/pkl_reader.hpp"
#include "controller/state_machine.hpp"

using namespace pkl_utils;

class PinocchioRobotSystem;
class G1ControlArchitecture;
class G1StateProvider;

enum InterpolationMethod {
  kZOH = 0,
  kLinear = 1,
};

class ReplayRecordedPlan : public StateMachine {
public:
  ReplayRecordedPlan(const StateId state_id,
                     PinocchioRobotSystem *robot,
                     std::vector<Matrix <double, 34, 1>> joint_pos,
                     std::vector<Matrix <double, 33, 1>> joint_vel,
                     std::vector<Matrix <double, 27, 1>> joint_tau,
                     std::vector<double> time_vec,
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

  std::vector<Matrix<double, 34, 1>> pkl_joint_pos_;
  std::vector<Matrix<double, 33, 1>> pkl_joint_vel_;
  std::vector<Matrix<double, 27, 1>> pkl_joint_tau_;
  std::vector<double> pkl_time_;
  unsigned int planner_counter_;
  InterpolationMethod k_interp_method_;

};
