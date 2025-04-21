#pragma once

#include "controller/state_machine.hpp"
#include "util/pkl_reader.hpp"

class G1ControlArchitecture;
class G1StateProvider;
class TrackPlan : public StateMachine {
public:
  TrackPlan(const StateId state_id, PinocchioRobotSystem *robot,
                       G1ControlArchitecture *ctrl_arch);
  ~TrackPlan() = default;

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  StateId GetNextState() override;

  void SetParameters(const YAML::Node &node) override;

  void DoTrackPlan() { b_tracking_plan_ = true; }

private:
  G1ControlArchitecture *ctrl_arch_;
  G1StateProvider *sp_;

  bool b_use_base_height_;
  bool b_tracking_plan_;

  double rf_z_max_interp_duration_;
  Eigen::Matrix<double, 6, 1> init_reaction_force_;
  Eigen::Matrix<double, 6, 1> des_reaction_force_;

  std::unique_ptr<pkl_utils::PickleReader> pkl_reader_;

  std::vector<pkl_utils::CompositeBezierCurve> bezier_curves_;
};
