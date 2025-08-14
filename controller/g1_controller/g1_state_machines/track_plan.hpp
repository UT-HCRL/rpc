#pragma once

#include <memory>
#include <mutex>
#include "controller/state_machine.hpp"
#include "util/pkl_reader.hpp"
#include <thread>
#include <atomic>

#include "humanoid_multicontact_tracker.hpp"

class G1ControlArchitecture;
class G1StateProvider;
class TrackPlan : public StateMachine {
public:
    TrackPlan(const StateId state_id,
              PinocchioRobotSystem *robot,
              std::vector<pkl_utils::CompositeBezierCurve> planned_bezier_curves,
              std::unordered_map<std::string, std::vector<Vector3d>> planned_frames_dyn,  
              std::vector<Vector3d> planned_com_des,
              std::vector<double> planned_time,
              G1ControlArchitecture *ctrl_arch);
  ~TrackPlan();

  void FirstVisit() override;
  void OneStep() override;
  void LastVisit() override;
  bool EndOfState() override;

  StateId GetNextState() override;

  void ComputeSync();

  void SetParameters(const YAML::Node &node) override;

  void DoTrackPlan() { b_tracking_plan_ = true; }

private:

  float desired_frequency_ = 100.0; // Hz | Just used for fixed rate stepping

  std::unique_ptr<HumanoidMulticontactTracker> g1_mpc_;

  std::thread compute_thread_;
  std::atomic<bool> run_threads_{true};

  bool b_wait_complete_{true}; // [true, false] <-> [wait for MPC to complete, use fixed rate stepping]

  std::mutex data_mutex_;
  Eigen::Matrix<double, 27, 1> mpc_q_;
  Eigen::Matrix<double, 27, 1> mpc_q_dot_;
  Eigen::Matrix<double, 27, 1> mpc_tau_;
  Eigen::MatrixXd K_;
  bool has_new_data_;

  G1ControlArchitecture *ctrl_arch_;
  G1StateProvider *sp_;

  bool b_use_base_height_;
  bool b_tracking_plan_;

  std::unique_ptr<pkl_utils::BezierCurvesManager> bezier_curves_mgr_;
  std::vector<Eigen::Vector3d> com_des_;

  std::unordered_map<std::string, std::vector<Eigen::Vector3d>> planned_frames_dyn_;
  std::vector<std::string> target_names_;
  bool b_kin_plan_;
  int planner_counter_;
  std::vector<double> pkl_time_;

  void Compute();
};
