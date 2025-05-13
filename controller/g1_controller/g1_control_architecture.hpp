#pragma once
#include "controller/control_architecture.hpp"
#if B_USE_FOXGLOVE
#include "UI/foxglove/client/parameter_subscriber.hpp"
#endif
#include "util/util.hpp"
#include <any>

namespace g1_states {
constexpr int kInitialize = 1;
constexpr int kDoubleSupportStandUp = 2;
constexpr int kDoubleSupportBalance = 3;
constexpr int kDoubleSupportSwaying = 4;
constexpr int kLFContactTransitionStart = 5;
constexpr int kLFContactTransitionEnd = 6;
constexpr int kLFSingleSupportSwing = 7;
constexpr int kRFContactTransitionStart = 8;
constexpr int kRFContactTransitionEnd = 9;
constexpr int kRFSingleSupportSwing = 10;
// constexpr int kMPCLocomotion = 11;
constexpr int kTeleopManipulation = 20;
} // namespace g1_states

class G1Controller;
class G1TCIContainer;
class DCMPlanner;
class FloatingBaseTrajectoryManager;
class UpperBodyTrajetoryManager;
class MaxNormalForceTrajectoryManager;
class EndEffectorTrajectoryManager;
class HandTrajectoryManager;
class DCMTrajectoryManager;
class G1StateProvider;
class TaskHierarchyManager;
class ForceTrajectoryManager;

class G1ControlArchitecture : public ControlArchitecture {
public:
  G1ControlArchitecture(PinocchioRobotSystem *robot, const YAML::Node &cfg);
  virtual ~G1ControlArchitecture();

  void GetCommand(void *command) override;

  G1TCIContainer *tci_container_;

  UpperBodyTrajetoryManager *upper_body_tm_;
  FloatingBaseTrajectoryManager *floating_base_tm_;
  MaxNormalForceTrajectoryManager *lf_max_normal_froce_tm_;
  MaxNormalForceTrajectoryManager *rf_max_normal_froce_tm_;
  EndEffectorTrajectoryManager *lf_SE3_tm_;
  EndEffectorTrajectoryManager *rf_SE3_tm_;
  HandTrajectoryManager *lh_SE3_tm_;
  HandTrajectoryManager *rh_SE3_tm_;
  DCMTrajectoryManager *dcm_tm_;
  ForceTrajectoryManager *lf_force_tm_;
  ForceTrajectoryManager *rf_force_tm_;

  TaskHierarchyManager *lf_pos_hm_;
  TaskHierarchyManager *lf_ori_hm_;
  TaskHierarchyManager *rf_pos_hm_;
  TaskHierarchyManager *rf_ori_hm_;
  TaskHierarchyManager *lh_pos_hm_;
  TaskHierarchyManager *lh_ori_hm_;
  TaskHierarchyManager *rh_pos_hm_;
  TaskHierarchyManager *rh_ori_hm_;

private:
  G1Controller *controller_;
  G1StateProvider *sp_;
  DCMPlanner *dcm_planner_;

#if B_USE_FOXGLOVE
  std::unordered_map<std::string, int *> param_map_int_;
  std::unordered_map<std::string, double *> param_map_double_;
  std::unordered_map<std::string, TaskHierarchyManager *> param_map_hm_;
  FoxgloveParameterSubscriber *param_subscriber_;
#endif
};
