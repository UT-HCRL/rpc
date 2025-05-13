#pragma once
#include "controller/control_architecture.hpp"
#if B_USE_FOXGLOVE
#include "UI/foxglove/client/parameter_subscriber.hpp"
#endif
#include "util/util.hpp"
#include <any>

namespace g1_states_wbic {
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
constexpr int kMPCLocomotion = 11;
// constexpr int kTeleopManipulation = 20;
} // namespace g1_states_wbic

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
class ForceTrajectoryManager;
class QPParamsManager;
class ConvexMPCLocomotion;
class GaitParams;
class MPCParams;
class CompositeRigidBodyInertia;

class G1ControlArchitecture_WBIC : public ControlArchitecture {
public:
  G1ControlArchitecture_WBIC(PinocchioRobotSystem *robot,
                                const YAML::Node &cfg);
  virtual ~G1ControlArchitecture_WBIC();

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
  QPParamsManager *qp_pm_;
  ConvexMPCLocomotion *convex_mpc_locomotion_;
  CompositeRigidBodyInertia *g1_crbi_model_;
  GaitParams *mpc_gait_params_;
  MPCParams *mpc_params_;

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
