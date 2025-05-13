#pragma once
#include "controller/whole_body_controller/task.hpp"
#include "util/util.hpp"
#include <stdexcept>

// ====================================================================
// Note that this class is deprecated!!!
// ====================================================================

namespace com_feedback_source {
constexpr int kComFeedback = 0;
constexpr int kDcmFeedback = 1;
} // namespace com_feedback_source

namespace com_height_target_source {
constexpr int kComHeight = 0;
constexpr int kBaseHeight = 1;
} // namespace com_height_target_source

class G1StateProvider;

class G1ComTask : public Task {
public:
  G1ComTask(PinocchioRobotSystem *robot);

  virtual ~G1ComTask() = default;

  void UpdateOpCommand(const Eigen::Matrix3d &world_R_local =
                           Eigen::Matrix3d::Identity()) override;

  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

  void SetParameters(const YAML::Node &node, const WBC_TYPE wbc_type) override;

private:
  int com_feedback_source_;
  int com_height_target_source_;

  G1StateProvider *sp_;

  bool b_sim_;
};
