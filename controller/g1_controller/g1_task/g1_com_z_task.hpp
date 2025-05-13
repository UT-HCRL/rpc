#pragma once

#include "controller/whole_body_controller/task.hpp"
#include "util/util.hpp"

class G1StateProvider;
class PinocchioRobotSystem;

namespace com_height {
constexpr int kCoM = 0;
constexpr int kBase = 1;
} // namespace com_height

class G1CoMZTask : public Task {
public:
  G1CoMZTask(PinocchioRobotSystem *robot);
  ~G1CoMZTask() = default;

  void UpdateOpCommand(const Eigen::Matrix3d &world_R_local =
                           Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

  void SetParameters(const YAML::Node &node, const WBC_TYPE wbc_type) override;

private:
  G1StateProvider *sp_;
  int com_height_;
};
