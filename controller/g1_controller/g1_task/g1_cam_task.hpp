#include "controller/whole_body_controller/task.hpp"
#include "util/util.hpp"

class PinocchioRobotSystem;
class G1StateProvider;

class G1CAMTask : public Task {
public:
  G1CAMTask(PinocchioRobotSystem *robot);
  ~G1CAMTask() = default;

  void UpdateOpCommand(const Eigen::Matrix3d &world_R_local =
                           Eigen::Matrix3d::Identity()) override;
  void UpdateJacobian() override;
  void UpdateJacobianDotQdot() override;

  void SetParameters(const YAML::Node &node, const WBC_TYPE wbc_type) override;

private:
  G1StateProvider *sp_;
};
