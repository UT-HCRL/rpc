#pragma once

#include <Eigen/Dense>
#include <vector>

#include "configuration.hpp"
#include "controller/g1_controller/state_estimator.hpp"
#include "util/util.hpp"

#if B_USE_MATLOGGER
#include <matlogger2/matlogger2.h>
#include <matlogger2/utils/mat_appender.h>
#endif

namespace com_vel_filter {
constexpr int kMovingAverage = 0;
constexpr int kExponentialSmoother = 1;
constexpr int kLowPassFilter = 2;
} // namespace com_vel_filter

class G1StateProvider;
class SimpleMovingAverage;
class ExponentialMovingAverageFilter;
class LowPassVelocityFilter;

class G1StateEstimator : public StateEstimator {
public:
  G1StateEstimator(PinocchioRobotSystem *robot, const YAML::Node &cfg);
  virtual ~G1StateEstimator();

  void Initialize(G1SensorData *sensor_data) override;
  void Update(G1SensorData *sensor_data) override;

  // simulation only
  void UpdateGroundTruthSensorData(G1SensorData *sensor_data) override;

private:
  void _ComputeDCM();
  G1StateProvider *sp_;

  Eigen::Matrix3d R_imu_base_com_;

  Eigen::Vector3d global_leg_odometry_;
  Eigen::Vector3d prev_base_joint_pos_;

  bool b_first_visit_;
  bool b_lp_first_visit_ = true;

  int com_vel_filter_type_;
  std::vector<SimpleMovingAverage *> com_vel_mv_avg_filter_;
  ExponentialMovingAverageFilter *com_vel_exp_filter_;
  LowPassVelocityFilter *com_vel_lp_filter_;

#if B_USE_MATLOGGER
  XBot::MatLogger2::Ptr logger_;
  XBot::MatAppender::Ptr appender_;
#endif
};
