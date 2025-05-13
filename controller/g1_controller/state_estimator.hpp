#pragma once

class G1SensorData;
class PinocchioRobotSystem;

class StateEstimator {
public:
  StateEstimator(PinocchioRobotSystem *robot) : robot_(robot) {};
  virtual ~StateEstimator() = default;

  virtual void Initialize(G1SensorData *sensor_Data) = 0;
  virtual void Update(G1SensorData *sensor_Data) = 0;

  // simulation only
  virtual void UpdateGroundTruthSensorData(G1SensorData *sensor_data) = 0;

protected:
  PinocchioRobotSystem *robot_;
};
