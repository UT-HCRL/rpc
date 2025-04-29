#pragma once
#include <Eigen/Dense>
#include <iostream>

#include "controller/robot_system/pinocchio_robot_system.hpp"

class RobotCommand {
public:
  RobotCommand(PinocchioRobotSystem *robot)
      : robot_(robot) {

    int dim_pos = robot_->NumActiveDof();
    int dim_vel = robot_->NumQdot();
    des_pos_ = Eigen::VectorXd::Zero(dim_pos);
    des_vel_ = Eigen::VectorXd::Zero(dim_vel);
    des_trq_ = Eigen::VectorXd::Zero(dim_vel);

  }
  ~RobotCommand() = default;

  // for orientation task, des_pos is a 4 dimensional vector [x,y,z,w]
  // for angular momentum task, des_pos is ignored
  void UpdateDesired(const Eigen::VectorXd &des_pos,
                     const Eigen::VectorXd &des_vel,
                     const Eigen::VectorXd &des_trq) {
    des_pos_ = des_pos;
    des_vel_ = des_vel;
    des_trq_ = des_trq;
  }

  // getter function
  Eigen::VectorXd DesiredPos() const { return des_pos_; }
  Eigen::VectorXd DesiredVel() const { return des_vel_; }
  Eigen::VectorXd DesiredTrq() const { return des_trq_; }

  // Debug
  void Debug() {
    std::cout << "=================================" << std::endl;
    std::cout << "des_pos: " << des_pos_.transpose() << std::endl;
    std::cout << "des_vel: " << des_vel_.transpose() << std::endl;
    std::cout << "des_trq: " << des_trq_.transpose() << std::endl;
  }

protected:
  PinocchioRobotSystem *robot_;

  //  desired quantities
  Eigen::VectorXd des_pos_;
  Eigen::VectorXd des_vel_;
  Eigen::VectorXd des_trq_;
};
