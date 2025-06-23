#pragma once

#include "g1.pb.h" //in build/messages/
#include "g1_definition.hpp"
#include <Eigen/Dense>
#include <memory>
#include <zmq.hpp>

struct G1Data {
public:
  G1Data() {};
  ~G1Data() = default;

  double time_ = 0;
  int phase_ = 1;

  Eigen::Vector3d est_base_joint_pos_ = Eigen::Vector3d::Zero();
  Eigen::VectorXd est_base_joint_ori_ = (Eigen::VectorXd(4) << 1.0, 0.0, 0.0, 0.0).finished();

  Eigen::Vector3d kf_base_joint_pos_ = Eigen::Vector3d::Zero();
  Eigen::VectorXd kf_base_joint_ori_ = (Eigen::VectorXd(4) << 1.0, 0.0, 0.0, 0.0).finished();

  Eigen::VectorXd joint_positions_ = Eigen::VectorXd::Zero(27);
  Eigen::VectorXd joint_velocities_ = Eigen::VectorXd::Zero(27);

  Eigen::Vector3d des_com_pos_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d act_com_pos_ = Eigen::Vector3d::Zero();

  Eigen::VectorXd lfoot_pos_ = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd rfoot_pos_ = Eigen::VectorXd::Zero(3);

  Eigen::VectorXd lhand_pos_ = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd rhand_pos_ = Eigen::VectorXd::Zero(3);

  Eigen::VectorXd lfoot_ori_ = (Eigen::VectorXd(4) << 1.0, 0.0, 0.0, 0.0).finished();
  Eigen::VectorXd rfoot_ori_ = (Eigen::VectorXd(4) << 1.0, 0.0, 0.0, 0.0).finished();

  Eigen::VectorXd lhand_ori_ = (Eigen::VectorXd(4) << 1.0, 0.0, 0.0, 0.0).finished();
  Eigen::VectorXd rhand_ori_ = (Eigen::VectorXd(4) << 1.0, 0.0, 0.0, 0.0).finished();

  Eigen::VectorXd lfoot_rf_cmd_ = Eigen::VectorXd::Zero(6);
  Eigen::VectorXd rfoot_rf_cmd_ = Eigen::VectorXd::Zero(6);

  bool b_lfoot_ = false;
  bool b_rfoot_ = false;
  double lfoot_volt_normal_raw_ = 0.;
  double rfoot_volt_normal_raw_ = 0.;
  double lfoot_rf_normal_ = 0.;
  double rfoot_rf_normal_ = 0.;
  double lfoot_rf_normal_filt_ = 0.;
  double rfoot_rf_normal_filt_ = 0.;

  Eigen::Vector2d est_icp = Eigen::Vector2d::Zero();
  Eigen::Vector2d des_icp = Eigen::Vector2d::Zero();

  Eigen::Vector2d des_cmp = Eigen::Vector2d::Zero();

  Eigen::Vector2d com_xy_weight = Eigen::Vector2d::Zero();
  Eigen::Vector2d com_xy_kp = Eigen::Vector2d::Zero();
  Eigen::Vector2d com_xy_kd = Eigen::Vector2d::Zero();
  Eigen::Vector2d com_xy_ki = Eigen::Vector2d::Zero();

  double com_z_weight = 0;
  double com_z_kp = 0;
  double com_z_kd = 0;

  Eigen::Vector3d torso_ori_weight = Eigen::Vector3d::Zero();
  Eigen::Vector3d torso_ori_kp = Eigen::Vector3d::Zero();
  Eigen::Vector3d torso_ori_kd = Eigen::Vector3d::Zero();

  Eigen::Vector3d lf_pos_weight = Eigen::Vector3d::Zero();
  Eigen::Vector3d lf_pos_kp = Eigen::Vector3d::Zero();
  Eigen::Vector3d lf_pos_kd = Eigen::Vector3d::Zero();

  Eigen::Vector3d rf_pos_weight = Eigen::Vector3d::Zero();
  Eigen::Vector3d rf_pos_kp = Eigen::Vector3d::Zero();
  Eigen::Vector3d rf_pos_kd = Eigen::Vector3d::Zero();

  Eigen::Vector3d lf_ori_weight = Eigen::Vector3d::Zero();
  Eigen::Vector3d lf_ori_kp = Eigen::Vector3d::Zero();
  Eigen::Vector3d lf_ori_kd = Eigen::Vector3d::Zero();

  Eigen::Vector3d rf_ori_weight = Eigen::Vector3d::Zero();
  Eigen::Vector3d rf_ori_kp = Eigen::Vector3d::Zero();
  Eigen::Vector3d rf_ori_kd = Eigen::Vector3d::Zero();

  Eigen::Quaterniond quat_world_local_ = Eigen::Quaterniond::Identity();

  // mpc data
  std::vector<Eigen::Vector3d> des_com_traj;
  std::vector<Eigen::Vector3d> des_torso_ori_traj;
  std::vector<Eigen::Vector3d> des_lf_pos_traj;
  std::vector<Eigen::Vector3d> des_rf_pos_traj;
  std::vector<Eigen::Vector3d> des_lf_ori_traj;
  std::vector<Eigen::Vector3d> des_rf_ori_traj;

  // offline planner data
  Eigen::VectorXd joint_pos_des;
  Eigen::VectorXd joint_vel_des;
  Eigen::VectorXd joint_trq_des;

  // MPC Costs:
  int total_iterations_;
  std::vector<double> xReg_costs_;
  std::vector<double> uReg_costs_;
  std::vector<double> xBound_costs_;
  std::vector<double> com_costs_;
  std::vector<double> torso_link_frame_costs_;
  std::vector<double> left_hand_frame_costs_;
  std::vector<double> right_hand_frame_costs_;
  std::vector<double> left_ankle_frame_costs_;
  std::vector<double> right_ankle_frame_costs_;
  std::vector<double> left_knee_frame_costs_;
  std::vector<double> right_knee_frame_costs_;

  // MPC Frame Desired Pose:
  Eigen::Vector3d torso_des_pos_;
  Eigen::Vector3d left_ankle_roll_des_pos_;
  Eigen::Vector3d right_ankle_roll_des_pos_;
  Eigen::Vector3d left_knee_des_pos_;
  Eigen::Vector3d right_knee_des_pos_;
  Eigen::Vector3d left_rubber_hand_des_pos_;
  Eigen::Vector3d right_rubber_hand_des_pos_;

  // MPC Frame Current Pose:
  Eigen::Vector3d torso_curr_pos_;
  Eigen::Vector3d left_ankle_roll_curr_pos_;
  Eigen::Vector3d right_ankle_roll_curr_pos_;
  Eigen::Vector3d left_knee_curr_pos_;
  Eigen::Vector3d right_knee_curr_pos_;
  Eigen::Vector3d left_rubber_hand_curr_pos_;
  Eigen::Vector3d right_rubber_hand_curr_pos_;

  Eigen::Vector3d com_curr_pos_;
  Eigen::Vector3d com_des_pos_;
  
  std::vector<Eigen::Vector3d> l_foot_rf_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> r_foot_rf_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> l_hand_rf_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> r_hand_rf_ = {Eigen::Vector3d::Zero()};

  // MPC Contact Costs:
  std::vector<double> left_hand_contact_costs_;
  std::vector<double> right_hand_contact_costs_;
  std::vector<double> left_foot_contact_costs_;
  std::vector<double> right_foot_contact_costs_;

  // MPC solve statistics
  bool b_fddp_feasible_ = false;
  float solve_duration_ = 0.0f;

  // MPC PredictedFrame Positions
  std::vector<Eigen::Vector3d> predicted_torso_pos_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> predicted_left_ankle_roll_pos_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> predicted_right_ankle_roll_pos_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> predicted_left_knee_pos_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> predicted_right_knee_pos_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> predicted_left_rubber_pos_ = {Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> predicted_right_rubber_pos_ = {Eigen::Vector3d::Zero()};
};

// Singleton class
class G1DataManager {
public:
  static G1DataManager *GetDataManager();
  ~G1DataManager() = default;

  void InitializeSocket(const std::string &ip_address);
  void SendData();

  bool IsInitialized(); // socket initialized boolean getter

  std::unique_ptr<G1Data> data_;

private:
  G1DataManager();

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t> socket_;

  bool b_initialize_socket_;
};
