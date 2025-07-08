#include <controller/g1_controller/g1_data_manager.hpp>

G1DataManager *G1DataManager::GetDataManager()
{
  static G1DataManager dm;
  return &dm;
}

G1DataManager::G1DataManager()
{
  // g1 data initialize
  data_ = std::make_unique<G1Data>();
  data_->b_trq_limit_.resize(g1::n_adof, false);

  // zmq stuff initialize
  context_ = std::make_unique<zmq::context_t>(1);
  socket_ = std::make_unique<zmq::socket_t>(*context_, ZMQ_PUB);

  // boolean initialize
  b_initialize_socket_ = false;
}

bool G1DataManager::IsInitialized() { return b_initialize_socket_; }

void G1DataManager::InitializeSocket(const std::string &ip_address)
{
  socket_->bind(ip_address);
  b_initialize_socket_ = true;
}

void G1DataManager::SendData()
{
  assert(b_initialize_socket_);

  g1::pnc_msg msg;
  msg.set_time(data_->time_);
  // msg.set_phase(data_->phase_);

  g1::Pos base_pos_msg;
  base_pos_msg.set_x(data_->est_base_joint_pos_[0]);
  base_pos_msg.set_y(data_->est_base_joint_pos_[1]);
  base_pos_msg.set_z(data_->est_base_joint_pos_[2]);
  msg.mutable_est_base_joint_pos()->CopyFrom(base_pos_msg);

  g1::Quat base_ori_msg;
  base_ori_msg.set_x(data_->est_base_joint_ori_[0]);
  base_ori_msg.set_y(data_->est_base_joint_ori_[1]);
  base_ori_msg.set_z(data_->est_base_joint_ori_[2]);
  base_ori_msg.set_w(data_->est_base_joint_ori_[3]);
  msg.mutable_est_base_joint_ori()->CopyFrom(base_ori_msg);

  for (int i(0); i < data_->joint_positions_.size(); i++)
  {
    msg.add_joint_positions(data_->joint_positions_[i]);
    msg.add_joint_velocities(data_->joint_velocities_[i]);
  }

  g1::Pos lfoot_pos_msg;
  lfoot_pos_msg.set_x(data_->lfoot_pos_[0]);
  lfoot_pos_msg.set_y(data_->lfoot_pos_[1]);
  lfoot_pos_msg.set_z(data_->lfoot_pos_[2]);
  msg.mutable_lfoot_pos()->CopyFrom(lfoot_pos_msg);

  g1::Pos rfoot_pos_msg;
  rfoot_pos_msg.set_x(data_->rfoot_pos_[0]);
  rfoot_pos_msg.set_y(data_->rfoot_pos_[1]);
  rfoot_pos_msg.set_z(data_->rfoot_pos_[2]);
  msg.mutable_rfoot_pos()->CopyFrom(rfoot_pos_msg);

  g1::Quat lfoot_ori_msg;
  lfoot_ori_msg.set_x(data_->lfoot_ori_[0]);
  lfoot_ori_msg.set_y(data_->lfoot_ori_[1]);
  lfoot_ori_msg.set_z(data_->lfoot_ori_[2]);
  lfoot_ori_msg.set_w(data_->lfoot_ori_[3]);
  msg.mutable_lfoot_ori()->CopyFrom(lfoot_ori_msg);

  g1::Quat rfoot_ori_msg;
  rfoot_ori_msg.set_x(data_->rfoot_ori_[0]);
  rfoot_ori_msg.set_y(data_->rfoot_ori_[1]);
  rfoot_ori_msg.set_z(data_->rfoot_ori_[2]);
  rfoot_ori_msg.set_w(data_->rfoot_ori_[3]);
  msg.mutable_rfoot_ori()->CopyFrom(rfoot_ori_msg);

  g1::Pos lhand_pos_msg;
  lhand_pos_msg.set_x(data_->lhand_pos_[0]);
  lhand_pos_msg.set_y(data_->lhand_pos_[1]);
  lhand_pos_msg.set_z(data_->lhand_pos_[2]);
  msg.mutable_lhand_pos()->CopyFrom(lhand_pos_msg);

  g1::Pos rhand_pos_msg;
  rhand_pos_msg.set_x(data_->rhand_pos_[0]);
  rhand_pos_msg.set_y(data_->rhand_pos_[1]);
  rhand_pos_msg.set_z(data_->rhand_pos_[2]);
  msg.mutable_rhand_pos()->CopyFrom(rhand_pos_msg);

  g1::Quat lhand_ori_msg;
  lhand_ori_msg.set_x(data_->lhand_ori_[0]);
  lhand_ori_msg.set_y(data_->lhand_ori_[1]);
  lhand_ori_msg.set_z(data_->lhand_ori_[2]);
  lhand_ori_msg.set_w(data_->lhand_ori_[3]);
  msg.mutable_lhand_ori()->CopyFrom(lhand_ori_msg);

  g1::Quat rhand_ori_msg;
  rhand_ori_msg.set_x(data_->rhand_ori_[0]);
  rhand_ori_msg.set_y(data_->rhand_ori_[1]);
  rhand_ori_msg.set_z(data_->rhand_ori_[2]);
  rhand_ori_msg.set_w(data_->rhand_ori_[3]);
  msg.mutable_rhand_ori()->CopyFrom(rhand_ori_msg);

  // for (int i(0); i < data_->lfoot_rf_cmd_.size(); i++)
  //   msg.add_lfoot_rf_cmd(data_->lfoot_rf_cmd_[i]);
  //
  // for (int i(0); i < data_->rfoot_rf_cmd_.size(); i++)
  //   msg.add_rfoot_rf_cmd(data_->rfoot_rf_cmd_[i]);

  // contact sensing measurements
  msg.set_b_lfoot(data_->b_lfoot_);
  msg.set_b_rfoot(data_->b_rfoot_);
  msg.set_b_lhand(data_->b_lhand_);
  msg.set_b_rhand(data_->b_rhand_);

  // =============================================================
  // Sensor Data
  // =============================================================
  g1::Pos lf_contact_force_msg;
  lf_contact_force_msg.set_x(data_->lf_contact_force_(0));
  lf_contact_force_msg.set_y(data_->lf_contact_force_(1));
  lf_contact_force_msg.set_z(data_->lf_contact_force_(2));
  msg.mutable_lf_contact_force()->CopyFrom(lf_contact_force_msg);

  g1::Pos rf_contact_force_msg;
  rf_contact_force_msg.set_x(data_->rf_contact_force_(0));
  rf_contact_force_msg.set_y(data_->rf_contact_force_(1));
  rf_contact_force_msg.set_z(data_->rf_contact_force_(2));
  msg.mutable_rf_contact_force()->CopyFrom(rf_contact_force_msg);

  g1::Pos lh_contact_force_msg;
  lh_contact_force_msg.set_x(data_->lh_contact_force_(0));
  lh_contact_force_msg.set_y(data_->lh_contact_force_(1));
  lh_contact_force_msg.set_z(data_->lh_contact_force_(2));
  msg.mutable_lh_contact_force()->CopyFrom(lh_contact_force_msg);

  g1::Pos rh_contact_force_msg;
  rh_contact_force_msg.set_x(data_->rh_contact_force_(0));
  rh_contact_force_msg.set_y(data_->rh_contact_force_(1));
  rh_contact_force_msg.set_z(data_->rh_contact_force_(2));
  msg.mutable_rh_contact_force()->CopyFrom(rh_contact_force_msg);

  // =============================================================
  // MPC variables
  // =============================================================
  g1::Pos com_msg;
  for (const auto &des_com : data_->des_com_traj)
  {
    com_msg.set_x(des_com(0));
    com_msg.set_y(des_com(1));
    com_msg.set_z(des_com(2));
    msg.add_des_com_traj()->CopyFrom(com_msg);
  }

  g1::Euler ori_msg;
  for (const auto &des_ori : data_->des_torso_ori_traj)
  {
    ori_msg.set_x(des_ori(0));
    ori_msg.set_y(des_ori(1));
    ori_msg.set_z(des_ori(2));
    msg.add_des_torso_ori_traj()->CopyFrom(ori_msg);
  }

  g1::Pos lf_pos_msg;
  for (const auto &des_pos : data_->des_lf_pos_traj)
  {
    lf_pos_msg.set_x(des_pos(0));
    lf_pos_msg.set_y(des_pos(1));
    lf_pos_msg.set_z(des_pos(2));
    msg.add_des_lf_pos_traj()->CopyFrom(lf_pos_msg);
  }
  g1::Pos rf_pos_msg;
  for (const auto &des_pos : data_->des_rf_pos_traj)
  {
    rf_pos_msg.set_x(des_pos(0));
    rf_pos_msg.set_y(des_pos(1));
    rf_pos_msg.set_z(des_pos(2));
    msg.add_des_rf_pos_traj()->CopyFrom(rf_pos_msg);
  }

  g1::Euler lf_ori_msg;
  for (const auto &des_ori : data_->des_lf_ori_traj)
  {
    lf_ori_msg.set_x(des_ori(0));
    lf_ori_msg.set_y(des_ori(1));
    lf_ori_msg.set_z(des_ori(2));
    msg.add_des_lf_ori_traj()->CopyFrom(lf_ori_msg);
  }

  g1::Euler rf_ori_msg;
  for (const auto &des_ori : data_->des_rf_ori_traj)
  {
    rf_ori_msg.set_x(des_ori(0));
    rf_ori_msg.set_y(des_ori(1));
    rf_ori_msg.set_z(des_ori(2));
    msg.add_des_rf_ori_traj()->CopyFrom(rf_ori_msg);
  }

  // =============================================================
  // Offline Planner
  // =============================================================
  for (int i(0); i < data_->joint_positions_.size(); i++)
  {
    msg.add_joint_pos_des(data_->joint_pos_des[i]);
    msg.add_joint_vel_des(data_->joint_vel_des[i]);
    msg.add_joint_trq_des(data_->joint_trq_des[i]);
  }

  // =============================================================
  // MPC Costs
  // =============================================================
  msg.set_total_iterations(data_->total_iterations_);
  for (int i(0); i < data_->xReg_costs_.size(); i++)
    msg.add_xreg_costs(data_->xReg_costs_[i]);
  for (int i(0); i < data_->uReg_costs_.size(); i++)
    msg.add_ureg_costs(data_->uReg_costs_[i]);
  for (int i(0); i < data_->xBound_costs_.size(); i++)
    msg.add_xbound_costs(data_->xBound_costs_[i]);
  for (int i(0); i < data_->com_costs_.size(); i++)
    msg.add_com_costs(data_->com_costs_[i]);
  for (int i(0); i < data_->left_hand_frame_costs_.size(); i++)
    msg.add_left_hand_frame_costs(data_->left_hand_frame_costs_[i]);
  for (int i(0); i < data_->right_hand_frame_costs_.size(); i++)
    msg.add_right_hand_frame_costs(data_->right_hand_frame_costs_[i]);
  for (int i(0); i < data_->left_ankle_frame_costs_.size(); i++)
    msg.add_left_ankle_frame_costs(data_->left_ankle_frame_costs_[i]);
  for (int i(0); i < data_->right_ankle_frame_costs_.size(); i++)
    msg.add_right_ankle_frame_costs(data_->right_ankle_frame_costs_[i]);
  for (int i(0); i < data_->left_knee_frame_costs_.size(); i++)
    msg.add_left_knee_frame_costs(data_->left_knee_frame_costs_[i]);
  for (int i(0); i < data_->right_knee_frame_costs_.size(); i++)
    msg.add_right_knee_frame_costs(data_->right_knee_frame_costs_[i]);
  for (int i(0); i < data_->torso_link_frame_costs_.size(); i++)
    msg.add_torso_link_frame_costs(data_->torso_link_frame_costs_[i]);
  for (int i(0); i < data_->left_hand_contact_costs_.size(); i++)
    msg.add_left_hand_contact_costs(data_->left_hand_contact_costs_[i]);
  for (int i(0); i < data_->right_hand_contact_costs_.size(); i++)
    msg.add_right_hand_contact_costs(data_->right_hand_contact_costs_[i]);
  for (int i(0); i < data_->left_foot_contact_costs_.size(); i++)
    msg.add_left_foot_contact_costs(data_->left_foot_contact_costs_[i]);
  for (int i(0); i < data_->right_foot_contact_costs_.size(); i++)
    msg.add_right_foot_contact_costs(data_->right_foot_contact_costs_[i]);

  // =============================================================
  // MPC Feasibility
  // =============================================================
  msg.set_b_fddp_feasible(data_->b_fddp_feasible_);
  msg.set_solve_duration(data_->solve_duration_);

  // =============================================================
  // MPC Desired Positions obtained from Offline Plan
  // =============================================================
  g1::Pos torso_des_pos_msg;
  torso_des_pos_msg.set_x(data_->torso_des_pos_(0));
  torso_des_pos_msg.set_y(data_->torso_des_pos_(1));
  torso_des_pos_msg.set_z(data_->torso_des_pos_(2));
  msg.mutable_torso_des_pos()->CopyFrom(torso_des_pos_msg);
  
  g1::Pos left_ankle_roll_des_pos_msg;
  left_ankle_roll_des_pos_msg.set_x(data_->left_ankle_roll_des_pos_(0));
  left_ankle_roll_des_pos_msg.set_y(data_->left_ankle_roll_des_pos_(1));
  left_ankle_roll_des_pos_msg.set_z(data_->left_ankle_roll_des_pos_(2));
  msg.mutable_left_ankle_roll_des_pos()->CopyFrom(left_ankle_roll_des_pos_msg);
  
  g1::Pos right_ankle_roll_des_pos_msg;
  right_ankle_roll_des_pos_msg.set_x(data_->right_ankle_roll_des_pos_(0));
  right_ankle_roll_des_pos_msg.set_y(data_->right_ankle_roll_des_pos_(1));
  right_ankle_roll_des_pos_msg.set_z(data_->right_ankle_roll_des_pos_(2));
  msg.mutable_right_ankle_roll_des_pos()->CopyFrom(right_ankle_roll_des_pos_msg);
  
  g1::Pos left_knee_des_pos_msg;
  left_knee_des_pos_msg.set_x(data_->left_knee_des_pos_(0));
  left_knee_des_pos_msg.set_y(data_->left_knee_des_pos_(1));
  left_knee_des_pos_msg.set_z(data_->left_knee_des_pos_(2));
  msg.mutable_left_knee_des_pos()->CopyFrom(left_knee_des_pos_msg);
  
  g1::Pos right_knee_des_pos_msg;
  right_knee_des_pos_msg.set_x(data_->right_knee_des_pos_(0));
  right_knee_des_pos_msg.set_y(data_->right_knee_des_pos_(1));
  right_knee_des_pos_msg.set_z(data_->right_knee_des_pos_(2));
  msg.mutable_right_knee_des_pos()->CopyFrom(right_knee_des_pos_msg);
  
  g1::Pos left_rubber_hand_des_pos_msg;
  left_rubber_hand_des_pos_msg.set_x(data_->left_rubber_hand_des_pos_(0));
  left_rubber_hand_des_pos_msg.set_y(data_->left_rubber_hand_des_pos_(1));
  left_rubber_hand_des_pos_msg.set_z(data_->left_rubber_hand_des_pos_(2));
  msg.mutable_left_rubber_hand_des_pos()->CopyFrom(left_rubber_hand_des_pos_msg);
  
  g1::Pos right_rubber_hand_des_pos_msg;
  right_rubber_hand_des_pos_msg.set_x(data_->right_rubber_hand_des_pos_(0));
  right_rubber_hand_des_pos_msg.set_y(data_->right_rubber_hand_des_pos_(1));
  right_rubber_hand_des_pos_msg.set_z(data_->right_rubber_hand_des_pos_(2));
  msg.mutable_right_rubber_hand_des_pos()->CopyFrom(right_rubber_hand_des_pos_msg);

  g1::Pos com_des_pos_msg;
  com_des_pos_msg.set_x(data_->com_des_pos_(0));
  com_des_pos_msg.set_y(data_->com_des_pos_(1));
  com_des_pos_msg.set_z(data_->com_des_pos_(2));
  msg.mutable_com_des_pos()->CopyFrom(com_des_pos_msg);

  for (int i(0); i < data_->right_foot_contact_costs_.size(); i++){
    g1::Pos l_foot_rf_msg;
    l_foot_rf_msg.set_x(data_->l_foot_rf_[i](0));
    l_foot_rf_msg.set_y(data_->l_foot_rf_[i](1));
    l_foot_rf_msg.set_z(data_->l_foot_rf_[i](2));
    msg.add_l_foot_rf()->CopyFrom(l_foot_rf_msg);
  }

  for (int i(0); i < data_->left_foot_contact_costs_.size(); i++){
    g1::Pos r_foot_rf_msg;
    r_foot_rf_msg.set_x(data_->r_foot_rf_[i](0));
    r_foot_rf_msg.set_y(data_->r_foot_rf_[i](1));
    r_foot_rf_msg.set_z(data_->r_foot_rf_[i](2));
    msg.add_r_foot_rf()->CopyFrom(r_foot_rf_msg);
  }
  
  for (int i(0); i < data_->left_hand_contact_costs_.size(); i++){
    g1::Pos l_hand_rf_msg;
    l_hand_rf_msg.set_x(data_->l_hand_rf_[i](0));
    l_hand_rf_msg.set_y(data_->l_hand_rf_[i](1));
    l_hand_rf_msg.set_z(data_->l_hand_rf_[i](2));
    msg.add_l_hand_rf()->CopyFrom(l_hand_rf_msg);
  }

  for (int i(0); i < data_->right_hand_contact_costs_.size(); i++){
    g1::Pos r_hand_rf_msg;
    r_hand_rf_msg.set_x(data_->r_hand_rf_[i](0));
    r_hand_rf_msg.set_y(data_->r_hand_rf_[i](1));
    r_hand_rf_msg.set_z(data_->r_hand_rf_[i](2));
    msg.add_r_hand_rf()->CopyFrom(r_hand_rf_msg);
  }

  for (int i(0); i < data_->torso_link_frame_costs_.size(); i++){
    g1::Pos t_predicted_frame_msg;
    t_predicted_frame_msg.set_x(data_->predicted_torso_pos_[i](0));
    t_predicted_frame_msg.set_y(data_->predicted_torso_pos_[i](1));
    t_predicted_frame_msg.set_z(data_->predicted_torso_pos_[i](2));
    msg.add_predicted_torso()->CopyFrom(t_predicted_frame_msg);
  }

  for (int i(0); i < data_->left_ankle_frame_costs_.size(); i++){
    g1::Pos la_predicted_frame_msg;
    la_predicted_frame_msg.set_x(data_->predicted_left_ankle_roll_pos_[i](0));
    la_predicted_frame_msg.set_y(data_->predicted_left_ankle_roll_pos_[i](1));
    la_predicted_frame_msg.set_z(data_->predicted_left_ankle_roll_pos_[i](2));
    msg.add_predicted_lankle()->CopyFrom(la_predicted_frame_msg);
  }

  for (int i(0); i < data_->right_ankle_frame_costs_.size(); i++){
    g1::Pos ra_predicted_frame_msg;
    ra_predicted_frame_msg.set_x(data_->predicted_right_ankle_roll_pos_[i](0));
    ra_predicted_frame_msg.set_y(data_->predicted_right_ankle_roll_pos_[i](1));
    ra_predicted_frame_msg.set_z(data_->predicted_right_ankle_roll_pos_[i](2));
    msg.add_predicted_rankle()->CopyFrom(ra_predicted_frame_msg);
  }

  for (int i(0); i < data_->left_knee_frame_costs_.size(); i++){
    g1::Pos lk_predicted_frame_msg;
    lk_predicted_frame_msg.set_x(data_->predicted_left_knee_pos_[i](0));
    lk_predicted_frame_msg.set_y(data_->predicted_left_knee_pos_[i](1));
    lk_predicted_frame_msg.set_z(data_->predicted_left_knee_pos_[i](2));
    msg.add_predicted_lknee()->CopyFrom(lk_predicted_frame_msg);
  }

  for (int i(0); i < data_->right_knee_frame_costs_.size(); i++){
    g1::Pos rk_predicted_frame_msg;
    rk_predicted_frame_msg.set_x(data_->predicted_right_knee_pos_[i](0));
    rk_predicted_frame_msg.set_y(data_->predicted_right_knee_pos_[i](1));
    rk_predicted_frame_msg.set_z(data_->predicted_right_knee_pos_[i](2));
    msg.add_predicted_rknee()->CopyFrom(rk_predicted_frame_msg);
  }

  for (int i(0); i < data_->left_hand_frame_costs_.size(); i++){
    g1::Pos lh_predicted_frame_msg;
    lh_predicted_frame_msg.set_x(data_->predicted_left_rubber_pos_[i](0));
    lh_predicted_frame_msg.set_y(data_->predicted_left_rubber_pos_[i](1));
    lh_predicted_frame_msg.set_z(data_->predicted_left_rubber_pos_[i](2));
    msg.add_predicted_lhand()->CopyFrom(lh_predicted_frame_msg);
  }

  for (int i(0); i < data_->right_hand_frame_costs_.size(); i++){
    g1::Pos rh_predicted_frame_msg;
    rh_predicted_frame_msg.set_x(data_->predicted_right_rubber_pos_[i](0));
    rh_predicted_frame_msg.set_y(data_->predicted_right_rubber_pos_[i](1));
    rh_predicted_frame_msg.set_z(data_->predicted_right_rubber_pos_[i](2));
    msg.add_predicted_rhand()->CopyFrom(rh_predicted_frame_msg);
  }

  // =============================================================

  // =============================================================
  // MPC Current Poses
  // =============================================================
  g1::Pos torso_curr_pos_msg;
  torso_curr_pos_msg.set_x(data_->torso_curr_pos_(0));
  torso_curr_pos_msg.set_y(data_->torso_curr_pos_(1));
  torso_curr_pos_msg.set_z(data_->torso_curr_pos_(2));
  msg.mutable_torso_curr_pos()->CopyFrom(torso_curr_pos_msg);
  
  g1::Pos left_ankle_roll_curr_pos_msg;
  left_ankle_roll_curr_pos_msg.set_x(data_->left_ankle_roll_curr_pos_(0));
  left_ankle_roll_curr_pos_msg.set_y(data_->left_ankle_roll_curr_pos_(1));
  left_ankle_roll_curr_pos_msg.set_z(data_->left_ankle_roll_curr_pos_(2));
  msg.mutable_left_ankle_roll_curr_pos()->CopyFrom(left_ankle_roll_curr_pos_msg);
  
  g1::Pos right_ankle_roll_curr_pos_msg;
  right_ankle_roll_curr_pos_msg.set_x(data_->right_ankle_roll_curr_pos_(0));
  right_ankle_roll_curr_pos_msg.set_y(data_->right_ankle_roll_curr_pos_(1));
  right_ankle_roll_curr_pos_msg.set_z(data_->right_ankle_roll_curr_pos_(2));
  msg.mutable_right_ankle_roll_curr_pos()->CopyFrom(right_ankle_roll_curr_pos_msg);
  
  g1::Pos left_knee_curr_pos_msg;
  left_knee_curr_pos_msg.set_x(data_->left_knee_curr_pos_(0));
  left_knee_curr_pos_msg.set_y(data_->left_knee_curr_pos_(1));
  left_knee_curr_pos_msg.set_z(data_->left_knee_curr_pos_(2));
  msg.mutable_left_knee_curr_pos()->CopyFrom(left_knee_curr_pos_msg);
  
  g1::Pos right_knee_curr_pos_msg;
  right_knee_curr_pos_msg.set_x(data_->right_knee_curr_pos_(0));
  right_knee_curr_pos_msg.set_y(data_->right_knee_curr_pos_(1));
  right_knee_curr_pos_msg.set_z(data_->right_knee_curr_pos_(2));
  msg.mutable_right_knee_curr_pos()->CopyFrom(right_knee_curr_pos_msg);
  
  g1::Pos left_rubber_hand_curr_pos_msg;
  left_rubber_hand_curr_pos_msg.set_x(data_->left_rubber_hand_curr_pos_(0));
  left_rubber_hand_curr_pos_msg.set_y(data_->left_rubber_hand_curr_pos_(1));
  left_rubber_hand_curr_pos_msg.set_z(data_->left_rubber_hand_curr_pos_(2));
  msg.mutable_left_rubber_hand_curr_pos()->CopyFrom(left_rubber_hand_curr_pos_msg);
  
  g1::Pos right_rubber_hand_curr_pos_msg;
  right_rubber_hand_curr_pos_msg.set_x(data_->right_rubber_hand_curr_pos_(0));
  right_rubber_hand_curr_pos_msg.set_y(data_->right_rubber_hand_curr_pos_(1));
  right_rubber_hand_curr_pos_msg.set_z(data_->right_rubber_hand_curr_pos_(2));
  msg.mutable_right_rubber_hand_curr_pos()->CopyFrom(right_rubber_hand_curr_pos_msg);

  g1::Pos com_curr_pos_msg;
  com_curr_pos_msg.set_x(data_->com_curr_pos_(0));
  com_curr_pos_msg.set_y(data_->com_curr_pos_(1));
  com_curr_pos_msg.set_z(data_->com_curr_pos_(2));
  msg.mutable_com_curr_pos()->CopyFrom(com_curr_pos_msg);

  for (int i(0); i < data_->b_trq_limit_.size(); i++)
    msg.add_b_trq_limit(data_->b_trq_limit_[i]);

  // serialize msg in string type
  std::string encoded_msg;
  msg.SerializeToString(&encoded_msg);

  // send data
  zmq::message_t zmq_msg(encoded_msg.size());
  memcpy((void *)zmq_msg.data(), encoded_msg.c_str(), encoded_msg.size());
  socket_->send(zmq_msg);
}
