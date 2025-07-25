#include "controller/g1_controller/g1_state_machines/track_plan.hpp"
#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_definition.hpp"
#include "controller/g1_controller/g1_state_provider.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "controller/g1_controller/g1_tci_container.hpp"

#if B_USE_ZMQ
  #include "controller/g1_controller/g1_data_manager.hpp"
#endif

#include "util/util.hpp"
#include "fstream"

namespace {
  double kGravity = 9.81;
}

TrackPlan::TrackPlan(const StateId state_id,
                     PinocchioRobotSystem *robot,
                     std::vector<pkl_utils::CompositeBezierCurve> planned_bezier_curves,
                     std::vector<Vector3d> planned_com_des,
                     std::vector<double> planned_time,
                     G1ControlArchitecture *ctrl_arch)
    : StateMachine(state_id, robot), ctrl_arch_(ctrl_arch), rf_z_max_interp_duration_(0.), b_tracking_plan_(false) {
  util::PrettyConstructor(2, "TrackPlan");

  has_new_data_ = false;
  sp_ = G1StateProvider::GetStateProvider();
  init_reaction_force_.setZero();
  des_reaction_force_.setZero();
  double half_mass = robot_->GetTotalMass() / 2.;
  init_reaction_force_(5) = half_mass * kGravity;
  des_reaction_force_(5) = half_mass * kGravity;
  
  std::string r_file_path = THIS_COM "/robot_model/g1/g1_29dof_lock_waist.urdf";
  const std::unordered_map<std::string, mpc_utils::Weights> gains = {
      {"torso",  mpc_utils::fromValues(1.0, 5., 0.5, 0.8, 0.8, 0.8)},
      {"feet",   mpc_utils::fromValues(8.0, 8.0, 8.0, 0.00001, 0.00001, 0.00001)},
      {"L_knee", mpc_utils::fromValues(4.0, 4.0, 4.0, 0.00001, 0.00001, 0.00001)},
      {"R_knee", mpc_utils::fromValues(4.0, 4.0, 4.0, 0.00001, 0.00001, 0.00001)},
      {"hands",  mpc_utils::fromValues(2.0, 2.0, 2.0, 0.00001, 0.00001, 0.00001)},
  };

  g1_mpc_ = std::make_unique<HumanoidMulticontactTracker>(r_file_path, gains);
  // g1_mpc_->printModel();

  std::vector<std::shared_ptr<pkl_utils::CompositeBezierCurve>> bezier_curves_ptrs;
  // Convert to pointers
  for (const auto& curve : planned_bezier_curves) {
    bezier_curves_ptrs.push_back(std::make_shared<pkl_utils::CompositeBezierCurve>(curve));
  }

  std::vector<std::string> target_names = g1_mpc_->getTargetFrameNames();
  bezier_curves_mgr_ = std::make_unique<pkl_utils::BezierCurvesManager>(bezier_curves_ptrs, target_names);
  com_des_ =  planned_com_des;
}

TrackPlan::~TrackPlan() {
  delete sp_;
  delete ctrl_arch_;
}

void TrackPlan::FirstVisit() {
  std::cout << "g1_states::kTrackPlan" << std::endl;
  state_machine_start_time_ = sp_->current_time_;

  if (b_wait_complete_){
    std::cout << "[State Machine] Warning: boolean set for MPC to complete solution before stepping MuJoCo physics." << std::endl;
  }else{
    run_threads_ = true;
    compute_thread_ = std::thread(&TrackPlan::Compute, this);
  }
}

void TrackPlan::OneStep() {

  if (b_wait_complete_){
    ComputeSync(); // Compute Sync is called on the same thread to 
                   // ensure that the MPC solution is ready before stepping MuJoCo physics.

    // std::this_thread::sleep_for(std::chrono::milliseconds(40)); // NOTE: Add sleep to simulate worse performance

    ctrl_arch_->tci_container_->robot_commands_->UpdateDesired(
        mpc_q_,
        mpc_q_dot_,
        mpc_tau_
    );
  }
  else{
    static auto last_time = std::chrono::steady_clock::now();
    const double desired_period = 1.0 / desired_frequency_;

    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_time = current_time - last_time;

    if (elapsed_time.count() >= desired_period) {
      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        if (has_new_data_) {

          has_new_data_ = false;
          ctrl_arch_->tci_container_->robot_commands_->UpdateDesired(
            mpc_q_,
            mpc_q_dot_,
            mpc_tau_
          );
        }
      }
      last_time = current_time;
    } else {
      double sleep_time = desired_period - elapsed_time.count();
      std::cout << "[DEBUG] Sleeping for " << sleep_time << " seconds to maintain desired frequency." << std::endl;
      std::this_thread::sleep_for(std::chrono::duration<double>(sleep_time));
    }
  }
}

void TrackPlan::ComputeSync(){

  static Eigen::VectorXd x0 = Eigen::VectorXd::Zero(g1_mpc_->getX0Size());
  static std::vector<Eigen::VectorXd> xs_out(g1_mpc_->getNhorizon() + 1, x0);
  static std::vector<Eigen::VectorXd> us_out(g1_mpc_->getNhorizon(), Eigen::VectorXd::Zero(27));
  
  static Eigen::Vector3d com_ref;
  com_ref = robot_->GetRobotComPos();

  static bool first_iteration = true;

  static mpc_utils::MPCData data_out;

  double controller_time = sp_->current_time_ - state_machine_start_time_;
  double fake_time = controller_time;

  std::vector<std::unordered_map<std::string, pinocchio::SE3>> desired_frames_vec;
  std::vector<Eigen::Vector3d> desired_com_vec;
  desired_frames_vec.resize(g1_mpc_->getNhorizon() +1);
  desired_com_vec.resize(g1_mpc_->getNhorizon() + 1);

  for(int i=0; i<g1_mpc_->getNhorizon() + 1; i++){
    std::unordered_map<std::string, pinocchio::SE3> desired_frames;
    const double t = controller_time + i * g1_mpc_->getDt();
    
    for(const auto& frame_name : g1_mpc_->getTargetFrameNames()) {
      pinocchio::SE3 temp_pose;
      temp_pose.setIdentity();
      temp_pose.translation() = bezier_curves_mgr_->getCurrentDesiredPosition(frame_name, t);
      desired_frames[frame_name] = temp_pose;

    }
    desired_frames_vec[i] = desired_frames;
    if (!com_des_.empty()) {
      desired_com_vec[i] = pkl_utils::get_com_des_pos(com_des_, t, g1_mpc_->getDt());
    }

  }

  xs_out[0] << robot_->GetQ(), robot_->GetQdot();
  if(first_iteration){
    us_out[0] = sp_->curr_joint_trq_cmd_.tail(27);
    first_iteration = false;
  }

  g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, controller_time);

  mpc_q_ = xs_out[0].head(robot_->GetQ().size()).tail(robot_->NumActiveDof()); // q_joints
  mpc_q_dot_ = xs_out[0].tail(robot_->NumActiveDof());  // qdot_joints
  mpc_tau_ = us_out[0];

  #if B_USE_ZMQ
    G1DataManager *dm = G1DataManager::GetDataManager();
    dm->data_->total_iterations_ = data_out.total_iterations;
    
    dm->data_->xReg_costs_.resize(data_out.xReg_costs.size());
    dm->data_->xReg_costs_ = data_out.xReg_costs;
    
    dm->data_->uReg_costs_.resize(data_out.uReg_costs.size());
    dm->data_->uReg_costs_ = data_out.uReg_costs;
    
    dm->data_->xBound_costs_.resize(data_out.xBound_costs.size());
    dm->data_->xBound_costs_ = data_out.xBound_costs;

    dm->data_->com_costs_.resize(data_out.com_costs.size());
    dm->data_->com_costs_ = data_out.com_costs;

    dm->data_->torso_des_pos_ = desired_frames_vec[0].at("torso_primitive_shape").translation();
    dm->data_->right_knee_des_pos_ = desired_frames_vec[0].at("right_knee_link").translation();
    dm->data_->left_knee_des_pos_ = desired_frames_vec[0].at("left_knee_link").translation();
    dm->data_->right_ankle_roll_des_pos_ = desired_frames_vec[0].at("right_ankle_roll_link").translation();
    dm->data_->left_ankle_roll_des_pos_ = desired_frames_vec[0].at("left_ankle_roll_link").translation();
    dm->data_->left_rubber_hand_des_pos_ = desired_frames_vec[0].at("left_rubber_hand").translation();
    dm->data_->right_rubber_hand_des_pos_ = desired_frames_vec[0].at("right_rubber_hand").translation();
    dm->data_->com_des_pos_ = desired_com_vec[0];

    dm->data_->torso_curr_pos_ = data_out.frame_current_pos["torso_primitive_shape"];
    dm->data_->left_ankle_roll_curr_pos_ = data_out.frame_current_pos["left_ankle_roll_link"];
    dm->data_->right_ankle_roll_curr_pos_ = data_out.frame_current_pos["right_ankle_roll_link"];
    dm->data_->left_knee_curr_pos_ = data_out.frame_current_pos["left_knee_link"];
    dm->data_->right_knee_curr_pos_ = data_out.frame_current_pos["right_knee_link"];
    dm->data_->left_rubber_hand_curr_pos_ = data_out.frame_current_pos["left_rubber_hand"];
    dm->data_->right_rubber_hand_curr_pos_ = data_out.frame_current_pos["right_rubber_hand"];
    dm->data_->com_curr_pos_ = data_out.com_curr_pos;

    dm->data_->left_hand_frame_costs_.resize(data_out.left_hand_frame_costs.size());
    dm->data_->right_hand_frame_costs_.resize(data_out.right_hand_frame_costs.size());
    dm->data_->left_hand_frame_costs_ = data_out.left_hand_frame_costs;
    dm->data_->right_hand_frame_costs_ = data_out.right_hand_frame_costs;

    dm->data_->left_ankle_frame_costs_.resize(data_out.left_ankle_frame_costs.size());
    dm->data_->right_ankle_frame_costs_.resize(data_out.right_ankle_frame_costs.size());
    dm->data_->left_ankle_frame_costs_ = data_out.left_ankle_frame_costs;
    dm->data_->right_ankle_frame_costs_ = data_out.right_ankle_frame_costs;

    dm->data_->left_knee_frame_costs_.resize(data_out.left_knee_frame_costs.size());
    dm->data_->right_knee_frame_costs_.resize(data_out.right_knee_frame_costs.size());
    dm->data_->left_knee_frame_costs_ = data_out.left_knee_frame_costs;
    dm->data_->right_knee_frame_costs_ = data_out.right_knee_frame_costs;

    dm->data_->torso_link_frame_costs_.resize(data_out.torso_link_frame_costs.size());
    dm->data_->torso_link_frame_costs_ = data_out.torso_link_frame_costs;

    dm->data_->left_hand_contact_costs_.resize(data_out.left_hand_contact_costs.size());
    dm->data_->right_hand_contact_costs_.resize(data_out.right_hand_contact_costs.size());
    dm->data_->left_hand_contact_costs_ = data_out.left_hand_contact_costs;
    dm->data_->right_hand_contact_costs_ = data_out.right_hand_contact_costs;

    dm->data_->left_foot_contact_costs_.resize(data_out.left_foot_contact_costs.size());
    dm->data_->right_foot_contact_costs_.resize(data_out.right_foot_contact_costs.size());
    dm->data_->left_foot_contact_costs_ = data_out.left_foot_contact_costs;
    dm->data_->right_foot_contact_costs_ = data_out.right_foot_contact_costs;

    dm->data_->l_foot_rf_.resize(g1_mpc_->getNhorizon());
    dm->data_->r_foot_rf_.resize(g1_mpc_->getNhorizon());
    dm->data_->l_hand_rf_.resize(g1_mpc_->getNhorizon());
    dm->data_->r_hand_rf_.resize(g1_mpc_->getNhorizon());

    dm->data_->predicted_torso_pos_.resize(g1_mpc_->getNhorizon());
    dm->data_->predicted_left_ankle_roll_pos_.resize(g1_mpc_->getNhorizon());
    dm->data_->predicted_right_ankle_roll_pos_.resize(g1_mpc_->getNhorizon());
    dm->data_->predicted_left_knee_pos_.resize(g1_mpc_->getNhorizon());
    dm->data_->predicted_right_knee_pos_.resize(g1_mpc_->getNhorizon());
    dm->data_->predicted_left_rubber_pos_.resize(g1_mpc_->getNhorizon());
    dm->data_->predicted_right_rubber_pos_.resize(g1_mpc_->getNhorizon());

    dm->data_->joint_vel_des_traj_.resize(g1_mpc_->getNhorizon());

    for (int i = 0; i < g1_mpc_->getNhorizon(); ++i) {
      dm->data_->l_foot_rf_[i] = data_out.contact_forces["l_foot_contact_contact_" + std::to_string(i)];
      dm->data_->r_foot_rf_[i] = data_out.contact_forces["r_foot_contact_contact_" + std::to_string(i)];
      dm->data_->l_hand_rf_[i] = data_out.contact_forces["left_rubber_hand_contact_" + std::to_string(i)];
      dm->data_->r_hand_rf_[i] = data_out.contact_forces["right_rubber_hand_contact_" + std::to_string(i)];

      // predicted frame positions
      dm->data_->predicted_torso_pos_[i] = data_out.predicted_frame_positions["torso_primitive_shape_" + std::to_string(i)];
      dm->data_->predicted_left_ankle_roll_pos_[i] = data_out.predicted_frame_positions["left_ankle_roll_link_" + std::to_string(i)];
      dm->data_->predicted_right_ankle_roll_pos_[i] = data_out.predicted_frame_positions["right_ankle_roll_link_" + std::to_string(i)];
      dm->data_->predicted_left_knee_pos_[i] = data_out.predicted_frame_positions["left_knee_link_" + std::to_string(i)];
      dm->data_->predicted_right_knee_pos_[i] = data_out.predicted_frame_positions["right_knee_link_" + std::to_string(i)];
      dm->data_->predicted_left_rubber_pos_[i] = data_out.predicted_frame_positions["left_rubber_hand_" + std::to_string(i)];
      dm->data_->predicted_right_rubber_pos_[i] = data_out.predicted_frame_positions["right_rubber_hand_" + std::to_string(i)];
      
      // fddp optimal state
      dm->data_->joint_vel_des_traj_[i] = xs_out[i].tail(robot_->NumActiveDof()); // qdot_joints
    }

    //NOTE: Joint pos, vel and torque are updated in _SaveData, data is updated using UpdateDesired

    dm->data_->b_fddp_feasible_ = data_out.b_fddp_feasible;
    dm->data_->solve_duration_ = data_out.solve_duration;
    dm->data_->b_trq_limit_ = sp_->b_torque_limit_;

  #endif

  data_out.xReg_costs.clear();
  data_out.uReg_costs.clear();
  data_out.xBound_costs.clear();
  data_out.com_costs.clear();
  data_out.frame_costs.clear();
  data_out.frame_des_pos.clear();
  data_out.frame_des_ori.clear();
  data_out.frame_ref_pos.clear();
  data_out.frame_ref_ori.clear();
  data_out.left_hand_frame_costs.clear();
  data_out.right_hand_frame_costs.clear();
  data_out.left_ankle_frame_costs.clear();
  data_out.right_ankle_frame_costs.clear();
  data_out.left_knee_frame_costs.clear();
  data_out.right_knee_frame_costs.clear();
  data_out.torso_link_frame_costs.clear();
  data_out.left_hand_contact_costs.clear();
  data_out.right_hand_contact_costs.clear();
  data_out.left_foot_contact_costs.clear();
  data_out.right_foot_contact_costs.clear();
  data_out.contact_forces.clear();
  data_out.predicted_frame_positions.clear();

}

void TrackPlan::Compute() {
  double controller_time = sp_->current_time_ - state_machine_start_time_;
  Eigen::VectorXd x0;
  x0.resize(g1_mpc_->getX0Size());

  std::vector<Eigen::VectorXd> xs_out(g1_mpc_->getNhorizon(), x0);
  std::vector<Eigen::VectorXd> us_out;

  Eigen::Vector3d com_ref;
  com_ref = robot_->GetRobotComPos();

  mpc_utils::MPCData data_out;

  while (run_threads_) {

      double controller_time = sp_->current_time_ - state_machine_start_time_;

      std::vector<std::unordered_map<std::string, pinocchio::SE3>> desired_frames_vec;
      std::vector<Eigen::Vector3d> desired_com_vec;
      desired_frames_vec.resize(g1_mpc_->getNhorizon() + 1);
      desired_com_vec.resize(g1_mpc_->getNhorizon() + 1);

      for (int i = 0; i < g1_mpc_->getNhorizon() + 1; i++) {
        std::unordered_map<std::string, pinocchio::SE3> desired_frames;
        const double t = controller_time + i * g1_mpc_->getDt();

        for (const auto& frame_name : g1_mpc_->getTargetFrameNames()) {
          pinocchio::SE3 temp_pose;
          temp_pose.setIdentity();
          temp_pose.translation() = bezier_curves_mgr_->getCurrentDesiredPosition(frame_name, t);
          desired_frames[frame_name] = temp_pose;
        }
        desired_frames_vec[i] = desired_frames;

        desired_com_vec[i] = pkl_utils::get_com_des_pos(com_des_, t, g1_mpc_->getDt());
      }

      xs_out[0] << robot_->GetQ(), robot_->GetQdot();
      g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, controller_time);
      
      #if B_USE_ZMQ
      G1DataManager *dm = G1DataManager::GetDataManager();
      dm->data_->total_iterations_ = data_out.total_iterations;
      
      dm->data_->xReg_costs_.resize(data_out.xReg_costs.size());
      dm->data_->xReg_costs_ = data_out.xReg_costs;
      
      dm->data_->uReg_costs_.resize(data_out.uReg_costs.size());
      dm->data_->uReg_costs_ = data_out.uReg_costs;
      
      dm->data_->xBound_costs_.resize(data_out.xBound_costs.size());
      dm->data_->xBound_costs_ = data_out.xBound_costs;

      dm->data_->com_costs_.resize(data_out.com_costs.size());
      dm->data_->com_costs_ = data_out.com_costs;

      dm->data_->torso_des_pos_ = bezier_curves_mgr_->getCurrentDesiredPosition("torso_primitive_shape", controller_time);
      dm->data_->right_knee_des_pos_ = bezier_curves_mgr_->getCurrentDesiredPosition("right_knee_link", controller_time);
      dm->data_->left_knee_des_pos_ = bezier_curves_mgr_->getCurrentDesiredPosition("left_knee_link", controller_time);
      dm->data_->right_ankle_roll_des_pos_ = bezier_curves_mgr_->getCurrentDesiredPosition("right_ankle_roll_link", controller_time);
      dm->data_->left_ankle_roll_des_pos_ = bezier_curves_mgr_->getCurrentDesiredPosition("left_ankle_roll_link", controller_time);
      dm->data_->left_rubber_hand_des_pos_ = bezier_curves_mgr_->getCurrentDesiredPosition("left_rubber_hand", controller_time);
      dm->data_->right_rubber_hand_des_pos_ = bezier_curves_mgr_->getCurrentDesiredPosition("right_rubber_hand", controller_time);
      dm->data_->com_des_pos_ = pkl_utils::get_com_des_pos(com_des_, controller_time, g1_mpc_->getDt());

      dm->data_->torso_curr_pos_ = data_out.frame_current_pos["torso_primitive_shape"];
      dm->data_->left_ankle_roll_curr_pos_ = data_out.frame_current_pos["left_ankle_roll_link"];
      dm->data_->right_ankle_roll_curr_pos_ = data_out.frame_current_pos["right_ankle_roll_link"];
      dm->data_->left_knee_curr_pos_ = data_out.frame_current_pos["left_knee_link"];
      dm->data_->right_knee_curr_pos_ = data_out.frame_current_pos["right_knee_link"];
      dm->data_->left_rubber_hand_curr_pos_ = data_out.frame_current_pos["left_rubber_hand"];
      dm->data_->right_rubber_hand_curr_pos_ = data_out.frame_current_pos["right_rubber_hand"];
      dm->data_->com_curr_pos_ = data_out.com_curr_pos;

      dm->data_->left_hand_frame_costs_.resize(data_out.left_hand_frame_costs.size());
      dm->data_->right_hand_frame_costs_.resize(data_out.right_hand_frame_costs.size());
      dm->data_->left_hand_frame_costs_ = data_out.left_hand_frame_costs;
      dm->data_->right_hand_frame_costs_ = data_out.right_hand_frame_costs;

      dm->data_->left_ankle_frame_costs_.resize(data_out.left_ankle_frame_costs.size());
      dm->data_->right_ankle_frame_costs_.resize(data_out.right_ankle_frame_costs.size());
      dm->data_->left_ankle_frame_costs_ = data_out.left_ankle_frame_costs;
      dm->data_->right_ankle_frame_costs_ = data_out.right_ankle_frame_costs;

      dm->data_->left_knee_frame_costs_.resize(data_out.left_knee_frame_costs.size());
      dm->data_->right_knee_frame_costs_.resize(data_out.right_knee_frame_costs.size());
      dm->data_->left_knee_frame_costs_ = data_out.left_knee_frame_costs;
      dm->data_->right_knee_frame_costs_ = data_out.right_knee_frame_costs;

      dm->data_->torso_link_frame_costs_.resize(data_out.torso_link_frame_costs.size());
      dm->data_->torso_link_frame_costs_ = data_out.torso_link_frame_costs;

      dm->data_->left_hand_contact_costs_.resize(data_out.left_hand_contact_costs.size());
      dm->data_->right_hand_contact_costs_.resize(data_out.right_hand_contact_costs.size());
      dm->data_->left_hand_contact_costs_ = data_out.left_hand_contact_costs;
      dm->data_->right_hand_contact_costs_ = data_out.right_hand_contact_costs;

      dm->data_->left_foot_contact_costs_.resize(data_out.left_foot_contact_costs.size());
      dm->data_->right_foot_contact_costs_.resize(data_out.right_foot_contact_costs.size());
      dm->data_->left_foot_contact_costs_ = data_out.left_foot_contact_costs;
      dm->data_->right_foot_contact_costs_ = data_out.right_foot_contact_costs;

      dm->data_->l_foot_rf_.resize(g1_mpc_->getNhorizon());
      dm->data_->r_foot_rf_.resize(g1_mpc_->getNhorizon());
      dm->data_->l_hand_rf_.resize(g1_mpc_->getNhorizon());
      dm->data_->r_hand_rf_.resize(g1_mpc_->getNhorizon());

      dm->data_->predicted_torso_pos_.resize(g1_mpc_->getNhorizon());
      dm->data_->predicted_left_ankle_roll_pos_.resize(g1_mpc_->getNhorizon());
      dm->data_->predicted_right_ankle_roll_pos_.resize(g1_mpc_->getNhorizon());
      dm->data_->predicted_left_knee_pos_.resize(g1_mpc_->getNhorizon());
      dm->data_->predicted_right_knee_pos_.resize(g1_mpc_->getNhorizon());
      dm->data_->predicted_left_rubber_pos_.resize(g1_mpc_->getNhorizon());
      dm->data_->predicted_right_rubber_pos_.resize(g1_mpc_->getNhorizon());

      dm->data_->joint_vel_des_traj_.resize(g1_mpc_->getNhorizon());

      for (int i = 0; i < g1_mpc_->getNhorizon(); ++i) {
        // reaction forces
        dm->data_->l_foot_rf_[i] = data_out.contact_forces["l_foot_contact_contact_" + std::to_string(i)];
        dm->data_->r_foot_rf_[i] = data_out.contact_forces["r_foot_contact_contact_" + std::to_string(i)];
        dm->data_->l_hand_rf_[i] = data_out.contact_forces["left_rubber_hand_contact_" + std::to_string(i)];
        dm->data_->r_hand_rf_[i] = data_out.contact_forces["right_rubber_hand_contact_" + std::to_string(i)];

        // predicted frame positions
        dm->data_->predicted_torso_pos_[i] = data_out.predicted_frame_positions["torso_primitive_shape_" + std::to_string(i)];
        dm->data_->predicted_left_ankle_roll_pos_[i] = data_out.predicted_frame_positions["left_ankle_roll_link_" + std::to_string(i)];
        dm->data_->predicted_right_ankle_roll_pos_[i] = data_out.predicted_frame_positions["right_ankle_roll_link_" + std::to_string(i)];
        dm->data_->predicted_left_knee_pos_[i] = data_out.predicted_frame_positions["left_knee_link_" + std::to_string(i)];
        dm->data_->predicted_right_knee_pos_[i] = data_out.predicted_frame_positions["right_knee_link_" + std::to_string(i)];
        dm->data_->predicted_left_rubber_pos_[i] = data_out.predicted_frame_positions["left_rubber_hand_" + std::to_string(i)];
        dm->data_->predicted_right_rubber_pos_[i] = data_out.predicted_frame_positions["right_rubber_hand_" + std::to_string(i)];

        // fddp optimal state
        dm->data_->joint_vel_des_traj_[i] = xs_out[i].tail(robot_->NumActiveDof()); // qdot_joints
      }

      dm->data_->b_fddp_feasible_ = data_out.b_fddp_feasible;
      dm->data_->solve_duration_ = data_out.solve_duration;

    #endif

    data_out.xReg_costs.clear();
    data_out.uReg_costs.clear();
    data_out.xBound_costs.clear();
    data_out.com_costs.clear();
    data_out.frame_costs.clear();
    data_out.frame_des_pos.clear();
    data_out.frame_des_ori.clear();
    data_out.frame_ref_pos.clear();
    data_out.frame_ref_ori.clear();
    data_out.left_hand_frame_costs.clear();
    data_out.right_hand_frame_costs.clear();
    data_out.left_ankle_frame_costs.clear();
    data_out.right_ankle_frame_costs.clear();
    data_out.left_knee_frame_costs.clear();
    data_out.right_knee_frame_costs.clear();
    data_out.torso_link_frame_costs.clear();
    data_out.left_hand_contact_costs.clear();
    data_out.right_hand_contact_costs.clear();
    data_out.left_foot_contact_costs.clear();
    data_out.right_foot_contact_costs.clear();
    data_out.contact_forces.clear();
    data_out.predicted_frame_positions.clear();

      {
        std::lock_guard<std::mutex> lock(data_mutex_);
        mpc_q_ = xs_out[0].head(robot_->GetQ().size()).tail(robot_->NumActiveDof());  // q_joints
        mpc_q_dot_ = xs_out[0].tail(robot_->NumActiveDof());  // qdot_joints
        mpc_tau_ = us_out[0];
        has_new_data_ = true;
      }

  }
}

void TrackPlan::LastVisit() {

  if(!b_wait_complete_) {
    run_threads_ = false;
    if (compute_thread_.joinable()) {
      compute_thread_.join();
    }
  }

}

bool TrackPlan::EndOfState() {
  return false; //TODO: define conditions to end state
}

StateId TrackPlan::GetNextState() {
  return g1_states::kDoubleSupportBalance;
}

void TrackPlan::SetParameters(const YAML::Node &node) {
}
