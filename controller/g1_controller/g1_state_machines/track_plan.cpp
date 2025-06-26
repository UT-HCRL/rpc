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

void logToFile(std::ofstream& f_duration, std::ofstream& f_iterations, std::ofstream& f_costs, const std::string &data, int count);

TrackPlan::TrackPlan(const StateId state_id,
                                           PinocchioRobotSystem *robot,
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

  std::string file_path = THIS_COM "data_example/g1_step_over_knee_knocker_latest_fix.pkl";
  pkl_reader_ = std::make_unique<pkl_utils::PickleReader>(file_path, pkl_utils::PickleType::COMPOSITE);

  if (!pkl_reader_->isReady()) {
  std::cerr << "Failed to open the file." << std::endl;
  }

  pkl_reader_->parse();

  std::vector<pkl_utils::CompositeBezierCurve> bezier_curves = pkl_reader_->getCompositeBezierCurves();
  std::vector<std::shared_ptr<pkl_utils::CompositeBezierCurve>> bezier_curves_ptrs;
  // Convert to pointers
  for (const auto& curve : bezier_curves) {
    bezier_curves_ptrs.push_back(std::make_shared<pkl_utils::CompositeBezierCurve>(curve));
  }

  std::vector<std::string> target_names = g1_mpc_->getTargetFrameNames();
  bezier_curves_mgr_ = std::make_unique<pkl_utils::BezierCurvesManager>(bezier_curves_ptrs, target_names);
  com_des_ =  pkl_reader_->getCoM();

  // for(const auto& com : com_des_) {
  //   std::cout << "CoM : " << com << std::endl;
  // }

  try {
    pkl_reader_.reset(); //NOTE: I need this otherwise on ctrl+c I get sigfault due to pybind scope 
  } catch (const std::exception& e) {
    std::cerr << "Exception caught while resetting pkl_reader_: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Unknown exception caught while resetting pkl_reader_" << std::endl;
  }

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

    new_q = mpc_q_;
    new_q_dot = mpc_q_dot_;
    new_tau = mpc_tau_;
    ctrl_arch_->tci_container_->robot_commands_->UpdateDesired(
        new_q,
        new_q_dot,
        new_tau
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
          // std::cout << "[DEBUG] New data available for OneStep." << std::endl;
          new_q = mpc_q_;
          new_q_dot = mpc_q_dot_;
          new_tau = mpc_tau_;
          has_new_data_ = false;

          ctrl_arch_->tci_container_->robot_commands_->UpdateDesired(
            new_q,
            new_q_dot,
            new_tau
          );
        } else {
          // std::cout << "[DEBUG] No new data available for OneStep." << std::endl;
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

  static Eigen::VectorXd x0;
  x0.resize(g1_mpc_->getX0Size());

  static std::vector<Eigen::VectorXd> xs_out(g1_mpc_->getNhorizon(), x0);
  static std::vector<Eigen::VectorXd> us_out;

  static Eigen::Vector3d com_ref;
  com_ref = robot_->GetRobotComPos();

  static bool remove_contact = false;

  static mpc_utils::MPCData data_out;

  double controller_time = sp_->current_time_ - state_machine_start_time_;
    // std::cout << "controller time: " <<controller_time <<std::endl;

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

      // std::cout << "Desired CoM at time " << t << ": " <<pkl_utils::get_com_des_pos(com_des_, t, g1_mpc_->getDt()) << std::endl;

      desired_com_vec[i] = pkl_utils::get_com_des_pos(com_des_, t, g1_mpc_->getDt());

    }

    xs_out[0] << robot_->GetQ(), robot_->GetQdot();

    auto start_time = std::chrono::high_resolution_clock::now();
    if(controller_time >= 3.0 && !remove_contact){ //FIXME: this should become a switching condition from the forces
      remove_contact = true;
      g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, true);
    }else if(controller_time >= 6.0 && remove_contact){
      std::cout<<"SHOULD BE NEW STEP NOW \n";
      exit(23);
      g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, false);
    }else{
      g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, false);
    }
   
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

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
      }

      // save MPC joint positions, velocities, and torques
      for (unsigned int i = 0; i < mpc_q_.size(); i++) {
        dm->data_->joint_pos_des[i] = mpc_q_[i];
        dm->data_->joint_pos_des[i] = mpc_q_dot_[i];
        dm->data_->joint_pos_des[i] = mpc_tau_[i];
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

}

void TrackPlan::Compute() {
  double controller_time = sp_->current_time_ - state_machine_start_time_;
  Eigen::VectorXd x0;
  x0.resize(g1_mpc_->getX0Size());

  std::vector<Eigen::VectorXd> xs_out(g1_mpc_->getNhorizon(), x0);
  std::vector<Eigen::VectorXd> us_out;

  Eigen::Vector3d com_ref;
  com_ref = robot_->GetRobotComPos();

  bool remove_contact = false;

  mpc_utils::MPCData data_out;

  #ifndef B_USE_ZMQ
    std::ofstream log_file("solve_timing_log.txt", std::ios::app);
    std::ofstream log_file2("solve_iteration_log.txt", std::ios::app);
    std::ofstream log_file3("data_out_log.txt", std::ios::app);
    int count = 0;
  #endif

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

      auto start_time = std::chrono::high_resolution_clock::now();
      if (controller_time >= 3.0 && !remove_contact) {
        remove_contact = true;
        g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, true);
      } else if (controller_time >= 6.0 && remove_contact) {
        std::cout << "SHOULD BE NEW STEP NOW \n";
        exit(23);
        g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, false);
      } else {
        g1_mpc_->solveOneStep(xs_out, us_out, data_out, desired_com_vec, desired_frames_vec, false);
      }

      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
      
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

      #ifndef B_USE_ZMQ
        logToFile(log_file, log_file2, log_file3, data_out, std::to_string(duration), count);
        count++;
      #endif

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

void logToFile(std::ofstream& f_duration, std::ofstream& f_iterations, std::ofstream& f_costs, const mpc_utils::MPCData &data_out, const std::string& duration, int count) {

    if (f_duration.is_open() && count < 10000) {
      f_duration << duration << std::endl;
      f_iterations << data_out.total_iterations << std::endl;
      count++;
    } else {
      f_duration.close();
      f_iterations.close();
      std::cerr << "File closed." << std::endl;
    }

    if(f_costs.is_open() && count< 1000) {
      for(int i=0; i<data_out.xReg_costs.size(); i++){
        f_costs << data_out.xReg_costs[i] << " ";
      }

      f_costs << std::endl;

      for(int i=0; i<data_out.uReg_costs.size(); i++){
        f_costs << data_out.uReg_costs[i] << " ";
      }

      f_costs << std::endl;

      for(int i=0; i<data_out.xBound_costs.size(); i++){
        f_costs << data_out.xBound_costs[i] << " ";
      }

      f_costs << std::endl;

      for(int i=0; i<data_out.com_costs.size(); i++){
        f_costs << data_out.com_costs[i] << " ";
      }

      f_costs << std::endl;

      for (const auto& frame : data_out.frame_costs) {
        f_costs << frame.first << ": ";
        for (const auto& cost : frame.second) {
          f_costs << cost << " ";
        }
        f_costs << std::endl;
      }
      f_costs << "----------------------------------------" << std::endl;
      
      count++;
    }else{
      f_costs.close();
      std::cerr << "File closed." << std::endl;
      // exit(23);
    }

}