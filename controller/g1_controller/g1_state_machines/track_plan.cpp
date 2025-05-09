#include "controller/g1_controller/g1_state_machines/track_plan.hpp"
#include "controller/g1_controller/g1_control_architecture.hpp"
#include "controller/g1_controller/g1_definition.hpp"
#include "controller/g1_controller/g1_state_provider.hpp"
#include "controller/robot_system/pinocchio_robot_system.hpp"
#include "controller/g1_controller/g1_tci_container.hpp"
#include "util/util.hpp"
#include "fstream"

namespace {
  double kGravity = 9.81;
}

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

  // std::string file_path = THIS_COM "data_example/g1_test.pkl";
  // pkl_reader_ = std::make_unique<pkl_utils::PickleReader>(file_path, pkl_utils::PickleType::BEZIER);

  // if (!pkl_reader_->isReady()) {
  //     std::cerr << "Failed to open the file." << std::endl;
  // }

  // pkl_reader_->parse();
  // bezier_curves_ = pkl_reader_->getCompositeBezierCurves();
  // pkl_reader_.reset();  //NOTE: I need this otherwise on ctrl+c I get sigfault due to pybind scope
  // Uncomment to test bezier curve read
  // const auto selected_bezier = bezier_curves_[0];
        
  // std::cout << "[MAIN] - Points of selected bezier curve:" << std::endl;
  // for (const auto& bezier : selected_bezier.getBeziers()) {
  //     const auto& points = bezier.getPoints();
  //     for (const auto& point : points) {
  //         std::cout << point.transpose() << std::endl;
  //     }
  // }

}

TrackPlan::~TrackPlan() {
  delete sp_;
  delete ctrl_arch_;
}

void TrackPlan::FirstVisit() {
  std::cout << "g1_states::kTrackPlan" << std::endl;
  state_machine_start_time_ = sp_->current_time_;

  run_threads_ = true;
  compute_thread_ = std::thread(&TrackPlan::Compute, this);

}

void TrackPlan::OneStep() {

  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (has_new_data_) {

      new_q = mpc_q_;
      new_q_dot = mpc_q_dot_;
      new_tau = mpc_tau_;
      has_new_data_ = false;
      ctrl_arch_->tci_container_->robot_commands_->UpdateDesired(new_q.tail(g1_mpc_->getQ0Size()), new_q_dot.tail(g1_mpc_->getQ0Size()), new_tau);  //FIXME: remove hardcoded magic numbers
      
    }
  }
}

void TrackPlan::Compute() {

  Eigen::VectorXd x0;
  x0.resize(g1_mpc_->getX0Size());

  std::vector<Eigen::VectorXd> xs_out(g1_mpc_->getNhorizon(), x0);
  std::vector<Eigen::VectorXd> us_out;

  Eigen::Vector3d com_ref;
  com_ref = robot_->GetRobotComPos();

  std::unordered_map<std::string, pinocchio::SE3> desired_frames;
  desired_frames.reserve(1);
  Eigen::Isometry3d torso = robot_->GetLinkIsometry("torso_link");
  Eigen::Vector3d torso_pos = torso.translation();
  Eigen::Vector3d torso_rot = torso.rotation().eulerAngles(0, 1, 2);
  pinocchio::SE3 current_pose = pinocchio::SE3(Eigen::AngleAxisd(torso_rot[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(torso_rot[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(torso_rot[2], Eigen::Vector3d::UnitZ()), torso_pos);
  desired_frames["torso_link"] = current_pose;

  mpc_utils::MPCData data_out;

  std::ofstream log_file("solve_timing_log.txt", std::ios::app);
  if (!log_file.is_open()) {
    std::cerr << "Failed to open log file for writing." << std::endl;
  }else{
    std::cout<<"Start writing computation time [ms] to file."<<std::endl;
  }

  std::ofstream log_file2("solve_iteration_log.txt", std::ios::app);
  if (!log_file2.is_open()) {
    std::cerr << "Failed to open log file for writing." << std::endl;
  }else{
    std::cout<<"Start writing iteration number to file."<<std::endl;
  }

  std::ofstream log_file3("data_out_log.txt", std::ios::app);
  if (!log_file3.is_open()) {
    std::cerr << "Failed to open log file for writing." << std::endl;
  }else{
    std::cout<<"Start writing data_out to file."<<std::endl;
  }

  int count = 0;

  while (run_threads_) {

    xs_out[0] << robot_->GetQ(), robot_->GetQdot();

    auto start_time = std::chrono::high_resolution_clock::now();
    g1_mpc_->solveOneStep(xs_out, us_out, data_out, com_ref, desired_frames);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // if (log_file.is_open() && count < 10000) {
    //   log_file << duration << std::endl;
    //   log_file2 << data_out.total_iterations << std::endl;
    //   count++;
    // } else {
    //   log_file.close();
    //   log_file2.close();
    //   std::cerr << "File closed." << std::endl;
    // }

    if(log_file3.is_open() && count< 1000) {
      for(int i=0; i<data_out.xReg_costs.size(); i++){
        log_file3 << data_out.xReg_costs[i] << " ";
      }

      log_file3 << std::endl;

      for(int i=0; i<data_out.uReg_costs.size(); i++){
        log_file3 << data_out.uReg_costs[i] << " ";
      }

      log_file3 << std::endl;

      for(int i=0; i<data_out.xBound_costs.size(); i++){
        log_file3 << data_out.xBound_costs[i] << " ";
      }

      log_file3 << std::endl;

      for(int i=0; i<data_out.com_costs.size(); i++){
        log_file3 << data_out.com_costs[i] << " ";
      }

      log_file3 << std::endl;

      for (const auto& frame : data_out.frame_costs) {
        log_file3 << frame.first << ": ";
        for (const auto& cost : frame.second) {
          log_file3 << cost << " ";
        }
        log_file3 << std::endl;
      }

      for (const auto& contact : data_out.contact_costs) {
        log_file3 << contact.first << ": ";
        for (const auto& cost : contact.second) {
          log_file3 << cost << " ";
        }
        log_file3 << std::endl;
      }
      log_file3 << "----------------------------------------" << std::endl;
      
      data_out.xReg_costs.clear();
      data_out.uReg_costs.clear();
      data_out.xBound_costs.clear();
      data_out.com_costs.clear();
      data_out.frame_costs.clear();
      data_out.contact_costs.clear();
      count++;
    }else{
      log_file3.close();
      std::cerr << "File closed." << std::endl;
      exit(23);
    }

    // std::cout<<"\n\n\nxReg_costs: "<<data_out.xReg_costs.back()<<std::endl;
    // std::cout<<"uReg_costs: "<<data_out.uReg_costs.back()<<std::endl;
    // std::cout<<"xBound_costs: "<<data_out.xBound_costs.back()<<std::endl;
    // std::cout<<"com_costs: "<<data_out.com_costs.back()<<std::endl;
    // std::cout<<"solve time: "<<duration<<std::endl;

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      mpc_q_ = xs_out[0].head(xs_out[0].size() / 2);
      mpc_q_dot_ = xs_out[0].tail(xs_out[0].size() / 2);
      mpc_tau_ = us_out[0];
      has_new_data_ = true;
    }
  }
}

void TrackPlan::LastVisit() {

  run_threads_ = false;
  if (compute_thread_.joinable()) {
    compute_thread_.join();
  }

}

bool TrackPlan::EndOfState() {
  return false; //TODO: define conditions to end state
}

StateId TrackPlan::GetNextState() {
  return g1_states::kDoubleSupportBalance;
}

void TrackPlan::SetParameters(const YAML::Node &node) {
  std::cerr << "TrackPlan::SetParameters not implemented" << std::endl;
}
