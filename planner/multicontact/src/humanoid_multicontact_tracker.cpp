#include "humanoid_multicontact_tracker.hpp"

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
#include "crocoddyl/core/integrator/rk.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/residuals/control.hpp"
#include "crocoddyl/core/utils/timer.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/core/activations/quadratic.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"

#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/contacts/contact-3d.hpp"
#include "crocoddyl/multibody/contacts/contact-6d.hpp"

#include "crocoddyl/multibody/states/multibody.hpp"

#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/residuals/contact-friction-cone.hpp"
#include "crocoddyl/multibody/residuals/contact-wrench-cone.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/core/utils/callbacks.hpp"

#include <pinocchio/algorithm/model.hpp>

#include <omp.h>

#include "util/pkl_utils.hpp"

class CostRecorderCallback : public crocoddyl::CallbackAbstract {
 public:
  explicit CostRecorderCallback() = default;
  virtual ~CostRecorderCallback() = default;

  // The operator() is called by the solver at each iteration
  void operator()(crocoddyl::SolverAbstract& solver) override {
    // Store the iteration number and cost in the map
    iteration_costs_[solver.get_iter()].push_back(solver.get_cost());

  }

  const std::map<int, std::vector<double>>& get_costs() const {
    return iteration_costs_;
  }

 private:
  std::map<int, std::vector<double>> iteration_costs_;
};

HumanoidMulticontactTracker::HumanoidMulticontactTracker(const std::string& robot_path, const std::unordered_map<std::string, mpc_utils::Weights>& cost_weights, const std::vector<int>& locked_joints_list, const bool croc_callbacks) : locked_joints_list_(locked_joints_list), enable_callbacks_(croc_callbacks) {

    //### Default problem formulation parameters ### TODO: remove them if we stick with config params
    dt_ = 0.02;
    N_horizon_ = 2;
    max_iter_ = 100;
    //###############################################

    config_path_ = THIS_COM "config/g1/sim/mujoco/ihwbc/crocoddyl_params.yaml";
    params_ = YAML::LoadFile(config_path_);
    loadMPCParams();

    frame_residuals_.resize(N_horizon_ + 1);
    force_residuals_.resize(N_horizon_ + 1);
    control_residuals_.resize(N_horizon_);

    pinocchio::urdf::buildModel(robot_path, pinocchio::JointModelFreeFlyer(), model_full_);
    pinocchio_data_ = std::make_unique<pinocchio::Data>(model_full_);

    // if(locked_joints_list_.empty()) {
    reduced_model_ = false;
    std::shared_ptr<crocoddyl::StateMultibody> state = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_full_));
    state_ = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_full_));
    // }

    // else {
    //     reduced_model_ = true;
    //     locked_joints_.reserve(locked_joints_list_.size());
    //     for (const auto& joint : locked_joints_list_) {
    //         locked_joints_.push_back(joint);
    //     }
        
    //     pinocchio::buildReducedModel(model_full_, locked_joints_, Eigen::VectorXd::Zero(model_full_.nq), model_);
    //     state_ = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_));
    // }

    actuation_ = std::make_shared<crocoddyl::ActuationModelFloatingBase>(state_);
    for(int i = 0; i < N_horizon_; i++){
        running_cost_model_.push_back(std::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu()));
    }
    terminal_cost_model_ = std::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());

    for(int i=0; i < N_horizon_; i++){
        running_contact_models_.push_back(std::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu()));
    }
    terminal_contact_models_ = std::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());

    //### Default class member init ###
    util::ReadParameter(params_["friction"], "w", friction_weight_);
    util::ReadParameter(params_["friction"], "mu", mu_);

    RH_rotation_ = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    LH_rotation_ = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    x0_ = Eigen::VectorXd::Zero(state_->get_nx());
    x_prev_= std::vector<Eigen::VectorXd>(N_horizon_+1, Eigen::VectorXd::Zero(state_->get_nx()));
    u_prev_= std::vector<Eigen::VectorXd>(N_horizon_, Eigen::VectorXd::Zero(state_->get_nv()-6));
    //#################################

    //### Configure switching problem ###
    contact_switching_manager_ = std::make_shared<ContactSwitchingManager>(std::unordered_map<std::string, bool>{
                {"l_foot_contact", true},
                {"r_foot_contact", true},
                {"left_rubber_hand", false},
                {"right_rubber_hand", false}
            },
            N_horizon_);
    
    detector_ = std::make_shared<TransitionDetector>();
    coordinator_ = std::make_unique<ContactTransitionCoordinator>(contact_switching_manager_, detector_);
    auto time_condition = std::make_shared<TimeElapsedCondition>("time_elapsed", 3.0, true);
    // auto plane_condition = std::make_shared<PlanePassCondition>(
    //     "plane_pass", 
    //     Eigen::Isometry3d(Eigen::Translation3d(0.0, 0.3, 0.0)),
    //     Eigen::Vector3d(0.0, 1.0, 0.0), 
    //     "left_rubber_hand"
    // );
    detector_->addCondition(time_condition);
    // detector_->addCondition(plane_condition);
    contact_switching_manager_->resetSwitchingMask();
    ctx_.time = 0.0;
    ctx_.pose["l_foot_contact"] = mpc_utils::SE3_to_Isometry(pinocchio_data_->oMf[model_full_.getFrameId("l_foot_contact")]);
    ctx_.pose["r_foot_contact"] = mpc_utils::SE3_to_Isometry(pinocchio_data_->oMf[model_full_.getFrameId("r_foot_contact")]);
    ctx_.pose["left_rubber_hand"] = mpc_utils::SE3_to_Isometry(pinocchio_data_->oMf[model_full_.getFrameId("left_rubber_hand")]);
    ctx_.pose["right_rubber_hand"] = mpc_utils::SE3_to_Isometry(pinocchio_data_->oMf[model_full_.getFrameId("right_rubber_hand")]);

    contact_mask_.resize(N_horizon_ + 1);
    //#####################################

    loadInitialConfiguration();
    loadCostMask();
    loadContactFrames();
    loadRegularizationWeights();
    loadBoundWeights();
    loadCoMWeights();
    loadTrackingFramesWeights();
    loadForceTrackingWeights();
    loadTorqueRateRegWeights();

    initializeSolver();
}

void HumanoidMulticontactTracker::loadCostMask(){
    util::ReadParameter(params_, "cost_mask", cost_mask_);
}

void HumanoidMulticontactTracker::loadInitialConfiguration() {
    
    std::vector<std::string> joint_names;
    util::ReadParameter(params_, "joint_names", joint_names);

    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(joint_names.size());
    for (int i = 0; i < joint_names.size(); i++) {
        double joint_value = 0.0;
        util::ReadParameter(params_["initial_config"], joint_names[i], joint_value);
        q0[i] = joint_value;
    }
    setInitialJointConfiguration(q0);

}

void HumanoidMulticontactTracker::loadContactFrames(){

    std::vector<std::string> contact_frames;
    util::ReadParameter(params_, "contact_frames", contact_frames);

    setFrames(contact_frames);

    for(const auto& frame_name : contact_frames){
        double w_tang = 0;
        double w_norm = 0;

        util::ReadParameter(params_["running_costs"]["contact_frames"][frame_name], "wt", w_tang);
        util::ReadParameter(params_["running_costs"]["contact_frames"][frame_name], "wn", w_norm);
        terminal_contact_weights_[frame_name] = mpc_utils::from2DValues(w_tang, w_norm);

        mpc_utils::normalize_weights(w_tang, N_horizon_);
        mpc_utils::normalize_weights(w_norm, N_horizon_);
        contact_weights_[frame_name] = mpc_utils::from2DValues(w_tang, w_norm);

    }
}

void HumanoidMulticontactTracker::loadRegularizationWeights() {
    std::vector<double> base_pos_weights;
    std::vector<double> base_rot_weights;
    double joint_pos_weights;
    double joint_vel_weights;
    std::vector<double> base_lin_vel_weights;
    std::vector<double> base_ang_vel_weights;

    int nv = state_->get_nv();

    util::ReadParameter(params_["running_costs"]["xReg"], "w", xreg_weight_);
    mpc_utils::normalize_weights(xreg_weight_, N_horizon_);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "base_pos", base_pos_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "base_rot", base_rot_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "joint_pos", joint_pos_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "joint_vel", joint_vel_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "base_lin_vel", base_lin_vel_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "base_ang_vel", base_ang_vel_weights);

    xreg_weights_ = Eigen::VectorXd::Zero(2 * state_->get_nv());
    for (int i = 0; i < 3; ++i) xreg_weights_(i) = pow(base_pos_weights[i],2);
    for (int i = 0; i < 3; ++i) xreg_weights_(3 + i) = pow(base_rot_weights[i],2);
    for (int i = 6; i < nv; ++i) xreg_weights_(i) = pow(joint_pos_weights,2);
    for (int i = 0; i < 3; ++i) xreg_weights_(nv + i) = pow(base_lin_vel_weights[i],2);
    for (int i = 3; i < 6; ++i) xreg_weights_(nv + i) = pow(base_ang_vel_weights[i],2);
    for (int i = 6; i < nv; ++i) xreg_weights_(nv + i) = pow(joint_vel_weights,2);
    mpc_utils::normalize_weights(xreg_weights_, N_horizon_);

    util::ReadParameter(params_["running_costs"]["uReg"], "w", ureg_weight_);
    mpc_utils::normalize_weights(ureg_weight_, N_horizon_);

    util::ReadParameter(params_["terminal_costs"]["xReg"], "w", terminal_xreg_weight_);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "base_pos", base_pos_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "base_rot", base_rot_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "joint_pos", joint_pos_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "joint_vel", joint_vel_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "base_lin_vel", base_lin_vel_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "base_ang_vel", base_ang_vel_weights);

    terminal_xreg_weights_ = Eigen::VectorXd::Zero(2 * state_->get_nv());
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(i) = pow(base_pos_weights[i],2);
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(3 + i) = pow(base_rot_weights[i],2);
    for (int i = 6; i < nv; ++i) terminal_xreg_weights_(i) = pow(joint_pos_weights,2);
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(nv + i) = pow(base_lin_vel_weights[i],2);
    for (int i = 3; i < 6; ++i) terminal_xreg_weights_(nv + i) = pow(base_ang_vel_weights[i],2);
    for (int i = 6; i < nv; ++i) terminal_xreg_weights_(nv + i) = pow(joint_vel_weights,2);

    util::ReadParameter(params_["terminal_costs"]["uReg"], "w", terminal_ureg_weight_);

}

void HumanoidMulticontactTracker::loadBoundWeights(){
    util::ReadParameter(params_["running_costs"]["xBound"], "w", xbound_weight_);
    mpc_utils::normalize_weights(xbound_weight_, N_horizon_);

    util::ReadParameter(params_["terminal_costs"]["xBound"], "w", terminal_xbound_weight_);
}

void HumanoidMulticontactTracker::loadCoMWeights(){
    util::ReadParameter(params_["running_costs"]["CoM"], "w", com_tracking_weight_);
    mpc_utils::normalize_weights(com_tracking_weight_, N_horizon_);

    util::ReadParameter(params_["terminal_costs"]["CoM"], "w", terminal_com_tracking_weight_);
}

void HumanoidMulticontactTracker::loadTrackingFramesWeights(){
    util::ReadParameter(params_, "tracking_frames", track_frame_names_);
    util::ReadParameter(params_["running_costs"]["tracking_frames"], "w", frame_tracking_weight_);
    util::ReadParameter(params_["running_costs"]["tracking_frames"], "contact_rot", tracking_contact_rot_weight_);
    util::ReadParameter(params_["running_costs"]["tracking_frames"], "swing_rot", tracking_swing_rot_weight_);
    for (const auto& frame_name : track_frame_names_) {
        double w_frame;
        double wo_frame;
        util::ReadParameter(params_["running_costs"]["tracking_frames"][frame_name], "pos", w_frame);
        util::ReadParameter(params_["running_costs"]["tracking_frames"][frame_name], "rot", wo_frame);
        mpc_utils::normalize_weights(w_frame, N_horizon_);
        // initialize feet with contact weights and hands with swing weights
        if (frame_name.find("ankle") != std::string::npos) {
            wo_frame = tracking_contact_rot_weight_;
        } else {
            wo_frame = tracking_swing_rot_weight_;
        }
        frame_targets_[frame_name] = mpc_utils::from2DValues(w_frame, wo_frame);
    }
    util::ReadParameter(params_["terminal_costs"]["tracking_frames"], "w", terminal_frame_tracking_weight_);
    util::ReadParameter(params_["terminal_costs"]["tracking_frames"], "contact_rot", terminal_tracking_contact_rot_weight_);
    util::ReadParameter(params_["terminal_costs"]["tracking_frames"], "swing_rot", terminal_tracking_swing_rot_weight_);
    for (const auto& frame_name : track_frame_names_) {
        double w_frame;
        double wo_frame;
        std::string frame_name_terminal = frame_name + "_terminal";
        util::ReadParameter(params_["terminal_costs"]["tracking_frames"][frame_name], "pos", w_frame);
        util::ReadParameter(params_["terminal_costs"]["tracking_frames"][frame_name], "rot", wo_frame);
        // initialize feet with contact weights and hands with swing weights
        if (frame_name.find("ankle") != std::string::npos) {
            wo_frame = terminal_tracking_contact_rot_weight_;
        } else {
            wo_frame = terminal_tracking_swing_rot_weight_;
        }
        frame_targets_terminal_[frame_name] = mpc_utils::from2DValues(w_frame, wo_frame);
    }
}

void HumanoidMulticontactTracker::loadForceTrackingWeights(){
    
    //NOTE: this assumes that frame_names is already populated by its loading call.
    util::ReadParameter(params_["running_costs"]["tracking_forces"], "w", force_cost_weight_);
    util::ReadParameter(params_["terminal_costs"]["tracking_forces"], "w", terminal_force_cost_weight_);

    for(const auto frame : frame_names_){
        std::vector<double> force_tracking_weight;
        util::ReadParameter(params_["running_costs"]["tracking_forces"], frame, force_tracking_weight);
        if(force_tracking_weight.size() != 3) {
            throw std::runtime_error("Force tracking weight for frame " + frame + " must be a vector of size 3, got: " + std::to_string(force_tracking_weight.size()));
        }
        force_tracking_weights_[frame] = mpc_utils::fromStdVector3(force_tracking_weight);
        mpc_utils::normalize_weights(force_tracking_weights_[frame], N_horizon_);

        if(force_tracking_weight.size() != 3) {
            throw std::runtime_error("Force tracking weight for frame " + frame + " must be a vector of size 3, got: " + std::to_string(force_tracking_weight.size()));
        }
        terminal_force_tracking_weights_[frame] = mpc_utils::fromStdVector3(force_tracking_weight);
        mpc_utils::normalize_weights(terminal_force_tracking_weights_[frame], N_horizon_);
    }
}

void HumanoidMulticontactTracker::loadTorqueRateRegWeights(){
    util::ReadParameter(params_["running_costs"]["tauRateReg"], "w", dtau_reg_weight_);
    mpc_utils::normalize_weights(dtau_reg_weight_, N_horizon_);

}

void HumanoidMulticontactTracker::loadMPCParams(){
    util::ReadParameter(params_["mpc"], "dt", dt_);
    util::ReadParameter(params_["mpc"], "horizon", N_horizon_);
    util::ReadParameter(params_["mpc"], "max_iter", max_iter_);

    std::string integration_method;
    util::ReadParameter(params_["mpc"], "integration_method", integration_method);

    std::transform(integration_method.begin(), integration_method.end(), integration_method.begin(), ::toupper);

    if (integration_method == "EULER") {
        integration_method_ = mpc_utils::IntegrationMethod::Euler;
    } else if (integration_method == "RK2") {
        integration_method_ = mpc_utils::IntegrationMethod::RK2;
    } else if (integration_method == "RK3") {
        integration_method_ = mpc_utils::IntegrationMethod::RK3;
    } else if (integration_method == "RK4") {
        integration_method_ = mpc_utils::IntegrationMethod::RK4;
    } else {
        throw std::invalid_argument("Invalid integration method. Supported methods are 'Euler', 'RK2' , 'RK3', 'RK4'.");
    }
}

void HumanoidMulticontactTracker::printModel() const {
    if(reduced_model_) {
            std::cout << "###\nReduced model NQ: " << model_.nq << std::endl;
            std::cout << "Full model NQ: " << model_full_.nq << "\n###\n";
    } else {
        std::cout << "###\nFull model NQ: " << model_full_.nq << "\n###\n";
    }

    std::cout << "state_->get_nx(): " << state_->get_nx() << std::endl;
    std::cout << "state_->get_nv(): " << state_->get_nv() << std::endl;
    std::cout << "state_->get_nq(): " << state_->get_nq() << std::endl;
    std::cout << "actuation_->get_nu(): " << actuation_->get_nu() << std::endl;
    std::cout << "x0_.size(): " << x0_.size() << std::endl;

    for (std::size_t i = 0; i < model_full_.frames.size(); ++i) {
        const auto& frame = model_full_.frames[i];
        std::cout << "Frame ID: " << i << ", Name: " << frame.name << ", Parent Joint: " << frame.parent << "\n";
    }

}

void HumanoidMulticontactTracker::printWeights() const{
    std::cout << "\n### Running costs weights ###\n" << std::endl;
    std::cout << "xreg_weights_: " << xreg_weights_.transpose() << std::endl;
    std::cout << "ureg_weight_: " << ureg_weight_ << std::endl;
    std::cout << "xbound_weight_: " << xbound_weight_ << std::endl;
    std::cout << "com_tracking_weight_: " << com_tracking_weight_ << std::endl;
    std::cout << "frame_tracking_weight_: " << frame_tracking_weight_ << std::endl;

    std::cout << "\n\n### Terminal costs weights ###\n" << std::endl;
    std::cout << "terminal_xreg_weights_: " << terminal_xreg_weights_.transpose() << std::endl;
    std::cout << "terminal_ureg_weight_: " << terminal_ureg_weight_ << std::endl;
    std::cout << "terminal_xbound_weight_: " << terminal_xbound_weight_ << std::endl;
    std::cout << "terminal_com_tracking_weight_: " << terminal_com_tracking_weight_ << std::endl;
    std::cout << "terminal_frame_tracking_weight_: " << terminal_frame_tracking_weight_ << std::endl;
}

void HumanoidMulticontactTracker::setInitialJointConfiguration(const Eigen::VectorXd& q0) {
    if (!reduced_model_ && q0.size() != model_full_.nq) {
        throw std::invalid_argument("Invalid size of q0, expected " + std::to_string(model_full_.nq) + " but got " + std::to_string(q0.size()));
    }
    else if(reduced_model_ && q0.size() != model_.nq){
        throw std::invalid_argument("Invalid size of q0, expected " + std::to_string(model_.nq) + " but got " + std::to_string(q0.size()));
    }
    q0_ = q0;

    x0_ << q0_, Eigen::VectorXd::Zero(state_->get_nv());
}

void HumanoidMulticontactTracker::addCoMCost(const double com_tracking_weight = 1e4, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index) {

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;
    
    com_reference_ << pinocchio::centerOfMass(model_full_, *pinocchio_data_, x0_.head(state_->get_nq())); // get only q0_  // 0, 0, 0.8; 

    com_residual_.push_back(std::make_shared<crocoddyl::ResidualModelCoMPosition>(state_, com_reference_, actuation_->get_nu()));

    std::shared_ptr<crocoddyl::ActivationModelAbstract> com_activation = std::make_shared<crocoddyl::ActivationModelQuad>(3);
    std::shared_ptr<crocoddyl::CostModelAbstract> com_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, com_activation, com_residual_[horizon_index]);
    cost_model->addCost("CoMTracking", com_cost, com_tracking_weight);
}

void HumanoidMulticontactTracker::addFrameTrackingCost(const std::string& frame_name, const double frame_tracking_weight=1.0, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;
    mpc_utils::Weights2D frame_cost_weight = (phase == mpc_utils::Phase::Running) ? frame_targets_[frame_name] : frame_targets_terminal_[frame_name];

    Eigen::VectorXd q = x0_.head(state_->get_nq());
    pinocchio::forwardKinematics(model_full_, *pinocchio_data_, q);
    pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);

    Eigen::Vector3d torso_pos = {0.0340706, -8.68116e-05, 0.696563};
    Eigen::Vector3d torso_rot = {0, 0, 0};
    pinocchio::SE3 current_pose = pinocchio::SE3(Eigen::AngleAxisd(torso_rot[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(torso_rot[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(torso_rot[2], Eigen::Vector3d::UnitZ()), torso_pos);

    std::shared_ptr<crocoddyl::ResidualModelFramePlacement> frame_residual = std::make_shared<crocoddyl::ResidualModelFramePlacement>(state_, model_full_.getFrameId(frame_name), current_pose, actuation_->get_nu());
    frame_residuals_[horizon_index][frame_name] = frame_residual;

    Eigen::VectorXd temp_weights(6);
    temp_weights << frame_cost_weight(0), frame_cost_weight(0), frame_cost_weight(0), frame_cost_weight(1), frame_cost_weight(1), frame_cost_weight(1);
    std::shared_ptr<crocoddyl::ActivationModelAbstract> frame_activation = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(temp_weights);
    std::shared_ptr<crocoddyl::CostModelAbstract> goal_tracking_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, frame_activation, frame_residuals_[horizon_index][frame_name]);
    cost_model->addCost("frame_" + frame_name, goal_tracking_cost, frame_tracking_weight);

}

void HumanoidMulticontactTracker::addXBoundCost(const double x_bound_weight = 50000.0, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index) {

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;

    Eigen::VectorXd x_lb(state_->get_lb().segment(1, state_->get_nv()).size() + state_->get_lb().tail(state_->get_nv()).size());
    Eigen::VectorXd x_ub(state_->get_ub().segment(1, state_->get_nv()).size() + state_->get_ub().tail(state_->get_nv()).size());
    x_lb << state_->get_lb().segment(1, state_->get_nv()), state_->get_lb().tail(state_->get_nv());
    x_ub << state_->get_ub().segment(1, state_->get_nv()), state_->get_ub().tail(state_->get_nv());
    crocoddyl::ActivationBounds x_bounds(x_lb, x_ub);

    std::shared_ptr<crocoddyl::ActivationModelAbstract> x_bound_activation = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(x_bounds);
    std::shared_ptr<crocoddyl::ResidualModelAbstract> x_bound_residual = std::make_shared<crocoddyl::ResidualModelState>(state_, actuation_->get_nu());
    std::shared_ptr<crocoddyl::CostModelAbstract> x_bound_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, x_bound_activation, x_bound_residual);
    cost_model->addCost("xBounds", x_bound_cost, x_bound_weight);
}

void HumanoidMulticontactTracker::addTorqueRateCost(const double dtau_reg_weight = 1.0, const mpc_utils::Phase phase = mpc_utils::Phase::Running,const int horizon_index = 0) {
    
    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;

    const std::size_t nu = actuation_->get_nu();
    auto activation = std::make_shared<crocoddyl::ActivationModelQuad>(nu);

    control_residuals_[horizon_index] = std::make_shared<crocoddyl::ResidualModelControl>(state_, nu);
    control_residuals_[horizon_index]->set_reference(Eigen::VectorXd::Zero(actuation_->get_nu()));

    auto rate_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, activation, control_residuals_[horizon_index]);
    cost_model->addCost("tauRate", rate_cost, dtau_reg_weight);
    cost_model->changeCostStatus("tauRate", false);
}

void HumanoidMulticontactTracker::addRegularizationCosts(const Eigen::VectorXd& x_weights, const double xreg_weight = 5e-2, const double ureg_weight = 1e-4, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index) {

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;

    xreg_activation_ = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(x_weights); //NOTE: power is computed in the weights not here
    xreg_cost_ = std::make_shared<crocoddyl::CostModelResidual>(state_, xreg_activation_, std::make_shared<crocoddyl::ResidualModelState>(state_, x0_, actuation_->get_nu()));
    ureg_cost_ = std::make_shared<crocoddyl::CostModelResidual>(state_, std::make_shared<crocoddyl::ResidualModelControl>(state_, actuation_->get_nu()));
    
    cost_model->addCost("xReg", xreg_cost_, xreg_weight);
    if(phase == mpc_utils::Phase::Running) cost_model->addCost("uReg", ureg_cost_, ureg_weight);

}

void HumanoidMulticontactTracker::addContactCosts(const std::vector<std::string>& frame_names, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;
    auto& contact_model = (phase == mpc_utils::Phase::Running) ? running_contact_models_[horizon_index] : terminal_contact_models_;

    for(size_t i = 0; i < frame_names.size(); i++){

        std::string frame_name = frame_names[i];

        pinocchio::SE3 contact_frame_pose = pinocchio::SE3::Identity();

        if(frame_name.find("right_rubber") != std::string::npos){
            contact_frame_pose.rotation() = RH_rotation_;
        }else if(frame_name.find("left_rubber") != std::string::npos){
            contact_frame_pose.rotation() = LH_rotation_;
        }

        Vector3d xref = Vector3d::Zero();
        if(phase == mpc_utils::Phase::Running){
            std::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model;
            // for hands, use 3D contact model
            if (frame_name.find("right_rubber") != std::string::npos || frame_name.find("left_rubber") != std::string::npos) {
                // xref = pinocchio_data_->oMf[model_full_.getFrameId(frame_name)].translation();
                support_contact_model =
                    std::make_shared<crocoddyl::ContactModel3D>(state_, model_full_.getFrameId(frame_name), xref, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), contact_weights_[frame_name]);
            } else {
                support_contact_model =
                    std::make_shared<crocoddyl::ContactModel6D>(state_, model_full_.getFrameId(frame_name), contact_frame_pose, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), contact_weights_[frame_name]);
            }
            contact_model->addContact(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_contact", support_contact_model);
        }
        if(phase == mpc_utils::Phase::Terminal){
            std::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model;
            // for hands, use 3D contact model
            if (frame_name.find("right_rubber") != std::string::npos || frame_name.find("left_rubber") != std::string::npos) {
                // xref = pinocchio_data_->oMf[model_full_.getFrameId(frame_name)].translation();
                support_contact_model =
                    std::make_shared<crocoddyl::ContactModel3D>(state_, model_full_.getFrameId(frame_name), xref, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), terminal_contact_weights_[frame_name]);
            } else {
                support_contact_model =
                    std::make_shared<crocoddyl::ContactModel6D>(state_, model_full_.getFrameId(frame_name), contact_frame_pose, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), terminal_contact_weights_[frame_name]);
            }
            contact_model->addContact(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_contact_terminal", support_contact_model);
        }

        Eigen::Matrix3d rotation;
        if(frame_name.find("right_rubber") != std::string::npos){
            // rotation = Eigen::Matrix3d::Identity();
            rotation = RH_rotation_;
            std::string contact_suffix;
            contact_suffix = (phase == mpc_utils::Phase::Running) ? "_contact" : "_contact_terminal";
            contact_model->changeContactStatus(model_full_.frames[model_full_.getFrameId(frame_name)].name + contact_suffix, false);
            crocoddyl::FrictionCone surf_cone(rotation, mu_, 4, true);
            crocoddyl::ActivationBounds bounds(surf_cone.get_lb(), surf_cone.get_ub());
            std::shared_ptr<crocoddyl::ActivationModelAbstract> surf_activation_friction = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
            std::shared_ptr<crocoddyl::ResidualModelAbstract> surf_residual = std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(state_, model_full_.getFrameId(frame_name), surf_cone, actuation_->get_nu());
            std::shared_ptr<crocoddyl::CostModelAbstract> surf_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, surf_activation_friction, surf_residual);
            cost_model->addCost(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", surf_cost, friction_weight_);
        }
        else if(frame_name.find("left_rubber") != std::string::npos){
            // rotation = Eigen::Matrix3d::Identity();
            rotation = LH_rotation_;
            std::string contact_suffix;
            contact_suffix = (phase == mpc_utils::Phase::Running) ? "_contact" : "_contact_terminal";
            contact_model->changeContactStatus(model_full_.frames[model_full_.getFrameId(frame_name)].name + contact_suffix, false);
            crocoddyl::FrictionCone surf_cone(rotation, mu_, 4, true);
            crocoddyl::ActivationBounds bounds(surf_cone.get_lb(), surf_cone.get_ub());
            std::shared_ptr<crocoddyl::ActivationModelAbstract> surf_activation_friction = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
            std::shared_ptr<crocoddyl::ResidualModelAbstract> surf_residual = std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(state_, model_full_.getFrameId(frame_name), surf_cone, actuation_->get_nu());
            std::shared_ptr<crocoddyl::CostModelAbstract> surf_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, surf_activation_friction, surf_residual);
            cost_model->addCost(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", surf_cost, friction_weight_);
        }
        else{
            rotation = Eigen::Matrix3d::Identity();
            Vector2d foot_size(0.12, 0.05); // goes from (-0.05 to 0.12, -0.025 to 0.035)
            crocoddyl::WrenchCone surf_cone(rotation, mu_, foot_size);
            crocoddyl::ActivationBounds bounds(surf_cone.get_lb(), surf_cone.get_ub());
            std::shared_ptr<crocoddyl::ActivationModelAbstract> surf_activation_friction = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
            std::shared_ptr<crocoddyl::ResidualModelAbstract> surf_residual = std::make_shared<crocoddyl::ResidualModelContactWrenchCone>(state_, model_full_.getFrameId(frame_name), surf_cone, actuation_->get_nu());
            std::shared_ptr<crocoddyl::CostModelAbstract> surf_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, surf_activation_friction, surf_residual);
            cost_model->addCost(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", surf_cost, friction_weight_);
        }
    }
}

void HumanoidMulticontactTracker::addForceTrackingCost(const std::string& frame_name, const pinocchio::Force& force_reference, const Eigen::Vector3d& weights, const double cost_weight, const mpc_utils::Phase phase, const int horizon_index, const size_t contact_f_dim, const bool fwwddyn){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;
    std::cout << "Adding force tracking cost for frame: " << frame_name << std::endl;

    std::shared_ptr<crocoddyl::ResidualModelContactForce> force_residual = std::make_shared<crocoddyl::ResidualModelContactForce>(state_, model_full_.getFrameId(frame_name), force_reference, contact_f_dim, actuation_->get_nu(), fwwddyn);
    
    force_residuals_[horizon_index][frame_name] = force_residual;
    std::cout << "Force residual for frame " << frame_name << " added with reference: " << force_reference << std::endl;

    std::shared_ptr<crocoddyl::ActivationModelAbstract> force_activation = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights);

    std::cout << "Force activation for frame " << frame_name << " added with weights: " << weights.transpose() << std::endl;

    std::shared_ptr<crocoddyl::CostModelAbstract> force_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, force_activation, force_residuals_[horizon_index][frame_name]);
    cost_model->addCost("force_tracking_" + frame_name, force_cost, cost_weight);

    std::cout << "Force tracking cost for frame " << frame_name << " added with weight: " << cost_weight << std::endl;
}

void HumanoidMulticontactTracker::deactivateContacts(const std::vector<std::string>& frame_names){

    for (const auto& frame_name : frame_names) {
        for(size_t i = 0; i < N_horizon_; i++) {
            running_contact_models_[i]->changeContactStatus(frame_name + "_contact", false);
            running_cost_model_[i]->changeCostStatus(frame_name + "_friction_cone", false);
        }
        terminal_contact_models_->changeContactStatus(frame_name + "_contact_terminal", false);
        terminal_cost_model_->changeCostStatus(frame_name + "_friction_cone", false);
    }
}

/**
 * @brief Activates the contacts for the specified frame names.
 * 
 * This function sets the reference pose for each frame in the `frame_names` vector at the current reached pose
 * and activates the corresponding contacts in the running and terminal contact models.
 * 
 * @param frame_names A vector of strings representing the names of the frames to activate contacts for.
 * 
 * @note This function assumes that the `pinocchio_data_` and `model_full_` are already initialized and contain the necessary frame information.
 */
void HumanoidMulticontactTracker::activateContacts(const std::vector<std::string>& frame_names){

    if(cost_mask_[3]) {
        for(size_t i = 0; i < N_horizon_ + 1; i++){
            for (const auto& frame_name : frame_names) {
                pinocchio::SE3 contact_pose = pinocchio_data_->oMf[model_full_.getFrameId(frame_name)];
                frame_residuals_[i][frame_name]->set_reference(contact_pose);
            }
        }
    }
    for (const auto& frame_name : frame_names) {
        for (size_t i = 0; i < N_horizon_; i++) {
            running_contact_models_[i]->changeContactStatus(frame_name + "_contact", true);
            running_cost_model_[i]->changeCostStatus(frame_name + "_friction_cone", true);
        }
        terminal_contact_models_->changeContactStatus(frame_name + "_contact_terminal", true);
        terminal_cost_model_->changeCostStatus(frame_name + "_friction_cone", true);
    }
}

void HumanoidMulticontactTracker::switchContacts(const std::vector<std::string>& active_frames, const std::vector<std::string>& inactive_frames, std::vector<bool> & contact_mask, Eigen::VectorXd& xs_prev, const Eigen::Vector3d& desired_com, bool use_quasistatic){

    if(contact_mask.size() != N_horizon_){
        throw std::invalid_argument("contact_mask size must be equal to N_horizon_ + 1");
    }


    for (const auto& frame_name: inactive_frames) {
        for(size_t i = 0; i < N_horizon_; i++) {
            if(contact_mask[i]){
                running_contact_models_[i]->changeContactStatus(frame_name + "_contact", false);
                running_cost_model_[i]->changeCostStatus(frame_name + "_friction_cone", false);
            }
        }
        if(contact_mask[N_horizon_]) {
            terminal_contact_models_->changeContactStatus(frame_name + "_contact_terminal", false);
            terminal_cost_model_->changeCostStatus(frame_name + "_friction_cone", false);
        }
    }

    for (const auto& frame_name: active_frames) {
        for(size_t i = 0; i < N_horizon_; i++) {
            if(contact_mask[i]){
                running_contact_models_[i]->changeContactStatus(frame_name + "_contact", true);
                running_cost_model_[i]->changeCostStatus(frame_name + "_friction_cone", true);
                frame_residuals_[i][frame_name]->set_reference(pinocchio_data_->oMf[model_full_.getFrameId(frame_name)]);
            }
        }
        if(contact_mask[N_horizon_]) {
            terminal_contact_models_->changeContactStatus(frame_name + "_contact_terminal", true);
            terminal_cost_model_->changeCostStatus(frame_name + "_friction_cone", true);
            frame_residuals_[N_horizon_][frame_name]->set_reference(pinocchio_data_->oMf[model_full_.getFrameId(frame_name)]);
        }
    }

    // during contact change, use static solution as initial guess only at first step of change
    if(use_quasistatic){
        std::cout<<"Computing quasi-static \n";

        xs_prev.tail(state_->get_nv()) = Eigen::VectorXd::Zero(state_->get_nv());
        std::vector<Eigen::VectorXd> xs(N_horizon_, xs_prev);
        Eigen::VectorXd us_guess = Eigen::VectorXd::Zero(state_->get_nv()-6);
        quasiStaticFootHandSolution(xs_prev.head(state_->get_nq()), desired_com, us_guess);

        auto first_true = std::distance(contact_mask.begin(), std::find(contact_mask.begin(), contact_mask.end(), true));
        for(size_t i = first_true ; i< N_horizon_; i++) {
            u_prev_[i] = us_guess;
        }
    }

    Eigen::Vector3d left_knee_reference(0.239804, 0.118586, 0.438857);
    Eigen::Vector3d left_ankle_roll_reference(-0.000541275, 0.118514, 0.141274);
    for (int i = 0; i < N_horizon_ + 1; i++) {
        frame_residuals_[i]["left_knee_link"]->set_reference(pinocchio::SE3(Eigen::Matrix3d::Identity(), left_knee_reference));
        frame_residuals_[i]["left_ankle_roll_link"]->set_reference(pinocchio::SE3(Eigen::Matrix3d::Identity(), left_ankle_roll_reference));
    }

    //TODO: add logic to update beziers after re-activating contacts

}

std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> HumanoidMulticontactTracker::createMultiFrameActionModel(const std::vector<std::string>& frame_names, const int horizon_index){

    if (cost_mask_[0]) {
        addRegularizationCosts(xreg_weights_, xreg_weight_, ureg_weight_, mpc_utils::Phase::Running, horizon_index);
    }
    if (cost_mask_[1]) {
        addXBoundCost(xbound_weight_, mpc_utils::Phase::Running, horizon_index);
    }
    if (cost_mask_[2]) {
        addCoMCost(com_tracking_weight_, mpc_utils::Phase::Running, horizon_index);
    }
    if (cost_mask_[3]) {
        for (const auto& frame : frame_targets_) {
            addFrameTrackingCost(frame.first, frame_tracking_weight_, mpc_utils::Phase::Running, horizon_index);
        }
    }
    if (cost_mask_[4]) {
        addContactCosts(frame_names, mpc_utils::Phase::Running, horizon_index);
    }
    if (cost_mask_[5]) {
        std::cout << "[Crocoddyl] Adding force tracking costs for running phase." << std::endl;
        for (const auto& frame_name : frame_names_){
            if (force_tracking_weights_.find(frame_name) != force_tracking_weights_.end()) { // FIXME: starting reference could be initialised different from zero from known vector
                addForceTrackingCost(frame_name, pinocchio::Force::Zero(), force_tracking_weights_[frame_name], force_cost_weight_, mpc_utils::Phase::Running, horizon_index, 3, true);
            }else {
                std::cout << "[Crocoddyl] Warning: Force tracking weights for frame " << frame_name << " not found. Skipping force tracking cost." << std::endl;
            }
        }
    }
    if (cost_mask_[6]){
        addTorqueRateCost(dtau_reg_weight_, mpc_utils::Phase::Running, horizon_index);
    }
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> runningDAM = std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, running_contact_models_[horizon_index], running_cost_model_[horizon_index]);
    return runningDAM;
}

std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> HumanoidMulticontactTracker::createMultiFrameTerminalActionModel(const std::vector<std::string>& frame_names){

    if (cost_mask_[0]) {
        addRegularizationCosts(terminal_xreg_weights_, terminal_xreg_weight_, terminal_ureg_weight_, mpc_utils::Phase::Terminal);
    }
    if (cost_mask_[1]) {
        addXBoundCost(terminal_xbound_weight_, mpc_utils::Phase::Terminal);
    }
    if (cost_mask_[2]) {
        addCoMCost(terminal_com_tracking_weight_, mpc_utils::Phase::Terminal);
    }
    if (cost_mask_[3]) {
        for (const auto& frame : frame_targets_terminal_) {
            addFrameTrackingCost(frame.first, terminal_frame_tracking_weight_, mpc_utils::Phase::Terminal, N_horizon_);
        }
    }
    if (cost_mask_[4]) {
        addContactCosts(frame_names, mpc_utils::Phase::Terminal);
    }
    if (cost_mask_[5]) {
        std::cout << "[Crocoddyl] Adding force tracking costs for terminal phase." << std::endl;
        for (const auto& frame_name : frame_names_){
            if (terminal_force_tracking_weights_.find(frame_name) != terminal_force_tracking_weights_.end()) { // FIXME: starting reference could be initialised different from zero from known vector
                addForceTrackingCost(frame_name, pinocchio::Force::Zero(), terminal_force_tracking_weights_[frame_name], terminal_force_cost_weight_, mpc_utils::Phase::Terminal, N_horizon_, 3, true);
            }else {
                std::cout << "[Crocoddyl] Warning: Force tracking weights for frame " << frame_name << " not found. Skipping force tracking cost." << std::endl;
            }
        }
    }
    
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminalDAM = std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, terminal_contact_models_, terminal_cost_model_);
    return terminalDAM;
}

void HumanoidMulticontactTracker::setFrames(const std::vector<std::string>& frame_names){
    frame_names_ = frame_names;
    for (const auto& frame_name : frame_names_) {
        if (model_full_.existFrame(frame_name)) {
            // std::cout << "Frame " << frame_name << " exists in the model." << std::endl;
        } else {
            std::cout << "[Crocoddyl] Error: Frame " << frame_name << " does not exist in the model." << std::endl;
        }
    }
}

void HumanoidMulticontactTracker::initializeSolver(){

    std::vector<std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>> running_DAMS;
    running_DAMS.reserve(N_horizon_);
    for(std::size_t i = 0; i < N_horizon_; ++i) {   
        running_DAMS.push_back(createMultiFrameActionModel(frame_names_, i));
    }
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminal_DAM = createMultiFrameTerminalActionModel(frame_names_);

    auto create_integrated_model = [&](const auto& dam, bool is_terminal = false)
        -> std::shared_ptr<crocoddyl::IntegratedActionModelAbstract> {
        
        if (integration_method_ == mpc_utils::IntegrationMethod::Euler) {
            return std::static_pointer_cast<crocoddyl::IntegratedActionModelAbstract>(
                std::make_shared<crocoddyl::IntegratedActionModelEuler>(dam, is_terminal ? 0.0 : dt_));
        } else if (integration_method_ == mpc_utils::IntegrationMethod::RK2) {
            return std::static_pointer_cast<crocoddyl::IntegratedActionModelAbstract>(
                std::make_shared<crocoddyl::IntegratedActionModelRK>(dam, crocoddyl::RKType::two, is_terminal ? 0.0 : dt_));
        } else if (integration_method_ == mpc_utils::IntegrationMethod::RK3) {
            return std::static_pointer_cast<crocoddyl::IntegratedActionModelAbstract>(
                std::make_shared<crocoddyl::IntegratedActionModelRK>(dam, crocoddyl::RKType::three, is_terminal ? 0.0 : dt_));
        } else if (integration_method_ == mpc_utils::IntegrationMethod::RK4) {
            return std::static_pointer_cast<crocoddyl::IntegratedActionModelAbstract>(
                std::make_shared<crocoddyl::IntegratedActionModelRK>(dam, crocoddyl::RKType::four, is_terminal ? 0.0 : dt_));
        } else {
            throw std::invalid_argument("Invalid integration method. Supported methods are Euler, RK2, RK3, and RK4.");
        }
    };

    integrated_action_models_.reserve(N_horizon_);
    for (const auto& dam : running_DAMS) {
        integrated_action_models_.push_back(create_integrated_model(dam));
    }
    integrated_terminal_action_model_ = create_integrated_model(terminal_DAM, true);

    problem_ = std::make_shared<crocoddyl::ShootingProblem>(x0_, integrated_action_models_, integrated_terminal_action_model_);
    fddp_ = std::make_shared<crocoddyl::SolverFDDP>(problem_);
    enable_callbacks_ = true;
    if(enable_callbacks_) fddp_->setCallbacks({std::make_shared<crocoddyl::CallbackVerbose>()});

    problem_->set_nthreads(8);
    std::cout << "\n[Crocoddyl] num of threads used: " << problem_->get_nthreads();
    
}

void HumanoidMulticontactTracker::changeWeightedQuadRotWeight(const std::string frame_name,
                                                   std::vector<bool> &contact_mask) {
    VectorXd debug_rot_weights;
    debug_rot_weights.resize(N_horizon_ + 1);

    // update cost weights in running and terminal cost models
    if (frame_targets_.find(frame_name) != frame_targets_.end()) {
        for (size_t i = 0; i < N_horizon_; ++i) {

            // get respective activation
            auto activation = std::dynamic_pointer_cast<crocoddyl::ActivationModelWeightedQuad>(
                running_cost_model_[i]->get_costs().at("frame_" + frame_name)->cost->get_activation());
            if (!activation) {
                std::cout << "Running activation is not of type ActivationModelWeightedQuad." << std::endl;
            }

            double pos_weight = frame_targets_[frame_name][0];
            double rot_weight = 0.;
            Eigen::VectorXd temp_weights(6);

            // if hand is in contact, foot gets swing weights
            if (contact_mask[i]) {
                rot_weight = tracking_swing_rot_weight_;
                temp_weights << pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight;
            } else {
                rot_weight = tracking_contact_rot_weight_;
                temp_weights << pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight;
            }
            activation.get()->set_weights(temp_weights);
            debug_rot_weights[i] = rot_weight;
        }

        // then process the terminal cost
        auto activation = std::dynamic_pointer_cast<crocoddyl::ActivationModelWeightedQuad>(
            terminal_cost_model_->get_costs().at("frame_" + frame_name)->cost->get_activation());
            if (!activation) {
                std::cout << "Terminal activation is not of type ActivationModelWeightedQuad." << std::endl;
            }
        Eigen::VectorXd temp_weights(6);
        double pos_weight = frame_targets_terminal_[frame_name][0];
        double rot_weight = 0.;

        // if hand is in contact, foot gets swing weights
        if (contact_mask[N_horizon_]) {
            rot_weight = terminal_tracking_swing_rot_weight_;
        } else {
            rot_weight = terminal_tracking_contact_rot_weight_;
        }
        temp_weights << pos_weight, pos_weight, pos_weight, rot_weight, rot_weight, rot_weight;
        activation.get()->set_weights(temp_weights);
        debug_rot_weights[N_horizon_] = rot_weight;

        // save new weights
        // frame_targets_[frame_name][1] = rot_weight;
        // frame_targets_terminal_[frame_name][1] = terminal_tracking_contact_rot_weight_;
    } else {
        std::cout << "Cost " << frame_name << " not found." << std::endl;
    }
}

void HumanoidMulticontactTracker::solveOneStep(std::vector<Eigen::VectorXd>& xs_out, std::vector<Eigen::VectorXd>& us_out, mpc_utils::MPCData& data_out, const std::vector<Eigen::Vector3d>& desired_com, std::vector<std::unordered_map<std::string, pinocchio::SE3>> desired_frames, const double time) {

    const std::size_t N = fddp_->get_problem()->get_T();
    long solve_duration = 0.;

    std::vector<Eigen::VectorXd> us_static;
    if(first_iteration_){
        fddp_->set_th_stop(1e-3);
        std::vector<Eigen::VectorXd> xs(N, x0_);
        us_static = problem_->quasiStatic_xs(xs);
        xs.push_back(x0_);
        
        first_iteration_ = false;

        if(cost_mask_[2]) {
            for(size_t i = 0; i < N + 1; i++){
                com_residual_[i]->set_reference(desired_com[i]);
                data_out.com_ref_pos =desired_com[i];
            }
        }
        if(cost_mask_[3]) {
            for(size_t i = 0; i < N + 1; i++){
                for (const auto& frame_name : frame_targets_) {
                    const pinocchio::SE3 temp = desired_frames[i][frame_name.first];
                    if(!isContactActive(frame_name.first)){
                        frame_residuals_[i][frame_name.first]->set_reference(temp);
                    }else{
                        // If the frame is in contact, we set the reference to the current position
                        // This is useful for frames that are already in contact and we want to keep them there
                        // std::cout<<"Setting reference for frame " << frame_name.first << " to current position in contact." << std::endl;
                        // std::cout<<"Current reference is: " << pinocchio_data_->oMf[model_full_.getFrameId(frame_name.first)] << std::endl;
                        pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs[0].head(state_->get_nq()));
                        pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
                        frame_residuals_[i][frame_name.first]->set_reference(pinocchio_data_->oMf[model_full_.getFrameId(frame_name.first)]);
                    }
                }
            }
        }

        problem_->set_x0(xs[0]);

        pinocchio::forwardKinematics(model_full_, *pinocchio_data_, x0_.head(state_->get_nq()));
        pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
        auto start_time = std::chrono::high_resolution_clock::now();
        try {
            fddp_->solve(xs, us_static, max_iter_);
        } catch (const std::exception& e) {
            std::cerr << "Error during FDDP solve: " << e.what() << std::endl;
            throw;
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        if(cost_mask_[6]){
            for (size_t i = 0; i < N; i++) {
                running_cost_model_[i]->changeCostStatus("tauRate", true);
            }
        }

    }else{

        std::vector<std::string> to_remove = {"l_foot_contact"};
        std::vector<std::string> to_add = {"left_rubber_hand"};
        pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
        pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);

        if (contact_switching_manager_->isMaskNotEmpty() && first_contact_change_) {
            std::cout << "Switching contacts at time: " << time << std::endl;
            for(int i=0; i<N_horizon_; i++){
                contact_mask_[i] = true;
            }
            switchContacts(to_add, to_remove, contact_mask_, xs_out[0], desired_com[0], first_contact_change_); //TODO: to_remove and to_add must come from the switching manager, which must contain the plan beforehand
            first_contact_change_ = false;
            std::fill(xs_out.begin(), xs_out.end(), xs_out[0]); // If contact switch happens, then fill xs_out with xs_out[0]
        }

        problem_->set_x0(xs_out[0]);

        if(cost_mask_[6]){
            for(size_t i = 0; i < N ; i++){
                control_residuals_[i]->set_reference(us_out[i]);
            }
        }

        if(cost_mask_[2]) {
            for(size_t i = 0; i < N + 1; i++){
                com_residual_[i]->set_reference(desired_com[i]);
            }
        }

        if(cost_mask_[3]) {
            for(size_t i = 0; i < N + 1; i++){
                for (const auto& frame_name : frame_targets_) {
                    const pinocchio::SE3 temp = desired_frames[i][frame_name.first];
                    if(!isContactActive(frame_name.first) && first_contact_change_){ //FIXME: after contact change I'm not updating references anymore
                        // std::cout << "Contact for frame " << frame_name.first << " is not active, setting reference to desired pose." << std::endl;

                        frame_residuals_[i][frame_name.first]->set_reference(temp);
                        // std::cout << "Setting reference for frame with name : " << frame_name.first <<"\n";
                    }
                }
            }
        }

        // set the joint reference to the current ones to avoid very fast movements
        auto xReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(running_cost_model_[0]->get_costs().at("xReg")->cost->get_residual());
        xReg_res->set_reference(x0_);
        x0_ << xs_out[0].head(state_->get_nq()), Eigen::VectorXd::Zero(state_->get_nv());
        for (size_t i = 1; i < N_horizon_; i++) {
            auto xReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(running_cost_model_[i]->get_costs().at("xReg")->cost->get_residual());
            xReg_res->set_reference(x0_);
        }
        auto xReg_res_term = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(terminal_cost_model_->get_costs().at("xReg")->cost->get_residual());
        xReg_res_term->set_reference(x0_);

        // TODO testing passing uRef from quasi_static or from previous solution
        if (cost_mask_[0]) {
            for (size_t i = 0; i < N_horizon_; i++) {
                auto uReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelControl>(running_cost_model_[i]->get_costs().at("uReg")->cost->get_residual());
                uReg_res->set_reference(u_prev_[0]);
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        try {
            fddp_->solve(xs_out, u_prev_, max_iter_);
        } catch (const std::exception& e) {
            std::cerr << "Error during FDDP solve: " << e.what() << std::endl;
            throw;
        }
        getEigenForceFromSolver();
        auto end_time = std::chrono::high_resolution_clock::now();
        solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    }
    fddp_->get_is_feasible() ? data_out.b_fddp_feasible = true : data_out.b_fddp_feasible = false;
    data_out.total_iterations = fddp_->get_iter();

    xs_out = fddp_->get_xs();
    us_out = fddp_->get_us();
    x_prev_ = fddp_->get_xs();
    u_prev_ = fddp_->get_us();

    pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
    pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
    
    for(int i=0; i<N; i++){
        if(cost_mask_[0]) data_out.xReg_costs.push_back(getCostValue("xReg", i));
        if(cost_mask_[0]) data_out.uReg_costs.push_back(getCostValue("uReg", i));
        if(cost_mask_[1]) data_out.xBound_costs.push_back(getCostValue("xBounds", i));
        if(cost_mask_[2]) data_out.com_costs.push_back(getCostValue("CoMTracking", i));
        if(cost_mask_[3]) data_out.left_hand_frame_costs.push_back(getCostValue("frame_left_rubber_hand", i));
        if(cost_mask_[3]) data_out.right_hand_frame_costs.push_back(getCostValue("frame_right_rubber_hand", i));
        if(cost_mask_[3]) data_out.left_ankle_frame_costs.push_back(getCostValue("frame_left_ankle_roll_link", i));
        if(cost_mask_[3]) data_out.right_ankle_frame_costs.push_back(getCostValue("frame_right_ankle_roll_link", i));
        if(cost_mask_[3]) data_out.left_knee_frame_costs.push_back(getCostValue("frame_left_knee_link", i));
        if(cost_mask_[3]) data_out.right_knee_frame_costs.push_back(getCostValue("frame_right_knee_link", i));
        if(cost_mask_[3]) data_out.torso_link_frame_costs.push_back(getCostValue("frame_torso_primitive_shape", i));
        if(cost_mask_[3]) data_out.left_hand_contact_costs.push_back(getCostValue("left_rubber_hand_friction_cone", i));
        if(cost_mask_[3]) data_out.right_hand_contact_costs.push_back(getCostValue("right_rubber_hand_friction_cone", i));
        if(cost_mask_[3]) data_out.left_foot_contact_costs.push_back(getCostValue("l_foot_contact_friction_cone", i));
        if(cost_mask_[3]) data_out.right_foot_contact_costs.push_back(getCostValue("r_foot_contact_friction_cone", i));
    }

    data_out.com_curr_pos =  pinocchio::centerOfMass(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));

    if(cost_mask_[3]) {
        for(size_t i = 0; i < N + 1; i++){
            for (const auto& frame_name : frame_targets_) {
                data_out.frame_current_pos[frame_name.first] = pinocchio_data_->oMf[model_full_.getFrameId(frame_name.first)].translation();
            }
        }
    }

    auto contact_forces = getEigenForceFromSolver();
    for(int i=0; i<N; i++){
        for(const auto& contacts : running_contact_models_[i]->get_contacts()) {
            std::string contact_id = contacts.first + "_" + std::to_string(i);
            data_out.contact_forces[contact_id] = contact_forces[i][contacts.first].head<3>(); // TODO: add terminal also
        }
    }

    // save predicted end effector positions
    for (int i = 0; i < N; ++i) {
        for (const auto& frame_name : track_frame_names_) {
            auto q_current = xs_out[i].head(state_->get_nq());
            pinocchio::forwardKinematics(model_full_, *pinocchio_data_, q_current);
            pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);

            pinocchio::SE3 frame_pose = pinocchio_data_->oMf[model_full_.getFrameId(frame_name)];
            std::string frame_id = frame_name + "_" + std::to_string(i);
            data_out.predicted_frame_positions[frame_id] = frame_pose.translation();
            future_poses_[frame_name].push_back(mpc_utils::SE3_to_Isometry(frame_pose));
        }
    }

    ctx_.time = time;
    if(coordinator_->processFutureTransitions(ctx_, future_poses_, dt_)){
        std::static_pointer_cast<TimeElapsedCondition>(detector_->getCondition("time_elapsed"))->update();
        // detector_->getCondition("plane_pass")->update(); //TODO: pass next plane condition
        // first_contact_change_ = true; // TODO: add this to reset quasi-static computation on next phase
        switch_trigger_ = true; // TODO: this will be used to start next phase
    }
    future_poses_.clear();
    
    contact_mask_ = contact_switching_manager_->getSwitchingMask();

    data_out.solve_duration = solve_duration;
}

double HumanoidMulticontactTracker::getCostValue(const std::string& cost_name, const int horizon_index) const {

    if (integration_method_ == mpc_utils::IntegrationMethod::Euler) {
        auto integrated_data = std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataEuler>(fddp_->get_problem()->get_runningDatas()[horizon_index]);
        if (!integrated_data || !integrated_data->differential) {
            throw std::runtime_error("Invalid data cast for IntegratedActionDataEuler or differential data is null.");
        }
        auto contact_data = std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(integrated_data->differential);
        if (!contact_data) {
            throw std::runtime_error("Invalid data cast for DifferentialActionDataContactFwdDynamics.");
        }
        auto it = contact_data->costs->costs.find(cost_name);
        if (it == contact_data->costs->costs.end()) {
            throw std::runtime_error("Cost name '" + cost_name + "' not found in the cost model.");
        }

        return it->second->cost;

    } else {
        auto integrated_data = std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataRK>(fddp_->get_problem()->get_runningDatas()[horizon_index]);
        if (!integrated_data || !integrated_data->differential.back()) {
            throw std::runtime_error("Invalid data cast for IntegratedActionDataEuler or differential data is null.");
        }
        auto contact_data = std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(integrated_data->differential.back());
        if (!contact_data) {
            throw std::runtime_error("Invalid data cast for DifferentialActionDataContactFwdDynamics.");
        }
        auto it = contact_data->costs->costs.find(cost_name);
        if (it == contact_data->costs->costs.end()) {
            throw std::runtime_error("Cost name '" + cost_name + "' not found in the cost model.");
        }

        return it->second->cost;
    }
}

void HumanoidMulticontactTracker::computeDARE(const std::vector<Eigen::VectorXd>& xs_out, const std::vector<Eigen::VectorXd>& us_out) {
    
    const std::size_t N = fddp_->get_problem()->get_T();

    problem_->calc(xs_out, us_out);
    problem_->calcDiff(xs_out, us_out);
    auto TModel = problem_->get_runningModels()[0];
    auto TData = problem_->get_runningDatas()[0];

    Eigen::MatrixXd A = TData->Fx;
    Eigen::MatrixXd B = TData->Fu;
    Eigen::MatrixXd Q = TData->Lxx;
    Eigen::MatrixXd R = TData->Luu;

    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = mpc_utils::calcDARE(A, B, Q, R, K_DARE_);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if(success){
        std::cout<<"\n\nOptimal P:\n" << K_DARE_ << std::endl;
        std::cout<<"Duration : " << duration << "ms\n";
    }

}

void HumanoidMulticontactTracker::quasiStaticFootHandSolution(const VectorXd& q_current,
                                                              const Vector3d& desired_com,
                                                              VectorXd& tau_guess) const {
    VectorXd v = VectorXd::Zero(model_full_.nv);
    VectorXd a = VectorXd::Zero(model_full_.nv);
    double percMass = 0.95; //percentage of the total mass on the foot

    // set desired forces to point towards the CoM
    Eigen::Vector3d left_hand_pos = pinocchio_data_->oMi[model_full_.frames[model_full_.getFrameId("left_rubber_hand")].parent].translation();
    Eigen::Vector3d right_foot_pos = pinocchio_data_->oMi[model_full_.frames[model_full_.getFrameId("right_ankle_roll_link")].parent].translation();
    double mass = 35.115;

    // use heuristic centroidal dynamics to compute the forces
    Vector3d com_pos_hand = left_hand_pos - desired_com;    // hand position w.r.t. CoM
    Vector3d com_pos_foot = right_foot_pos - desired_com;   // foot position w.r.t. CoM
    Matrix3d skew_foot = util::SkewSymmetric(com_pos_foot);
    Matrix3d skew_hand = util::SkewSymmetric(com_pos_hand);

    Matrix<double, 7, 6> EE_pos_aug = Matrix<double, 7, 6>::Zero();
    Matrix<double, 7, 1> forces_aug = Matrix<double, 7, 1>::Zero();
    EE_pos_aug.block<3, 3>(0, 0) = Matrix3d::Identity();
    EE_pos_aug.block<3, 3>(0, 3) = Matrix3d::Identity();
    EE_pos_aug.block<3, 3>(3, 0) = skew_foot;
    EE_pos_aug.block<3, 3>(3, 3) = skew_hand;
    EE_pos_aug(6,2) = 1.0;  // assume some percentage of the total mass is on the foot
    forces_aug(2, 0) = -mass * model_full_.gravity981.z();
    forces_aug(6, 0) = -percMass * mass * model_full_.gravity981.z();

    auto pInvEE = EE_pos_aug.completeOrthogonalDecomposition().pseudoInverse();
    Matrix<double, 6, 1> forces_ini_guess = pInvEE * forces_aug;

    PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext(model_full_.joints.size(), pinocchio::Force::Zero());
    auto rf = model_full_.getJointId("right_ankle_pitch_joint");
    auto lh = model_full_.getJointId("left_wrist_roll_joint");
    fext[rf] = pinocchio::Force(forces_ini_guess.head(3), Vector3d::Zero());
    fext[lh] = pinocchio::Force(forces_ini_guess.tail(3), Vector3d::Zero());

    // solve for torques
    tau_guess = (rnea(model_full_, *pinocchio_data_, q_current, v, a, fext)).tail(u_prev_[0].size());
}

void HumanoidMulticontactTracker::printContacts() const {
    std::cout << "Active contacts at this step:" << std::endl;
    for(int i=0; i<N_horizon_; i++){
        for (const auto& contact_pair : running_contact_models_[i]->get_contacts()) {
            const std::string& name = contact_pair.first;
            const auto& contact = contact_pair.second;
            if (contact->active) {
                std::cout << "Contact: " << name << " at i: " <<i <<" is active." << std::endl;
            } else {
                std::cout << "Contact: " << name << " at i: " <<i << " is inactive." << std::endl;
            }
            if (name.find("left_rubber_hand") != std::string::npos) {
                auto cone = std::dynamic_pointer_cast<crocoddyl::ResidualModelContactFrictionCone>(
                    running_cost_model_[i]->get_costs().at("left_rubber_hand_friction_cone")->cost->get_residual());
                std::cout << "got cone" << std::endl;
                std::cout << "cone ref: " << std::endl << cone->get_reference() << std::endl;
            }
        }
    }
    
    for (const auto& contact_pair : terminal_contact_models_->get_contacts()) {
        const std::string& name = contact_pair.first;
        const auto& contact = contact_pair.second;
        if (contact->active) {
            std::cout << "Contact: " << name << " is active." << std::endl;
        } else {
            std::cout << "Contact: " << name << " is inactive." << std::endl;
        }
    }
}

bool HumanoidMulticontactTracker::isContactActive(const std::string& contact_name) const {
    
    try {
        if(contact_name == "left_ankle_roll_link"){
            return running_contact_models_[0]->get_contacts().at("l_foot_contact_contact")->active;
        }else if(contact_name == "right_ankle_roll_link"){
            return running_contact_models_[0]->get_contacts().at("r_foot_contact_contact")->active;
        }else{
            return running_contact_models_[0]->get_contacts().at(contact_name + "_contact")->active;
        }
    } catch (const std::out_of_range&) {
        return false;
    }

}

std::vector<std::vector<std::map<std::string, pinocchio::Force>>> const HumanoidMulticontactTracker::getForceFromSolver(){

    auto running_models = problem_->get_runningModels();
    auto running_datas = problem_->get_runningDatas();
    std::vector<std::vector<std::map<std::string, pinocchio::Force>>> forces_trajectory;

    for (std::size_t t = 0; t < N_horizon_; ++t) {
        auto model = running_models[t];
        auto data = running_datas[t];
        
        std::vector<std::map<std::string, pinocchio::Force>> time_step_forces;
        std::map<std::string, pinocchio::Force> contact_forces;

        if (integration_method_ == mpc_utils::IntegrationMethod::Euler) {
            auto integrated_data = std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataEuler>(data);
            if (!integrated_data || !integrated_data->differential) {
                forces_trajectory.push_back(time_step_forces);
                continue;
            }
            auto contact_data = std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(
            integrated_data->differential);
            if (!contact_data) {
                forces_trajectory.push_back(time_step_forces);
                continue;
            }
            const auto& contacts_data = contact_data->multibody.contacts->contacts;
            for (const auto& contact_pair : contacts_data) {
                const std::string& name = contact_pair.first;
                const auto& contact = contact_pair.second;
                contact_forces[name] = contact->f;
                // std::cout << "t=" << t << ", " << name
                //         << ": f_linear=" << contact->f.linear().transpose()
                //         << ", f_angular=" << contact->f.angular().transpose() << std::endl;
            }

            if (!contact_forces.empty()) {
                time_step_forces.push_back(contact_forces);
            }
            forces_trajectory.push_back(time_step_forces);

        } else{
            auto integrated_data = std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataRK>(data);
            if (!integrated_data || !integrated_data->differential.back()) {
                forces_trajectory.push_back(time_step_forces);
                continue;
            }
            auto contact_data = std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(
            integrated_data->differential.back());
            if (!contact_data) {
                forces_trajectory.push_back(time_step_forces);
                continue;
            }
            const auto& contacts_data = contact_data->multibody.contacts->contacts;
            for (const auto& contact_pair : contacts_data) {
                const std::string& name = contact_pair.first;
                const auto& contact = contact_pair.second;
                contact_forces[name] = contact->f;
                // std::cout << "t=" << t << ", " << name
                //         << ": f_linear=" << contact->f.linear().transpose()
                //         << ", f_angular=" << contact->f.angular().transpose() << std::endl;
            }

            if (!contact_forces.empty()) {
                time_step_forces.push_back(contact_forces);
            }
            forces_trajectory.push_back(time_step_forces);
        }

    }
    
    return forces_trajectory;

}

std::vector<std::map<std::string, Eigen::Matrix<double,6,1>>> const HumanoidMulticontactTracker::getEigenForceFromSolver(){

    auto running_models = problem_->get_runningModels();
    auto running_datas = problem_->get_runningDatas();
    std::vector<std::map<std::string, Eigen::Matrix<double,6,1>>> forces_trajectory;

    for (std::size_t t = 0; t < N_horizon_; t++) {
        auto model = running_models[t];
        auto data = running_datas[t];
        
        std::map<std::string, Eigen::Matrix<double,6,1>> contact_forces;

        if (integration_method_ == mpc_utils::IntegrationMethod::Euler) {
            auto integrated_data = std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataEuler>(data);
            if (!integrated_data || !integrated_data->differential) {
                forces_trajectory.push_back(contact_forces);
                continue;
            }
        
        auto contact_data = std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(
        integrated_data->differential);
        if (!contact_data) {
            forces_trajectory.push_back(contact_forces);
            continue;
        }

        const auto& contacts_data = contact_data->multibody.contacts->contacts;
        for (const auto& contact_pair : contacts_data) {
            const std::string& name = contact_pair.first;
            const auto& contact = contact_pair.second;
            // auto joint = state_->get_pinocchio()->frames[contact->frame].parent;
            // auto oMf = pinocchio_data_->oMi[joint] * contact->jMf;
            // auto fiMo = pinocchio::SE3(contact->pinocchio->oMi[joint].rotation().transpose(), contact->jMf.translation());
            // auto force =  fiMo.actInv(contact->f);
            auto force = contact->f; // Use the force directly from the contact data assuming that its in world frame, but what about the rotation we pass earlier? 
            contact_forces[name] = mpc_utils::fromPinocchioForce(force);
        }

        forces_trajectory.push_back(contact_forces);
    
        } else {
            auto integrated_data = std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataRK>(data);
            if (!integrated_data || !integrated_data->differential.back()) {
                forces_trajectory.push_back(contact_forces);
                continue;
            }

            auto contact_data = std::dynamic_pointer_cast<crocoddyl::DifferentialActionDataContactFwdDynamics>(
            integrated_data->differential.back());
            if (!contact_data) {
                forces_trajectory.push_back(contact_forces);
                continue;
            }

            const auto& contacts_data = contact_data->multibody.contacts->contacts;
            for (const auto& contact_pair : contacts_data) {
                const std::string& name = contact_pair.first;
                const auto& contact = contact_pair.second;
                // auto joint = state_->get_pinocchio()->frames[contact->frame].parent;
                // auto oMf = pinocchio_data_->oMi[joint] * contact->jMf;
                // auto fiMo = pinocchio::SE3(contact->pinocchio->oMi[joint].rotation().transpose(), contact->jMf.translation());
                // auto force =  fiMo.actInv(contact->f);
                auto force = contact->f; // Use the force directly from the contact data assuming that its in world frame, but what about the rotation we pass earlier? 
                contact_forces[name] = mpc_utils::fromPinocchioForce(force);
            }

            forces_trajectory.push_back(contact_forces);
        }
    }
    
    return forces_trajectory;

}