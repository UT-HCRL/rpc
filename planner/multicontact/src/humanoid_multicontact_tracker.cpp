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

    config_path_ = THIS_COM "config/g1/sim/mujoco/ihwbc/crocoddyl_params.yaml";
    params_ = YAML::LoadFile(config_path_);
    loadMPCParams();

    frame_residuals_.resize(N_horizon_ + 1);
    force_residuals_.resize(N_horizon_ + 1);
    control_residuals_.resize(N_horizon_);

    pinocchio::urdf::buildModel(robot_path, pinocchio::JointModelFreeFlyer(), model_full_);
    pinocchio_data_ = std::make_unique<pinocchio::Data>(model_full_);

    reduced_model_ = false;
    std::shared_ptr<crocoddyl::StateMultibody> state = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_full_));
    state_ = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_full_));

    actuation_ = std::make_shared<crocoddyl::ActuationModelFloatingBase>(state_);
    for(int i = 0; i < N_horizon_; i++){
        running_cost_model_.push_back(std::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu()));
        running_contact_models_.push_back(std::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu()));
    }
    terminal_cost_model_ = std::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());
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
    const std::vector<std::string> contact_frames = {"l_foot_contact", "r_foot_contact", "left_rubber_hand", "right_rubber_hand"};
    contact_switching_manager_ = std::make_shared<ContactSwitchingManager>(
        contact_frames,
        N_horizon_
    );
    
    //FIXME: these should be passed from track_plan, reading the planner pkl file
    // STEPOVER PLAN
    // contact_switching_manager_->addContactPhase({true, true, false, false}); // Phase 0 double support
    // contact_switching_manager_->addContactPhase({false, true, true, false}); // Phase 1 step over with left foot
    // contact_switching_manager_->addContactPhase({true, true, false, false});  // Phase 2 double support
    // contact_switching_manager_->addContactPhase({true, false, true, false}); // Phase 3 step over with right foot
    // contact_switching_manager_->addContactPhase({true, true, false, false}); // Phase 4 double support

    // STEPON PLAN
                                            //   LF,    RF,    LH,    RH
    // contact_switching_manager_->addContactPhase({true, true, false, false}); // Phase 0 double support
    // contact_switching_manager_->addContactPhase({true, false, true, false}); // Phase 1 step over with left foot
    // contact_switching_manager_->addContactPhase({true, true, true, false});  // Phase 2 double support

    // TWO HAND STEPON PLAN
    contact_switching_manager_->addContactPhase({true, true, false, false}); // Phase 0 double support
    contact_switching_manager_->addContactPhase({true, false, true, true}); // Phase 1 two hands and 1 foot
    contact_switching_manager_->addContactPhase({false, true, true, true}); // Phase 2 two hands and 1 foot
    contact_switching_manager_->addContactPhase({false, true, false, true}); // Phase 3, 1 hand 1 foot
    contact_switching_manager_->getNextPlannedContacts();

    
    detector_ = std::make_shared<TransitionDetector>();
    coordinator_ = std::make_unique<ContactTransitionCoordinator>(contact_switching_manager_, detector_);
    auto time_condition = std::make_shared<TimeElapsedCondition>("time_elapsed", 3.0, false);
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

    contact_mask_.resize(N_horizon_);
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
    loadCoMPolytopeWeights();

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
        double baum_a = 0;
        double baum_b = 0;

        util::ReadParameter(params_["running_costs"]["baumgarte_gains"][frame_name], "alpha", baum_a);
        util::ReadParameter(params_["running_costs"]["baumgarte_gains"][frame_name], "beta", baum_b);
        terminal_contact_weights_[frame_name] = mpc_utils::from2DValues(baum_a, baum_b);

        contact_weights_[frame_name] = mpc_utils::from2DValues(baum_a, baum_b);

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

    xreg_weights_ = Eigen::VectorXd::Zero(2 * nv);
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

    terminal_xreg_weights_ = Eigen::VectorXd::Zero(2 * nv);
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(i) = pow(base_pos_weights[i],2);
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(3 + i) = pow(base_rot_weights[i],2);
    for (int i = 6; i < nv; ++i) terminal_xreg_weights_(i) = pow(joint_pos_weights,2);
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(nv + i) = pow(base_lin_vel_weights[i],2);
    for (int i = 3; i < 6; ++i) terminal_xreg_weights_(nv + i) = pow(base_ang_vel_weights[i],2);
    for (int i = 6; i < nv; ++i) terminal_xreg_weights_(nv + i) = pow(joint_vel_weights,2);

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

void HumanoidMulticontactTracker::loadCoMPolytopeWeights(){
    util::ReadParameter(params_["running_costs"]["CoMPolytope"], "w", com_polytope_weight_);
    mpc_utils::normalize_weights(com_polytope_weight_, N_horizon_);

    util::ReadParameter(params_["terminal_costs"]["CoMPolytope"], "w", terminal_com_polytope_weight_);
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
        if(force_tracking_weight.size() != 6) {
            throw std::runtime_error("Force tracking weight for frame " + frame + " must be a vector of size 6, got: " + std::to_string(force_tracking_weight.size()));
        }
        force_tracking_weights_[frame] = mpc_utils::fromStdVector(force_tracking_weight);
        mpc_utils::normalize_weights(force_tracking_weights_[frame], N_horizon_);

        if(force_tracking_weight.size() != 6) {
            throw std::runtime_error("Force tracking weight for frame " + frame + " must be a vector of size 6, got: " + std::to_string(force_tracking_weight.size()));
        }
        terminal_force_tracking_weights_[frame] = mpc_utils::fromStdVector(force_tracking_weight);
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
    util::ReadParameter(params_["mpc"], "verbose", enable_callbacks_);

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

    double contact_lookahead;
    util::ReadParameter(params_["mpc"], "contact_lookahead", contact_lookahead);
    knots_lh_ = static_cast<int>(std::ceil(contact_lookahead / (dt_*1000)));
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

void HumanoidMulticontactTracker::addCoMPolytopeCost(const double com_poly_weight,
                                                    const mpc_utils::Phase phase,
                                                    const int horizon_index) {
    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;

    mpc_utils::computeHalfSpaceRep(mpc_utils::computeConvexHull(getContactPoints(horizon_index)), A_, b_);

    // Set bounds for quadratic barrier: residual <= 0 (inside polytope)
    Eigen::VectorXd lb = Eigen::VectorXd::Constant(A_.rows(), -1e10);  // very large negative lower bound
    Eigen::VectorXd ub = Eigen::VectorXd::Zero(A_.rows());             // upper bound zero
    crocoddyl::ActivationBounds bounds(lb, ub);

    auto activation = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
    auto residual = std::make_shared<crocoddyl::ResidualModelStaticPolytope>(state_, A_, b_, actuation_->get_nu());
    com_polytope_residuals_.push_back(residual);
    auto cost = std::make_shared<crocoddyl::CostModelResidual>(state_, activation, residual);
    cost_model->addCost("com_polytope", cost, com_poly_weight);
}

void HumanoidMulticontactTracker::addCoMPolytopeVariantsCost(const double com_poly_weight, const mpc_utils::Phase phase, const int horizon_index){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;

    std::vector<Eigen::Vector2d> dummy_poly4 = {
        {0.1,  0.1},
        {0.1, -0.1},
        {-0.1, -0.1},
        {-0.1,  0.1}
    };
    Eigen::MatrixXd A4;
    Eigen::VectorXd b4;
    mpc_utils::computeHalfSpaceRep(dummy_poly4, A4, b4);
    Eigen::VectorXd lb4 = Eigen::VectorXd::Constant(A4.rows(), -1e10);
    Eigen::VectorXd ub4 = Eigen::VectorXd::Zero(A4.rows());
    auto activation4 = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(lb4, ub4));
    auto residual4 = std::make_shared<crocoddyl::ResidualModelStaticPolytope>(state_, A4, b4, actuation_->get_nu());
    residuals_poly4_.push_back(residual4);
    cost_model->addCost("com_polytope_4", std::make_shared<crocoddyl::CostModelResidual>(state_, activation4, residual4), com_poly_weight);
    cost_model->changeCostStatus("com_polytope_4", false);

    std::vector<Eigen::Vector2d> dummy_poly5 = {
        {0.1,  0.1},
        {0.1, -0.1},
        {-0.1, -0.1},
        {-0.1,  0.1},
        {0.0,  0.2}
    };
    Eigen::MatrixXd A5;
    Eigen::VectorXd b5;
    mpc_utils::computeHalfSpaceRep(dummy_poly5, A5, b5);
    Eigen::VectorXd lb5 = Eigen::VectorXd::Constant(A5.rows(), -1e10);
    Eigen::VectorXd ub5 = Eigen::VectorXd::Zero(A5.rows());
    auto activation5 = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(lb5, ub5));
    auto residual5 = std::make_shared<crocoddyl::ResidualModelStaticPolytope>(state_, A5, b5, actuation_->get_nu());
    residuals_poly5_.push_back(residual5);
    cost_model->addCost("com_polytope_5", std::make_shared<crocoddyl::CostModelResidual>(state_, activation5, residual5), com_poly_weight);
    cost_model->changeCostStatus("com_polytope_5", false);

    std::vector<Eigen::Vector2d> dummy_poly6 = {
        { 0.1,  0.1},
        { 0.1, -0.1},
        {-0.1, -0.1},
        {-0.1,  0.1},
        { 0.0,  0.2},
        { -0.2, 0.0}
    };
    Eigen::MatrixXd A6;
    Eigen::VectorXd b6;
    mpc_utils::computeHalfSpaceRep(dummy_poly6, A6, b6);
    Eigen::VectorXd lb6 = Eigen::VectorXd::Constant(A6.rows(), -1e10);
    Eigen::VectorXd ub6 = Eigen::VectorXd::Zero(A6.rows());
    auto activation6 = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(lb6, ub6));
    auto residual6 = std::make_shared<crocoddyl::ResidualModelStaticPolytope>(state_, A6, b6, actuation_->get_nu());
    residuals_poly6_.push_back(residual6);
    cost_model->addCost("com_polytope_6", std::make_shared<crocoddyl::CostModelResidual>(state_, activation6, residual6), com_poly_weight);
    cost_model->changeCostStatus("com_polytope_6", false);
    
}

std::vector<Eigen::Vector2d> HumanoidMulticontactTracker::getContactPoints(const int horizon_index, const double length, const double width){

    std::vector<Eigen::Vector2d> active_contact_positions;

    std::vector<Eigen::Vector3d> local_corners = {
                    { length/2,  width/2, 0.0},
                    { length/2, -width/2, 0.0},
                    {-length/2, -width/2, 0.0},
                    {-length/2,  width/2, 0.0}
                };
    for (const auto& frame_name : frame_names_) {
        if (running_contact_models_[horizon_index]->get_contacts().at(frame_name + "_contact")->active) {
            pinocchio::FrameIndex fid = model_full_.getFrameId(frame_name);
            std::cout << "Contact active for frame: " << frame_name << " at horizon index " << horizon_index << std::endl;
            const pinocchio::SE3& pose = pinocchio_data_->oMf[fid];
            std::cout << "contacts for " <<frame_name<<"\n";
            if (frame_name.find("foot") != std::string::npos) {
                for (const auto& corner_local : local_corners) {
                    Eigen::Vector3d corner_world = pose.act(corner_local);
                    active_contact_positions.push_back(corner_world.head<2>());
                    std::cout << "[" << corner_world.x() << ", " << corner_world.y() << "] ";
                }
                std::cout << std::endl;
            } else {
                active_contact_positions.push_back(pose.translation().head<2>());
                std::cout << "[" << pose.translation().x() << ", " << pose.translation().y() << "] ";
            }
        }else{
            // std::cout << "Contact not active for frame: " << frame_name << " at horizon index " << horizon_index << std::endl;
        }
    }

    return active_contact_positions;

    // std::cout << "Active contact points at horizon index " << horizon_index << ":" << std::endl;
    // for (const auto& contact_position : active_contact_positions) {
    //     std::cout << "Contact point: [" << contact_position.x() << ", " << contact_position.y() << "]" << std::endl;
    // }
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
    cost_model->changeCostStatus("frame_" + frame_name, false); // disable cost by default

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
    bool is_active = true;

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
                support_contact_model =
                    std::make_shared<crocoddyl::ContactModel3D>(state_, model_full_.getFrameId(frame_name), xref, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), contact_weights_[frame_name]);
                    is_active = false;
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
                support_contact_model =
                    std::make_shared<crocoddyl::ContactModel3D>(state_, model_full_.getFrameId(frame_name), xref, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), terminal_contact_weights_[frame_name]);
                is_active = false;
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
            cost_model->changeCostStatus(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", is_active);
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
            cost_model->changeCostStatus(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", is_active);
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

void HumanoidMulticontactTracker::addForceTrackingCost(const std::string& frame_name, const pinocchio::Force& force_reference, const Eigen::VectorXd weights, const double cost_weight, const mpc_utils::Phase phase, const int horizon_index, const size_t contact_f_dim, const bool fwwddyn){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;
    std::cout << "Adding force tracking cost for frame: " << frame_name << std::endl;

    std::shared_ptr<crocoddyl::ResidualModelContactForce> force_residual = std::make_shared<crocoddyl::ResidualModelContactForce>(state_, model_full_.getFrameId(frame_name), force_reference, contact_f_dim, actuation_->get_nu(), fwwddyn);
    
    force_residuals_[horizon_index][frame_name] = force_residual;
    std::cout << "Force residual for frame " << frame_name << " added with reference: " << force_reference << std::endl;


    if(contact_f_dim == 6){
        std::shared_ptr<crocoddyl::ActivationModelAbstract> force_activation = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights);
        std::cout << "Force activation for frame " << frame_name << " added with weights: " << weights.transpose() << std::endl;
        std::shared_ptr<crocoddyl::CostModelAbstract> force_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, force_activation, force_residuals_[horizon_index][frame_name]);
        cost_model->addCost("force_tracking_" + frame_name, force_cost, cost_weight);
        std::cout << "Force tracking cost for frame " << frame_name << " added with weight: " << cost_weight << std::endl;
    }else{
        std::shared_ptr<crocoddyl::ActivationModelAbstract> force_activation = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(weights.head<3>());
        std::cout << "Force activation for frame " << frame_name << " added with weights: " << weights.transpose() << std::endl;
        std::shared_ptr<crocoddyl::CostModelAbstract> force_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, force_activation, force_residuals_[horizon_index][frame_name]);
        cost_model->addCost("force_tracking_" + frame_name, force_cost, cost_weight);
        std::cout << "Force tracking cost for frame " << frame_name << " added with weight: " << cost_weight << std::endl;
    }

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

void HumanoidMulticontactTracker::updateRunningWeights(double xReg_multiplier, double uReg_multiplier, const std::vector<bool>& contact_mask){

    for(size_t i = 0; i < N_horizon_; i++){
        if(contact_mask[i]){
            // auto& left_hand_tracking_item = running_cost_model_[i]->get_costs().at("frame_left_rubber_hand");
            // left_hand_tracking_item->weight += 100;
            auto& xReg_cost_item = running_cost_model_[i]->get_costs().at("xReg");
            xReg_cost_item->weight *= xReg_multiplier;
            auto& uReg_cost_item = running_cost_model_[i]->get_costs().at("uReg");
            uReg_cost_item->weight *= uReg_multiplier;
            // diviso 1.25
        }
    }
}

void HumanoidMulticontactTracker::updateTerminalWeights(double xReg_multiplier, const std::vector<bool>& contact_mask){
    
    if(contact_mask.back()){
        // auto& left_hand_tracking_item = terminal_cost_model_->get_costs().at("frame_left_rubber_hand");
        // left_hand_tracking_item->weight += 100;
        // auto& xReg_cost_item = terminal_cost_model_->get_costs().at("xReg");
        // xReg_cost_item->weight *= xReg_multiplier;

    }
}

void HumanoidMulticontactTracker::switchContacts(const std::vector<std::string>& active_frames, const std::vector<std::string>& inactive_frames, std::vector<bool> & contact_mask, Eigen::VectorXd& xs_prev, const Eigen::Vector3d& desired_com, bool use_quasistatic){

    if(contact_mask.size() != N_horizon_){
        throw std::invalid_argument("contact_mask size must be equal to N_horizon_");
    }

    for (const auto& frame_name: inactive_frames) {
        std::string tracking_frame_name;
        if(frame_name == "l_foot_contact") tracking_frame_name = "left_ankle_roll_link"; //FIXME: create a mapping or else
        else if(frame_name == "r_foot_contact") tracking_frame_name = "right_ankle_roll_link";
        else tracking_frame_name = frame_name;
        for(size_t i = 0; i < N_horizon_; i++) {
            if(contact_mask[i]){
                running_contact_models_[i]->changeContactStatus(frame_name + "_contact", false);
                running_cost_model_[i]->changeCostStatus(frame_name + "_friction_cone", false);
                running_cost_model_[i]->changeCostStatus("frame_" + tracking_frame_name, true);
                auto activation = std::dynamic_pointer_cast<crocoddyl::ActivationModelWeightedQuad>(running_cost_model_[i]->get_costs().at("frame_" + tracking_frame_name)->cost->get_activation());
                if (activation) {
                    Eigen::VectorXd weights(6);
                    double pos_w = frame_targets_[tracking_frame_name](0);
                    double rot_w = tracking_swing_rot_weight_; // swing rotational weight
                    weights << pos_w, pos_w, pos_w, rot_w, rot_w, rot_w;
                    activation->set_weights(weights);
                    std::cout << "Changing weight for: " << "frame_" + tracking_frame_name << " to: " << weights.transpose() << std::endl;

                }
            }
        }
        if(contact_mask.back()) {
            terminal_contact_models_->changeContactStatus(frame_name + "_contact_terminal", false);
            terminal_cost_model_->changeCostStatus(frame_name + "_friction_cone", false);
            terminal_cost_model_->changeCostStatus("frame_" + tracking_frame_name, true);
            auto activation = std::dynamic_pointer_cast<crocoddyl::ActivationModelWeightedQuad>(terminal_cost_model_->get_costs().at("frame_" + tracking_frame_name)->cost->get_activation());
            if (activation) {
                Eigen::VectorXd weights(6);
                double pos_w = frame_targets_[tracking_frame_name](0);
                double rot_w = tracking_swing_rot_weight_; // swing rotational weight
                weights << pos_w, pos_w, pos_w, rot_w, rot_w, rot_w;
                activation->set_weights(weights);
            }
        }
    }

    for (const auto& frame_name: active_frames) {
        std::string tracking_frame_name;
        if(frame_name == "l_foot_contact") tracking_frame_name = "left_ankle_roll_link"; //FIXME: create a mapping or else
        else if(frame_name == "r_foot_contact") tracking_frame_name = "right_ankle_roll_link";
        else tracking_frame_name = frame_name;

        for(size_t i = 0; i < N_horizon_; i++) {
            if(contact_mask[i]){
                running_contact_models_[i]->changeContactStatus(frame_name + "_contact", true);
                running_cost_model_[i]->changeCostStatus(frame_name + "_friction_cone", true);
                running_cost_model_[i]->changeCostStatus("frame_" + tracking_frame_name, false);
                // frame_residuals_[i][tracking_frame_name]->set_reference(pinocchio_data_->oMf[model_full_.getFrameId(frame_name)]);
                auto activation = std::dynamic_pointer_cast<crocoddyl::ActivationModelWeightedQuad>(running_cost_model_[i]->get_costs().at("frame_" + tracking_frame_name)->cost->get_activation());
                if (activation) {
                    Eigen::VectorXd weights(6);
                    double pos_w = frame_targets_[tracking_frame_name](0);
                    double rot_w = tracking_contact_rot_weight_; // swing rotational weight
                    weights << pos_w, pos_w, pos_w, rot_w, rot_w, rot_w;
                    activation->set_weights(weights);
                    std::cout << "Changing weight for: " << "frame_" + tracking_frame_name << " to: " << weights.transpose() << std::endl;
                }
                
            }
        }
        if(contact_mask.back()) {
            terminal_contact_models_->changeContactStatus(frame_name + "_contact_terminal", true);
            terminal_cost_model_->changeCostStatus(frame_name + "_friction_cone", true);
            terminal_cost_model_->changeCostStatus("frame_" + tracking_frame_name, false);
            // frame_residuals_[N_horizon_][tracking_frame_name]->set_reference(pinocchio_data_->oMf[model_full_.getFrameId(frame_name)]);
            auto activation = std::dynamic_pointer_cast<crocoddyl::ActivationModelWeightedQuad>(terminal_cost_model_->get_costs().at("frame_" + tracking_frame_name)->cost->get_activation());
            if (activation) {
                Eigen::VectorXd weights(6);
                double pos_w = frame_targets_[tracking_frame_name](0);
                double rot_w = tracking_contact_rot_weight_; // swing rotational weight
                weights << pos_w, pos_w, pos_w, rot_w, rot_w, rot_w;
                activation->set_weights(weights);
            }
        }
    }

    if(use_quasistatic){
        std::cout<<"Computing quasi-static \n";

        xs_prev.tail(state_->get_nv()) = Eigen::VectorXd::Zero(state_->get_nv());
        std::vector<Eigen::VectorXd> xs(N_horizon_, xs_prev);
        // Eigen::VectorXd us_guess = Eigen::VectorXd::Zero(state_->get_nv()-6);
        std::vector<Eigen::VectorXd> us_guess(N_horizon_, Eigen::VectorXd::Zero(u_prev_[0].size()));
        auto plan = contact_switching_manager_->getCurrentPlannedContacts();
        quasiStaticSolution(xs_prev, plan, desired_com, us_guess);
        // quasiStaticFootHandSolution(xs_prev.head(state_->get_nq()), desired_com, us_guess);
        
        auto first_true = std::distance(contact_mask.begin(), std::find(contact_mask.begin(), contact_mask.end(), true));
        for(size_t i = first_true ; i< N_horizon_; i++) {
            u_prev_[i] = us_guess[0];
        }

        // updateRunningWeights(0.1, 1.0, contact_mask);
        // updateTerminalWeights(0.0, contact_mask);
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
                int contact_f_dim = (frame_name.find("hand") != std::string::npos) ? 3 : 6;
                addForceTrackingCost(frame_name, pinocchio::Force::Zero(), force_tracking_weights_[frame_name], force_cost_weight_, mpc_utils::Phase::Running, horizon_index, contact_f_dim, true);
            }else {
                std::cout << "[Crocoddyl] Warning: Force tracking weights for frame " << frame_name << " not found. Skipping force tracking cost." << std::endl;
            }
        }
    }
    if (cost_mask_[6]) {
        addTorqueRateCost(dtau_reg_weight_, mpc_utils::Phase::Running, horizon_index);
    }
    if (cost_mask_[7]) {
        // addCoMPolytopeCost(com_polytope_weight_, mpc_utils::Phase::Running, horizon_index); 
        addCoMPolytopeVariantsCost(com_polytope_weight_, mpc_utils::Phase::Running, horizon_index);
    }
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> runningDAM = std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, running_contact_models_[horizon_index], running_cost_model_[horizon_index]);
    return runningDAM;
}

std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> HumanoidMulticontactTracker::createMultiFrameTerminalActionModel(const std::vector<std::string>& frame_names){

    if (cost_mask_[0]) {
        addRegularizationCosts(terminal_xreg_weights_, terminal_xreg_weight_, 0.0, mpc_utils::Phase::Terminal);
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
                int contact_f_dim = (frame_name.find("hand") != std::string::npos) ? 3 : 6;
                addForceTrackingCost(frame_name, pinocchio::Force::Zero(), terminal_force_tracking_weights_[frame_name], terminal_force_cost_weight_, mpc_utils::Phase::Terminal, N_horizon_, contact_f_dim, true);
            }else {
                std::cout << "[Crocoddyl] Warning: Force tracking weights for frame " << frame_name << " not found. Skipping force tracking cost." << std::endl;
            }
        }
    }
    if (cost_mask_[7]) {
        // addCoMPolytopeCost(terminal_com_polytope_weight_, mpc_utils::Phase::Terminal); 
        addCoMPolytopeVariantsCost(terminal_com_polytope_weight_, mpc_utils::Phase::Terminal);
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

void HumanoidMulticontactTracker::updatePolytope(){
    auto cp = getContactPoints(0); //FIXME: we can do it for all the knots, but rn we dont have mixed contacts in the horizon
    std::cout << "Contact Points: ";
    for (const auto& point : cp) {
        std::cout << "[" << point.x() << ", " << point.y() << "] ";
    }
    std::cout << std::endl;
    auto contact_points_hull = mpc_utils::computeConvexHull(cp); 
    std::cout << "Contact Points Hull Size: " << contact_points_hull.size() << std::endl;
    std::cout << "Contact Points Hull: ";
    for (const auto& point : contact_points_hull) {
        std::cout << "[" << point.x() << ", " << point.y() << "] ";
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < N_horizon_; ++i) {        
        if (contact_points_hull.size() == 4) {
            mpc_utils::computeHalfSpaceRep(contact_points_hull, A_, b_);
            residuals_poly4_[i]->setPolytope(A_, b_);
            running_cost_model_[i]->changeCostStatus("com_polytope_4", true);
            running_cost_model_[i]->changeCostStatus("com_polytope_5", false);
            running_cost_model_[i]->changeCostStatus("com_polytope_6", false);
            if(i == N_horizon_ -1){
                terminal_cost_model_->changeCostStatus("com_polytope_4", true);
                terminal_cost_model_->changeCostStatus("com_polytope_5", false);
                terminal_cost_model_->changeCostStatus("com_polytope_6", false);
            }
        } else if (contact_points_hull.size() == 5) {
            mpc_utils::computeHalfSpaceRep(contact_points_hull, A_, b_);
            residuals_poly5_[i]->setPolytope(A_, b_);
            running_cost_model_[i]->changeCostStatus("com_polytope_4", false);
            running_cost_model_[i]->changeCostStatus("com_polytope_5", true);
            running_cost_model_[i]->changeCostStatus("com_polytope_6", false);
            if(i == N_horizon_ -1){
                terminal_cost_model_->changeCostStatus("com_polytope_4", false);
                terminal_cost_model_->changeCostStatus("com_polytope_5", true);
                terminal_cost_model_->changeCostStatus("com_polytope_6", false);
            }
        } else if (contact_points_hull.size() == 6) {
            mpc_utils::computeHalfSpaceRep(contact_points_hull, A_, b_);
            residuals_poly6_[i]->setPolytope(A_, b_);
            running_cost_model_[i]->changeCostStatus("com_polytope_4", false);
            running_cost_model_[i]->changeCostStatus("com_polytope_5", false);
            running_cost_model_[i]->changeCostStatus("com_polytope_6", true);
            if(i == N_horizon_ -1){
                terminal_cost_model_->changeCostStatus("com_polytope_4", false);
                terminal_cost_model_->changeCostStatus("com_polytope_5", false);
                terminal_cost_model_->changeCostStatus("com_polytope_6", true);
            }
        } else {
            std::cout << "[Crocoddyl] Warning: Unsupported number of contact points for polytope cost. Expected 4-5-6, got " << contact_points_hull.size() << ". Skipping polytope cost." << std::endl;
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

    if (cost_mask_[7]) {
        updatePolytope();
    }
    
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
    if(enable_callbacks_) fddp_->setCallbacks({std::make_shared<crocoddyl::CallbackVerbose>()});

    // fddp_->set_reg_incfactor(fddp_->get_reg_incfactor()*10);
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

    if(first_iteration_){
        fddp_->set_th_stop(1e-3);
        // std::vector<Eigen::VectorXd> xs(N, x0_);
        // us_static = problem_->quasiStatic_xs(xs);
        // xs.push_back(x0_);
        std::cout << "First iteration \n";
        std::vector<Eigen::VectorXd> xs(N+1, xs_out[0]);
        std::vector<Eigen::VectorXd> us_static(N, us_out[0]);
        
        first_iteration_ = false;

        if(cost_mask_[0]) {
            for (size_t i = 0; i < N_horizon_; i++) {
                auto uReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelControl>(running_cost_model_[i]->get_costs().at("uReg")->cost->get_residual());
                uReg_res->set_reference(us_out[0]);         // TODO testing passing uRef from quasi_static or from previous solution
            }
            for (size_t i = 0; i < N_horizon_; i++) {
                auto xReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(running_cost_model_[i]->get_costs().at("xReg")->cost->get_residual());
                xReg_res->set_reference(xs_out[i]);
            }
            auto xReg_res_term = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(terminal_cost_model_->get_costs().at("xReg")->cost->get_residual());
            xReg_res_term->set_reference(xs_out[N_horizon_]);
        }

        if(cost_mask_[2]) {
            for(size_t i = 0; i < N + 1; i++){
                com_residual_[i]->set_reference(desired_com[i]);
                data_out.com_ref_pos = desired_com[i];
            }
        }
        if(cost_mask_[3]) {
            for(size_t i = 0; i < N + 1; i++){
                for (const auto& frame_name : frame_targets_) {
                    const pinocchio::SE3 temp = desired_frames[i][frame_name.first];
                    if(!isContactActive(frame_name.first)){
                        if(i < N) running_cost_model_[i]->changeCostStatus("frame_" + frame_name.first, true);
                        else terminal_cost_model_->changeCostStatus("frame_" + frame_name.first, true);
                        frame_residuals_[i][frame_name.first]->set_reference(temp);
                    }else{
                        // If the frame is in contact, we set the reference to the current position //NOTE: OLD APPROACH
                        // This is useful for frames that are already in contact and we want to keep them there
                        // std::cout<<"Setting reference for frame " << frame_name.first << " to current position in contact." << std::endl;
                        // std::cout<<"Current reference is: " << pinocchio_data_->oMf[model_full_.getFrameId(frame_name.first)] << std::endl;
                        // pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs[0].head(state_->get_nq()));
                        // pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
                        // frame_residuals_[i][frame_name.first]->set_reference(pinocchio_data_->oMf[model_full_.getFrameId(frame_name.first)]);
                    }
                }
            }
        }

        problem_->set_x0(xs_out[0]);

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

        // if(cost_mask_[6]){
        //     for (size_t i = 0; i < N; i++) {
        //         running_cost_model_[i]->changeCostStatus("tauRate", true);
        //     }
        // }

    }else{

        shiftSolution(u_prev_, 1);
        shiftSolution(x_prev_, 1);
                
        pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
        pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);

        data_out.com_curr_pos = pinocchio::centerOfMass(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
        // std::cout << "Current CoM position: " << data_out.com_curr_pos.transpose() << std::endl;

        if(contact_switching_manager_->hasTrueCountGreaterThan(N_horizon_ - knots_lh_)) {
            
            std::cout << "Switching contacts at time: " << ctx_.time << std::endl;
            auto [to_add, to_remove] = contact_switching_manager_->getContactsLists();
            for (const auto& contact : to_add) {
                std::cout << "Contact to add: " << contact << std::endl;
            }

            Eigen::Vector3d current_com = pinocchio::centerOfMass(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
            switchContacts(to_add, to_remove, contact_mask_, xs_out[0], current_com, quasi_static_trigger_);
            quasi_static_trigger_ = false; // reset the first contact change flag
            if(cost_mask_[7]) updatePolytope();
        }
        if(transition_trigger_){
            transition_trigger_ = false;
            quasi_static_trigger_ = true;
            contact_switching_manager_->resetSwitchingMask();
            std::static_pointer_cast<TimeElapsedCondition>(detector_->getCondition("time_elapsed"))->update(ctx_.time + 3.0);
            // detector_->getCondition("plane_pass")->update();
            contact_switching_manager_->getNextPlannedContacts(); //NOTE: this updates the contact status
            std::cout << "Updating time_elapsed at time: " << ctx_.time << std::endl;
        }

        problem_->set_x0(xs_out[0]);

        // if(cost_mask_[6]){
        //     for(size_t i = 0; i < N ; i++){
        //         control_residuals_[i]->set_reference(us_out[0]);
        //     }
        // }

        if(cost_mask_[2]) {
            for(size_t i = 0; i < N + 1; i++){
                com_residual_[i]->set_reference(desired_com[i]);
            }
        }

        if(cost_mask_[3]) {
            for(size_t i = 0; i < N + 1; i++){
                for (const auto& frame_name : frame_targets_) {
                    const pinocchio::SE3 temp = desired_frames[i][frame_name.first];
                    if(!isContactActive(frame_name.first)){
                        frame_residuals_[i][frame_name.first]->set_reference(temp);
                    }else{
                        // pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
                        // pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
                        // frame_residuals_[i][frame_name.first]->set_reference(pinocchio_data_->oMf[model_full_.getFrameId(frame_name.first)]);
                    }
                }
            }
        }

        if(cost_mask_[7]) {

            
            // mpc_utils::computeHalfSpaceRep(mpc_utils::computeConvexHull(getContactPoints(0)), A_, b_); //FIXME: we can compute the polytope for each knot, bur rn, we don't have a mixed contact horizon
            // std::cout << "A_ :\n" << A_ << std::endl;
            // std::cout << "b_ :\n" << b_.transpose() << std::endl;
            // for(size_t i = 0; i < N_horizon_ + 1; i++){
            //     com_polytope_residuals_[i]->setPolytope(A_, b_);
            // }
        }
        // shiftSolution(u_prev_, 5); // I'm using 5 under the assumption that MPC dt=0.025, MuJoCo dt=0.0125, MPC call is each ten steps of MuJoCo
        // shiftSolution(x_prev_, 5);

        // for (auto& x : xs_out) {
        //     x.tail(state_->get_nv()).setZero();
        // }

        // // set the joint reference to the previous solution
        double oblivion_factor = 0.9; //0.8
        auto xReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(running_cost_model_[0]->get_costs().at("xReg")->cost->get_residual());
        xReg_res->set_reference(xs_out[0]);
        for (size_t i = 1; i < N_horizon_; i++) {
            auto xReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(running_cost_model_[i]->get_costs().at("xReg")->cost->get_residual());
            Eigen::VectorXd adjusted_reference = oblivion_factor * xs_out[i] + (1.0 - oblivion_factor) * xs_out[0];
            xReg_res->set_reference(adjusted_reference);
        }
        auto xReg_res_term = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(terminal_cost_model_->get_costs().at("xReg")->cost->get_residual());
        Eigen::VectorXd adjusted_reference_term = oblivion_factor * xs_out[N_horizon_] + (1.0 - oblivion_factor) * xs_out[0];
        xReg_res_term->set_reference(adjusted_reference_term);

        // for (size_t i = 0; i < N_horizon_; i++) {
        //     auto xReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(running_cost_model_[i]->get_costs().at("xReg")->cost->get_residual());
        //     xReg_res->set_reference(x_prev_[i]);
        // }
        // auto xReg_res_term = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(terminal_cost_model_->get_costs().at("xReg")->cost->get_residual());
        // xReg_res_term->set_reference(x_prev_[N_horizon_]);

        // for (size_t i = 0; i < N_horizon_; i++) {
        //     auto xReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(running_cost_model_[i]->get_costs().at("xReg")->cost->get_residual());
        //     xReg_res->set_reference(xs_out[0]);
        // }
        // auto xReg_res_term = std::dynamic_pointer_cast<crocoddyl::ResidualModelState>(terminal_cost_model_->get_costs().at("xReg")->cost->get_residual());
        // xReg_res_term->set_reference(xs_out[0]);

        if (cost_mask_[0]) {
            for (size_t i = 0; i < N_horizon_; i++) {
                auto uReg_res = std::dynamic_pointer_cast<crocoddyl::ResidualModelControl>(running_cost_model_[i]->get_costs().at("uReg")->cost->get_residual());
                uReg_res->set_reference(u_prev_[0]);
            }
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        try {
            fddp_->solve(x_prev_, u_prev_, max_iter_);
            // printCosts();

        } catch (const std::exception& e) {
            std::cerr << "Error during FDDP solve: " << e.what() << std::endl;
            throw;
        }
        getEigenForceFromSolver();
        auto end_time = std::chrono::high_resolution_clock::now();
        solve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Solve duration: " << solve_duration << " ms" << std::endl;

    }
    data_out.b_fddp_feasible = fddp_->get_is_feasible();
    data_out.total_iterations = fddp_->get_iter();

    xs_out = fddp_->get_xs();
    us_out = fddp_->get_us();
    x_prev_ = fddp_->get_xs();
    u_prev_ = fddp_->get_us();
    auto K_new = fddp_->get_K();
    K_old_.resize(K_new.size());
    for (size_t i = 0; i < K_new.size(); ++i) {
        K_old_[i] = K_new[i];
    }
    data_out.K = K_old_[0];

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
        transition_trigger_ = true;
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

void HumanoidMulticontactTracker::quasiStaticMultiContactSolution(
    const VectorXd& q_current,
    const Vector3d& desired_com,
    const std::vector<std::string>& active_contacts,
    const std::vector<double>& alpha,
    std::vector<VectorXd>& tau_guess) const
{
    VectorXd v = VectorXd::Zero(model_full_.nv);
    VectorXd a = VectorXd::Zero(model_full_.nv);
    double mass = 35.115;
    Vector3d gravity = model_full_.gravity981;

    // ---- Parameters from first approach ----
    const double mu = 0.4;                // friction coefficient
    const double min_hand_normal = 120.0;  // N
    const double min_foot_normal = 2.0 * min_hand_normal; // N
    // ----------------------------------------

    int n_contacts = active_contacts.size();
    int base_rows = 6;
    int cols = 3 * n_contacts;  // 3D force per contact
    int rows = base_rows + n_contacts;   // add bias rows

    MatrixXd EE_aug = MatrixXd::Zero(rows, cols);
    VectorXd forces_aug = VectorXd::Zero(rows);

    if (alpha.size() != n_contacts) {
        throw std::invalid_argument("Alpha size must match the number of active contacts.");
    }
    double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
    if (std::abs(alpha_sum - 1.0) > 1e-6) {
        throw std::invalid_argument("Sum of alpha values must be equal to 1. Current sum: " + std::to_string(alpha_sum));
    }

    // Force equilibrium
    for (int i = 0; i < n_contacts; ++i) {
        EE_aug.block<3,3>(0, 3*i) = Matrix3d::Identity();
    }
    forces_aug.segment<3>(0) = -mass * gravity;

    // Moment equilibrium
    for (int i = 0; i < n_contacts; ++i) {
        const std::string& contact_name = active_contacts[i];
        pinocchio::FrameIndex fid = model_full_.getFrameId(contact_name);
        Vector3d pos = pinocchio_data_->oMi[model_full_.frames[fid].parent].translation();
        Vector3d rel_pos = pos - desired_com;
        EE_aug.block<3,3>(3, 3*i) = util::SkewSymmetric(rel_pos);
    }
    forces_aug.segment<3>(3) = Vector3d::Zero();

    // Bias constraint for vertical load distribution
    for (int i = 0; i < n_contacts; ++i) {
        EE_aug(base_rows + i, 3*i + 2) = 1.0;
        forces_aug(base_rows + i) = -alpha[i] * mass * gravity.z();
    }

    // Solve using pseudo-inverse
    auto pInv = EE_aug.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::VectorXd f_guess = pInv * forces_aug;

    // ---- Post-process: enforce min normal + friction cone ----
    for (int i = 0; i < n_contacts; ++i) {
        const std::string& cname = active_contacts[i];
        Vector3d f = f_guess.segment<3>(3*i);

        Vector3d n_world;
        double fn_min;
        if (cname.find("hand") != std::string::npos) {
            if (cname.find("left") != std::string::npos) n_world = Vector3d(0.0, -1.0, 0.0);
            else if (cname.find("right") != std::string::npos) n_world = Vector3d(0.0, 1.0, 0.0);
            fn_min = min_hand_normal;
        } else {
            n_world = Vector3d::UnitZ();
            fn_min = min_foot_normal;
        }

        double fn = n_world.dot(f);
        Vector3d ft = f - fn * n_world;
        double ft_norm = ft.norm();

        std::cout << "\nContact [" << cname << "] raw f: " << f.transpose()
                << "\n  n_world: " << n_world.transpose()
                << "  fn: " << fn << "  ft_norm: " << ft_norm << std::endl;

        bool changed = false;

        if (fn < fn_min) {
            double need = fn_min - fn;
            f += need * n_world;
            fn = n_world.dot(f); // recompute for accuracy
            ft = f - fn * n_world;
            ft_norm = ft.norm();
            changed = true;
            std::cout << "  Increased normal by " << need
                    << " -> new f: " << f.transpose()
                    << " (fn=" << fn << ", ft_norm=" << ft_norm << ")\n";
        } else {
            std::cout << "  Normal OK, no increase applied.\n";
        }

        double tmax = mu * fn;
        if (ft_norm > 1e-12 && ft_norm > tmax) {
            std::cout << "  Tangential " << ft_norm << " exceeds tmax " << tmax << ", clamping\n";
            ft *= (tmax / ft_norm);
            f = fn * n_world + ft;
            changed = true;
            ft_norm = ft.norm(); // recompute after clamp
        } else {
            std::cout << "  Tangential OK (" << ft_norm << " <= " << tmax << "), no clamp applied.\n";
        }

        std::cout << "  Final clamped f: " << f.transpose()
                << " (fn=" << fn << ", ft_norm=" << ft_norm
                << ", tmax=" << tmax << ")\n";

        f_guess.segment<3>(3*i) = f;
    }

    // Dump full vector for cross-check
    std::cout << "\nFinal f_guess (all contacts):\n";
    for (int i = 0; i < n_contacts; ++i) {
        std::cout << "  " << active_contacts[i] << ": "
                << f_guess.segment<3>(3*i).transpose() << "\n";
    }

    // Build fext for Pinocchio
    PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext(model_full_.joints.size(), pinocchio::Force::Zero());
    for (int i = 0; i < n_contacts; ++i) {
        auto fid = model_full_.getFrameId(active_contacts[i]);
        auto jid = model_full_.frames[fid].parent;
        Vector3d force = f_guess.segment<3>(3*i);
        fext[jid] = pinocchio::Force(force, Vector3d::Zero());
    }

    Eigen::VectorXd tau = rnea(model_full_, *pinocchio_data_, q_current, v, a, fext);
    tau_guess[0] = tau.tail(u_prev_[0].size());
}


// void HumanoidMulticontactTracker::quasiStaticMultiContactSolution(const VectorXd& q_current, const Vector3d& desired_com, const std::vector<std::string>& active_contacts, const std::vector<double>& alpha, std::vector<VectorXd>& tau_guess)const {
    
//     VectorXd v = VectorXd::Zero(model_full_.nv);
//     VectorXd a = VectorXd::Zero(model_full_.nv);
//     double mass = 35.115;
//     Vector3d gravity = model_full_.gravity981;

//     int n_contacts = active_contacts.size();
//     int base_rows = 6;
//     int cols = 3 * n_contacts;  // 3D force per contact
//     int rows = base_rows + n_contacts;   // add bias rows

//     MatrixXd EE_aug = MatrixXd::Zero(rows, cols);
//     VectorXd forces_aug = VectorXd::Zero(rows);

//     // Check that alpha size is equal to number of contacts, as we are creating bias rows dynamically
//     if (alpha.size() != n_contacts) {
//         throw std::invalid_argument("Alpha size must match the number of active contacts.");
//     }
//     double alpha_sum = std::accumulate(alpha.begin(), alpha.end(), 0.0);
//     if (std::abs(alpha_sum - 1.0) > 1e-6) {
//         throw std::invalid_argument("Sum of alpha values must be equal to 1. Current sum: " + std::to_string(alpha_sum));
//     }

//     // Force equilibrium
//     for (int i = 0; i < n_contacts; ++i) {
//         EE_aug.block<3,3>(0, 3*i) = Matrix3d::Identity();
//     }
//     forces_aug.segment<3>(0) = -mass * gravity;

//     // Moment equilibrium
//     for (int i = 0; i < n_contacts; ++i) {
//         const std::string& contact_name = active_contacts[i];
//         pinocchio::FrameIndex fid = model_full_.getFrameId(contact_name);
//         Vector3d pos = pinocchio_data_->oMi[model_full_.frames[fid].parent].translation();
//         Vector3d rel_pos = pos - desired_com;
//         EE_aug.block<3,3>(3, 3*i) = util::SkewSymmetric(rel_pos);
//     }
//     forces_aug.segment<3>(3) = Vector3d::Zero();

//     // Add bias constraint
//     for (int i = 0; i < n_contacts; ++i) {
//         EE_aug(base_rows + i, 3*i + 2) = 1.0;
//         forces_aug(base_rows + i) = -alpha[i] * mass * gravity.z();
//     }

//     // Solve using pseudo-inverse
//     auto pInv = EE_aug.completeOrthogonalDecomposition().pseudoInverse();
//     Eigen::VectorXd f_guess = pInv * forces_aug;

//     // Build fext for Pinocchio
//     PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext(model_full_.joints.size(), pinocchio::Force::Zero());
//     for (int i = 0; i < n_contacts; ++i) {
//         const std::string& contact_name = active_contacts[i];
//         auto fid = model_full_.getFrameId(active_contacts[i]);
//         auto jid = model_full_.frames[fid].parent;
//         Vector3d force = f_guess.segment<3>(3*i);
//         fext[jid] = pinocchio::Force(force, Vector3d::Zero()); // Only forces
//     }

//     // Compute torques via RNEA
//     Eigen::VectorXd tau = rnea(model_full_, *pinocchio_data_, q_current, v, a, fext);
//     tau_guess[0] = tau.tail(u_prev_[0].size());
//     std::cout << "tau_guess" <<tau_guess[0].transpose() <<"\n";
// }

void HumanoidMulticontactTracker::quasiStaticFootHandSolution(const VectorXd& q_current,
                                                              const Vector3d& desired_com,
                                                              std::vector<VectorXd>& tau_guess) const {
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

    // std::cout << "Rank of EE_pos_aug: " << EE_pos_aug.fullPivLu().rank() << std::endl;

    auto pInvEE = EE_pos_aug.completeOrthogonalDecomposition().pseudoInverse();
    Matrix<double, 6, 1> forces_ini_guess = pInvEE * forces_aug;

    PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext(model_full_.joints.size(), pinocchio::Force::Zero());
    auto rf = model_full_.getJointId("right_ankle_pitch_joint");
    auto lh = model_full_.getJointId("left_wrist_roll_joint");
    fext[rf] = pinocchio::Force(forces_ini_guess.head(3), Vector3d::Zero());
    fext[lh] = pinocchio::Force(forces_ini_guess.tail(3), Vector3d::Zero());

    // solve for torques
    // tau_guess = (rnea(model_full_, *pinocchio_data_, q_current, v, a, fext)).tail(u_prev_[0].size());
    Eigen::VectorXd tau = rnea(model_full_, *pinocchio_data_, q_current, v, a, fext);
    tau_guess[0] = tau.tail(u_prev_[0].size());

}

void HumanoidMulticontactTracker::quasiStaticSolution(const VectorXd& x_prev, const std::vector<bool>& contact_config, const Vector3d& desired_com, std::vector<VectorXd>& tau_guess){
    if(contact_config == std::vector<bool>{true, true, false, false}){ // left foot and right foot in contact use crocoddyl quasiStatic
        std::cout << "[Crocoddyl] Using crocoddyl quasiStatic solution." << std::endl;
        auto x_prev_new = x_prev;
        x_prev_new.tail(state_->get_nv()) = Eigen::VectorXd::Zero(state_->get_nv());
        std::vector<Eigen::VectorXd> xs(N_horizon_, x_prev_new);
        tau_guess = problem_->quasiStatic_xs(xs);
        // quasiStaticMultiContactSolution(x_prev.head(state_->get_nq()), desired_com, {"left_ankle_roll_link", "right_ankle_roll_link"}, {0.5, 0.5}, tau_guess);
    }else if(contact_config == std::vector<bool>{false, true, true, false}){ // left hand and right foot in contact use heuristic quasiStatic
        std::cout << "[Crocoddyl] Using heuristic quasiStatic solution." << std::endl;
        // quasiStaticFootHandSolution(x_prev.head(state_->get_nq()), desired_com, tau_guess);
        quasiStaticMultiContactSolution(x_prev.head(state_->get_nq()), desired_com, {"right_ankle_roll_link", "left_rubber_hand"}, {0.9, 0.1}, tau_guess);
    }else if(contact_config == std::vector<bool>{true, true, true, false}){
        quasiStaticMultiContactSolution(x_prev.head(state_->get_nq()), desired_com, {"left_ankle_roll_link", "right_ankle_roll_link", "left_rubber_hand"}, {0.5, 0.5, 0.0}, tau_guess);
    }else if(contact_config == std::vector<bool>{true, false, true, false}){
        quasiStaticMultiContactSolution(x_prev.head(state_->get_nq()), desired_com, {"left_ankle_roll_link", "left_rubber_hand"}, {1.0, 0.0}, tau_guess);
    }else if(contact_config == std::vector<bool>{true, false, true, true}){
            quasiStaticMultiContactSolution(x_prev.head(state_->get_nq()), desired_com, {"left_ankle_roll_link", "left_rubber_hand", "right_rubber_hand"}, {0.9, 0.05, 0.05}, tau_guess);
    }else if(contact_config == std::vector<bool>{false, true, true, true}){
            quasiStaticMultiContactSolution(x_prev.head(state_->get_nq()), desired_com, {"right_ankle_roll_link", "left_rubber_hand", "right_rubber_hand"}, {0.9, 0.05, 0.05}, tau_guess);
    }
    else{
        std::cerr<< "shouldn't be here \n";
    }
}

void HumanoidMulticontactTracker::shiftSolution(std::vector<Eigen::VectorXd>& x, const int shift){
    if(shift < 0 || shift >= N_horizon_){
        std::cerr << "Shift value out of bounds. Must be between 0 and " << N_horizon_ - 1 << "." << std::endl;
        return;
    }
    for(int i = 0; i < N_horizon_ - shift; ++i){
        x[i] = x[i + shift];
    }
    for(int i = N_horizon_ - shift; i < N_horizon_; ++i){
        x[i] = x[N_horizon_ - 1]; // fill the rest with the last element
    }
}

Eigen::Vector3d HumanoidMulticontactTracker::computeCoMReference(std::vector<std::string> active_contacts, std::string torso_frame){
    
    for (auto& contact : active_contacts) {
        size_t pos_terminal = contact.find("_terminal");
        if (pos_terminal != std::string::npos) {
            contact.erase(pos_terminal, std::string("_terminal").length());
        }

        size_t pos_contact_contact = contact.find("_contact_contact");
        if (pos_contact_contact != std::string::npos) {
            contact.erase(pos_contact_contact, std::string("_contact").length());
        }
    }

    Eigen::Vector2d support_xy(0.0, 0.0);
    for(const auto& contact : active_contacts){
        pinocchio::FrameIndex fid = model_full_.getFrameId(contact);
        Eigen::Vector3d translation = pinocchio_data_->oMf[fid].translation();
        support_xy += translation.head<2>();
    }
    support_xy /= static_cast<double>(active_contacts.size());

    double delta_z = 0.25; // This is a heuristic value to lower the CoM reference
    pinocchio::FrameIndex fid = model_full_.getFrameId(torso_frame);
    const Eigen::Vector3d torso_pos = pinocchio_data_->oMf[fid].translation();
    double com_z = torso_pos.z() - delta_z;

    return Eigen::Vector3d(support_xy.x(), support_xy.y(), com_z);
}

void HumanoidMulticontactTracker::printActiveSet(const mpc_utils::Phase phase) const {
    if (phase == mpc_utils::Phase::Running) {
        std::cout << "Active costs in running phase:" << std::endl;
        for (size_t i = 0; i < N_horizon_; ++i) {
            std::cout << "Number of active costs in horizon " << i << ": " << running_cost_model_[i]->get_active_set().size() << std::endl;
            std::cout << "Horizon index: " << i << std::endl;
            for (const auto& cost_name : running_cost_model_[i]->get_active_set()) {
                std::cout << " - " << cost_name << std::endl;
            }
        }
    } else if (phase == mpc_utils::Phase::Terminal) {
        std::cout << "Number of active costs in terminal phase: " << terminal_cost_model_->get_active_set().size() << std::endl;
        std::cout << "Active costs in terminal phase:" << std::endl;
        for (const auto& cost_name : terminal_cost_model_->get_active_set()) {
            std::cout << " - " << cost_name << std::endl;
        }
    } else {
        throw std::invalid_argument("Invalid phase specified. Use mpc_utils::Phase::Running or mpc_utils::Phase::Terminal.");
    }
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

void HumanoidMulticontactTracker::printCosts() const {

    double total_cost = fddp_->get_cost();
    std::cout << "\n###\n";
    std::cout << "Total cost: " << total_cost << std::endl;
    std::cout << "\n###\n";
    double max_cost = 0.0;
    std::string max_cost_name;

    for (int i = 0; i < N_horizon_; ++i) {
        for (const auto& cost_pair : running_cost_model_[i]->get_costs()) {
            const std::string& cost_name = cost_pair.first;
            const auto& cost_item = cost_pair.second;
            if (cost_item->active) {
                double cost_value = getCostValue(cost_name, i);
                if (cost_value > max_cost) {
                    max_cost = cost_value;
                    max_cost_name = cost_name;
                }
            }
        }
        std::cout << "Cost with the highest value in the horizon: " << max_cost_name << " with value: " << max_cost << std::endl;
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
