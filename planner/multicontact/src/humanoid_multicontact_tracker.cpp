#include "humanoid_multicontact_tracker.hpp"

#include "crocoddyl/core/optctrl/shooting.hpp"
#include "crocoddyl/core/integrator/euler.hpp"
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
#include "crocoddyl/multibody/residuals/com-position.hpp"
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
    mu_ = 0.7;

    RH_rotation_ = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    LH_rotation_ = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    cost_weights_ = cost_weights;
    x0_ = Eigen::VectorXd::Zero(state_->get_nx());
    //#################################

    loadInitialConfiguration();
    loadCostMask();
    loadContactFrames();
    loadRegularizationWeights();
    loadBoundWeights();
    loadCoMWeights();
    loadTrackingFramesWeights();

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

    int nv = state_->get_nv();

    util::ReadParameter(params_["running_costs"]["xReg"], "w", xreg_weight_);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "base_pos", base_pos_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "base_rot", base_rot_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "joint_pos", joint_pos_weights);
    util::ReadParameter(params_["running_costs"]["xReg"]["state_w"], "joint_vel", joint_vel_weights);
    mpc_utils::normalize_weights(xreg_weight_, N_horizon_);

    xreg_weights_ = Eigen::VectorXd::Ones(2 * state_->get_nv());
    for (int i = 0; i < 3; ++i) xreg_weights_(i) = pow(base_pos_weights[i],2);
    for (int i = 0; i < 3; ++i) xreg_weights_(3 + i) = pow(base_rot_weights[i],2);
    for (int i = 6; i < nv; ++i) xreg_weights_(i) = pow(joint_pos_weights,2);
    for (int i = 0; i < nv; ++i) xreg_weights_(nv + i) = pow(joint_vel_weights,2);
    mpc_utils::normalize_weights(xreg_weights_, N_horizon_);

    util::ReadParameter(params_["running_costs"]["uReg"], "w", ureg_weight_);
    mpc_utils::normalize_weights(ureg_weight_, N_horizon_);

    util::ReadParameter(params_["terminal_costs"]["xReg"], "w", terminal_xreg_weight_);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "base_pos", base_pos_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "base_rot", base_rot_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "joint_pos", joint_pos_weights);
    util::ReadParameter(params_["terminal_costs"]["xReg"]["state_w"], "joint_vel", joint_vel_weights);

    terminal_xreg_weights_ = Eigen::VectorXd::Ones(2 * state_->get_nv());
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(i) = pow(base_pos_weights[i],2);
    for (int i = 0; i < 3; ++i) terminal_xreg_weights_(3 + i) = pow(base_rot_weights[i],2);
    for (int i = 6; i < nv; ++i) terminal_xreg_weights_(i) = pow(joint_pos_weights,2);
    for (int i = 0; i < nv; ++i) terminal_xreg_weights_(nv + i) = pow(joint_vel_weights,2);

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
    for (const auto& frame_name : track_frame_names_) {
        double w_frame;
        double wo_frame;
        util::ReadParameter(params_["running_costs"]["tracking_frames"][frame_name], "pos", w_frame);
        util::ReadParameter(params_["running_costs"]["tracking_frames"][frame_name], "rot", wo_frame);
        mpc_utils::normalize_weights(w_frame, N_horizon_);
        frame_targets_[frame_name] = mpc_utils::from2DValues(w_frame, wo_frame);
    }
    for (const auto& frame_name : track_frame_names_) {
        double w_frame;
        double wo_frame;
        std::string frame_name_terminal = frame_name + "_terminal";
        util::ReadParameter(params_["terminal_costs"]["tracking_frames"][frame_name], "pos", w_frame);
        util::ReadParameter(params_["terminal_costs"]["tracking_frames"][frame_name], "rot", wo_frame);
        frame_targets_terminal_[frame_name] = mpc_utils::from2DValues(w_frame, wo_frame);
    }
}

void HumanoidMulticontactTracker::loadMPCParams(){
    util::ReadParameter(params_["mpc"], "dt", dt_);
    util::ReadParameter(params_["mpc"], "horizon", N_horizon_);
    util::ReadParameter(params_["mpc"], "max_iter", max_iter_);
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

    std::cout << "\n\n### Terminal costs weights ###\n" << std::endl;
    std::cout << "terminal_xreg_weights_: " << terminal_xreg_weights_.transpose() << std::endl;
    std::cout << "terminal_ureg_weight_: " << terminal_ureg_weight_ << std::endl;
    std::cout << "terminal_xbound_weight_: " << terminal_xbound_weight_ << std::endl;
    std::cout << "terminal_com_tracking_weight_: " << terminal_com_tracking_weight_ << std::endl;
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

void HumanoidMulticontactTracker::addFrameTrackingCost(const std::string& frame_name, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;
    mpc_utils::Weights2D frame_cost_weight = (phase == mpc_utils::Phase::Running) ? frame_targets_[frame_name] : frame_targets_terminal_[frame_name];

    Eigen::VectorXd q = x0_.head(state_->get_nq());
    pinocchio::forwardKinematics(model_full_, *pinocchio_data_, q);
    pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);

    // pinocchio::SE3 current_pose = data_full_.oMf[model_full_.getFrameId(frame_name)];
    // std::cout << "current_pose which will be desired is: " << current_pose.translation().transpose() << std::endl;
    // std::cout << "current_pose rotation is: " << current_pose.rotation().eulerAngles(0, 1, 2).transpose() << std::endl;

    Eigen::Vector3d torso_pos = {0.0340706, -8.68116e-05, 0.696563};
    Eigen::Vector3d torso_rot = {0, 0, 0};
    pinocchio::SE3 current_pose = pinocchio::SE3(Eigen::AngleAxisd(torso_rot[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(torso_rot[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(torso_rot[2], Eigen::Vector3d::UnitZ()), torso_pos);

    std::shared_ptr<crocoddyl::ResidualModelFramePlacement> frame_residual_ = std::make_shared<crocoddyl::ResidualModelFramePlacement>(state_, model_full_.getFrameId(frame_name), current_pose, actuation_->get_nu());
    frame_residuals_[horizon_index][frame_name] = frame_residual_;

    Eigen::VectorXd temp_weights(6);
    temp_weights << frame_cost_weight(0), frame_cost_weight(0), frame_cost_weight(0), frame_cost_weight(1), frame_cost_weight(1), frame_cost_weight(1);
    std::shared_ptr<crocoddyl::ActivationModelAbstract> frame_activation = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(temp_weights);
    std::shared_ptr<crocoddyl::CostModelAbstract> goal_tracking_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, frame_activation, frame_residuals_[horizon_index][frame_name]);
    cost_model->addCost("frame_" + frame_name, goal_tracking_cost, 1.0);

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

void HumanoidMulticontactTracker::addRegularizationCosts(const Eigen::VectorXd& x_weights, const double xreg_weight = 5e-2, const double ureg_weight = 1e-4, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index) {

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;

    xreg_activation_ = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(x_weights); //NOTE: power is computed in the weights not here
    xreg_cost_ = std::make_shared<crocoddyl::CostModelResidual>(state_, xreg_activation_, std::make_shared<crocoddyl::ResidualModelState>(state_, x0_, actuation_->get_nu()));
    ureg_cost_ = std::make_shared<crocoddyl::CostModelResidual>(state_, std::make_shared<crocoddyl::ResidualModelControl>(state_, actuation_->get_nu()));
    
    cost_model->addCost("xReg", xreg_cost_, xreg_weight);
    cost_model->addCost("uReg", ureg_cost_, ureg_weight);

}

void HumanoidMulticontactTracker::addContactCosts(const std::vector<std::string>& frame_names, const mpc_utils::Phase phase = mpc_utils::Phase::Running, const int horizon_index){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_[horizon_index] : terminal_cost_model_;
    auto& contact_model = (phase == mpc_utils::Phase::Running) ? running_contact_models_[horizon_index] : terminal_contact_models_;

    for (const auto& frame_name : frame_names) {
        if (frame_name.find("hand") != std::string::npos) {
            contact_weights_[frame_name] = Eigen::Vector2d::Zero();
            terminal_contact_weights_[frame_name] = Eigen::Vector2d::Zero();
        }
    }

    for(size_t i = 0; i < frame_names.size(); i++){

        std::string frame_name = frame_names[i];

        pinocchio::SE3 contact_frame_pose = pinocchio::SE3::Identity();

        if(frame_name.find("right_rubber") != std::string::npos){
            // contact_frame_pose.rotation() = Eigen::Matrix3d::Identity();
            contact_frame_pose.rotation() = Eigen::AngleAxisd( M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
        }else if(frame_name.find("left_rubber") != std::string::npos){
            contact_frame_pose.rotation() = Eigen::AngleAxisd( - M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
            // contact_frame_pose.rotation() = Eigen::AngleAxisd( - M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix().transpose();
            // contact_frame_pose.rotation() = Eigen::AngleAxisd( M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
            // contact_frame_pose.rotation() = Eigen::AngleAxisd( M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix().transpose();
            // contact_frame_pose.rotation() = Eigen::Matrix3d::Identity();
        }

        if(phase == mpc_utils::Phase::Running){
            std::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model6D =
            std::make_shared<crocoddyl::ContactModel6D>(state_, model_full_.getFrameId(frame_name), contact_frame_pose, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), contact_weights_[frame_name]);
            contact_model->addContact(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_contact", support_contact_model6D);
        }
        if(phase == mpc_utils::Phase::Terminal){
            std::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model6D =
            std::make_shared<crocoddyl::ContactModel6D>(state_, model_full_.getFrameId(frame_name), contact_frame_pose, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation_->get_nu(), terminal_contact_weights_[frame_name]);
            contact_model->addContact(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_contact_terminal", support_contact_model6D);
        }

        Eigen::Matrix3d rotation;
        if(frame_name.find("right_rubber") != std::string::npos){
            rotation = Eigen::Matrix3d::Identity(); 
            // rotation = Eigen::AngleAxisd( - M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix().transpose();
            // rotation = RH_rotation_;
            std::string contact_suffix;
            contact_suffix = (phase == mpc_utils::Phase::Running) ? "_contact" : "_contact_terminal";
            contact_model->changeContactStatus(model_full_.frames[model_full_.getFrameId(frame_name)].name + contact_suffix, false);
        }
        else if(frame_name.find("left_rubber") != std::string::npos){
            rotation = Eigen::Matrix3d::Identity();
            // rotation = Eigen::AngleAxisd( - M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
            // rotation = Eigen::AngleAxisd( - M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix().transpose();
            // rotation = Eigen::AngleAxisd( M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
            std::cout << "Friction cone Z axis: " << rotation.col(2).transpose() << std::endl;
            // rotation = LH_rotation_;
            std::string contact_suffix;
            contact_suffix = (phase == mpc_utils::Phase::Running) ? "_contact" : "_contact_terminal";
            contact_model->changeContactStatus(model_full_.frames[model_full_.getFrameId(frame_name)].name + contact_suffix, false);
        }
        else{
            rotation = Eigen::Matrix3d::Identity();
        }
        crocoddyl::FrictionCone surf_cone(rotation, mu_, 4, false);
        crocoddyl::ActivationBounds bounds(surf_cone.get_lb(), surf_cone.get_ub());
        std::shared_ptr<crocoddyl::ActivationModelAbstract> surf_activation_friction = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
        std::shared_ptr<crocoddyl::ResidualModelAbstract> surf_residual = std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(state_, model_full_.getFrameId(frame_name), surf_cone, actuation_->get_nu());
        std::shared_ptr<crocoddyl::CostModelAbstract> surf_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, surf_activation_friction, surf_residual);
        cost_model->addCost(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", surf_cost, 1e4);

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
        }
        terminal_contact_models_->changeContactStatus(frame_name + "_contact_terminal", true);
    }
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
            addFrameTrackingCost(frame.first, mpc_utils::Phase::Running, horizon_index);
        }
    }
    if (cost_mask_[4]) {
        addContactCosts(frame_names, mpc_utils::Phase::Running, horizon_index);
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
            addFrameTrackingCost(frame.first, mpc_utils::Phase::Terminal, N_horizon_);
        }
    }
    if (cost_mask_[4]) {
        addContactCosts(frame_names, mpc_utils::Phase::Terminal);
    }
    
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminalDAM = std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, terminal_contact_models_, terminal_cost_model_);
    return terminalDAM;
}

void HumanoidMulticontactTracker::setFrames(const std::vector<std::string>& frame_names){
    frame_names_ = frame_names;
    for (const auto& frame_name : frame_names_) {
        if (model_full_.existFrame(frame_name)) {
            std::cout << "Frame " << frame_name << " exists in the model." << std::endl;
        } else {
            std::cout << "Frame " << frame_name << " does not exist in the model." << std::endl;
        }
    }
}

void HumanoidMulticontactTracker::initializeSolver(){

    std::vector<std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics>> running_DAMS_;
    for(std::size_t i = 0; i < N_horizon_; ++i) {
        running_DAMS_.push_back(createMultiFrameActionModel(frame_names_, i));
    }

    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminal_DAM = createMultiFrameTerminalActionModel(frame_names_);

    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> runningModelsWithEuler;
    for(std::size_t i = 0; i < N_horizon_; ++i) {
        runningModelsWithEuler.push_back(std::make_shared<crocoddyl::IntegratedActionModelEuler>(running_DAMS_[i], dt_));
    }
    std::shared_ptr<crocoddyl::ActionModelAbstract> terminalModelWithEuler = std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_DAM, dt_);

    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models; // TODO move running_models to class property?
    for(std::size_t i = 0; i < N_horizon_; ++i) {
        running_models.push_back(runningModelsWithEuler[i]);
    }

    problem_ = std::make_shared<crocoddyl::ShootingProblem>(x0_, running_models, terminalModelWithEuler);
    fddp_ = std::make_shared<crocoddyl::SolverFDDP>(problem_);
    if(enable_callbacks_) fddp_->setCallbacks({std::make_shared<crocoddyl::CallbackVerbose>()});

    // cost_callback_ = std::make_shared<CostRecorderCallback>();
    // std::vector<std::shared_ptr<crocoddyl::CallbackAbstract>> callbacks;
    // callbacks.push_back(cost_callback_);
    // fddp_->setCallbacks(callbacks);

}

void HumanoidMulticontactTracker::solveOneStep(std::vector<Eigen::VectorXd>& xs_out, std::vector<Eigen::VectorXd>& us_out, mpc_utils::MPCData& data_out, const std::vector<Eigen::Vector3d>& desired_com, std::vector<std::unordered_map<std::string, pinocchio::SE3>> desired_frames, bool contact_trigger) {

    static bool first_iteration = true;
    const std::size_t N = fddp_->get_problem()->get_T();
    // static int iteration = 0;
    // static double avg_solve_time_ = 0.0;

    if(first_iteration){
        std::vector<Eigen::VectorXd> xs(N, x0_);
        std::vector<Eigen::VectorXd> us = problem_->quasiStatic_xs(xs);
        xs.push_back(x0_);
        
        first_iteration = false;

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

        // problem_->calc(xs, us);
        // problem_->calcDiff(xs, us);
        // auto TModel = problem_->get_terminalModel();
        // auto TData = problem_->get_terminalData();
        // Eigen::MatrixXd Q = TData->Lxx;
        // std::cout << "\n\nLxx [The one I set] :\n\n" << Q << std::endl;
        // computeDARE(xs, us);

        // std::cout << "Desired torso frame position: " << desired_frames[0]["torso_primitive_shape"].translation().transpose() << std::endl;
        // std::cout << "Desired ankle left frame position: " << desired_frames[0]["left_ankle_roll_link"].translation().transpose() << std::endl;
        pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
        pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
        // std::cout << "Current torso frame position: " << pinocchio_data_->oMf[model_full_.getFrameId("torso_primitive_shape")].translation().transpose() << std::endl;
        // std::cout << "Current ankle frame position: " << pinocchio_data_->oMf[model_full_.getFrameId("left_ankle_roll_link")].translation().transpose() << std::endl;
        fddp_->setCandidate(xs, us, false);
        fddp_->solve(xs, us, max_iter_);

        // auto contact_forces = getEigenForceFromSolver();
        // const auto& f = contact_forces[0]["l_foot_contact"];
        // std::cout << "l_foot contact force — linear: " << f.head<3>().transpose()
        //           << ", angular: " << f.tail<3>().transpose() << std::endl;
        

    }else{

        if (contact_trigger) {
            std::cout << " HERE IS DELETING CONTACT \n";
            // std::vector<std::string> to_remove = {"l_foot_contact"};
            std::vector<std::string> to_add = {"left_rubber_hand"};
            pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
            pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
            
            // deactivateContacts(to_remove);
            activateContacts(to_add);
        }

        std::vector<Eigen::VectorXd> xs(N, xs_out[0]);
        std::vector<Eigen::VectorXd> us = us_out;
        xs.push_back(xs_out[0]);
        problem_->set_x0(xs[0]);

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
                    }
                }
            }
        }

        fddp_->setCandidate(xs, us, true);
        auto start_time = std::chrono::high_resolution_clock::now();
        fddp_->solve(xs, us, max_iter_);
        getEigenForceFromSolver();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        // std::cout << "Solver duration: " << duration << " ms" << std::endl;
        // avg_solve_time_ = (avg_solve_time_ * iteration + duration) / (iteration + 1);
        // iteration++;
        // if(iteration% 10 == 0) {
        //     std::cout << "Iteration: " << iteration << ", Average solver time: " << avg_solve_time_ << " ms" << std::endl;
        // }
        // std::cout << "Problem is " << (fddp_->get_is_feasible() ? "Feasible" : "Infeasible") 
        //       << ", Iterations: " << fddp_->get_iter() << "\n";

    }
    
    xs_out = fddp_->get_xs();
    us_out = fddp_->get_us();

    pinocchio::forwardKinematics(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));
    pinocchio::updateFramePlacements(model_full_, *pinocchio_data_);
    
    for(int i=0; i<N; i++){
        data_out.xReg_costs.push_back(getCostValue("xReg", i));
        data_out.uReg_costs.push_back(getCostValue("uReg", i));
        data_out.xBound_costs.push_back(getCostValue("xBounds", i));
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
        if(cost_mask_[2]) data_out.com_costs.push_back(getCostValue("CoMTracking", i));
    }

    if(cost_mask_[2]) data_out.com_curr_pos =  pinocchio::centerOfMass(model_full_, *pinocchio_data_, xs_out[0].head(state_->get_nq()));

    if(cost_mask_[3]) {
        for(size_t i = 0; i < N + 1; i++){
            for (const auto& frame_name : frame_targets_) {
                data_out.frame_current_pos[frame_name.first] = pinocchio_data_->oMf[model_full_.getFrameId(frame_name.first)].translation();
            }
        }
    }

    auto contact_forces = getEigenForceFromSolver();
    for(const auto& contacts : running_contact_models_[0]->get_contacts()) {
        // std::cout << "Contact forces for frame: " << contacts.first << std::endl;
        // std::cout << "Linear: " << contact_forces[0][contacts.first].head<3>().transpose() << std::endl;
        // std::cout << "Angular: " << contact_forces[0][contacts.first].tail<3>().transpose() << std::endl;
        data_out.contact_forces[contacts.first] = contact_forces[0][contacts.first].head<3>();
    }

}

double HumanoidMulticontactTracker::getCostValue(const std::string& cost_name, const int horizon_index) const {
    auto running_data = fddp_->get_problem()->get_runningDatas()[horizon_index];
    auto integrated_data = std::dynamic_pointer_cast<crocoddyl::IntegratedActionDataEuler>(running_data);
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
    // Eigen::MatrixXd NN = TData->Lxu; // I am assuming I don't have coupled terms.
    // if (NN.isZero()) {
    //     std::cout << "NN is zero." << std::endl;
    // } else {
    //     std::cout << "NN is not zero." << std::endl;
    // }

    auto start_time = std::chrono::high_resolution_clock::now();
    bool success = mpc_utils::calcDARE(A, B, Q, R, K_DARE_);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    if(success){
        std::cout<<"\n\nOptimal P:\n" << K_DARE_ << std::endl;
        std::cout<<"Duration : " << duration << "ms\n";
    }

}

void HumanoidMulticontactTracker::printContacts() const {
    std::cout << "Active contacts at this step:" << std::endl;
    for(int i=0; i<N_horizon_; i++){
        for (const auto& contact_pair : running_contact_models_[i]->get_contacts()) {
            const std::string& name = contact_pair.first;
            const auto& contact = contact_pair.second;
            if (contact->active) {
                std::cout << "Contact: " << name << " at i: <<" <<i <<" is active." << std::endl;
            } else {
                std::cout << "Contact: " << name << " at i: <<" <<i << " is inactive." << std::endl;
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
            contact_forces[name] = mpc_utils::fromPinocchioForce(contact->f);
        }

        forces_trajectory.push_back(contact_forces);
    }
    
    return forces_trajectory;

}
