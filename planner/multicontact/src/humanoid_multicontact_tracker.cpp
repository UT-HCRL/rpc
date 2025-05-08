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

HumanoidMulticontactTracker::HumanoidMulticontactTracker(const std::string& robot_path, const std::unordered_map<std::string, mpc_utils::Weights>& cost_weights, const std::vector<int>& locked_joints_list = {0}) : locked_joints_list_(locked_joints_list) {

    pinocchio::urdf::buildModel(robot_path, pinocchio::JointModelFreeFlyer(), model_full_);

    if(locked_joints_list_.empty()) {
        // std::cout<<"No locked joints, using full model\n";
        reduced_model_ = false;
        std::shared_ptr<crocoddyl::StateMultibody> state = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_full_));
        state_ = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_full_));
    }

    else {
        reduced_model_ = true;
        locked_joints_.reserve(locked_joints_list_.size());
        for (const auto& joint : locked_joints_list_) {
            locked_joints_.push_back(joint);
        }
        
        pinocchio::buildReducedModel(model_full_, locked_joints_, Eigen::VectorXd::Zero(model_full_.nq), model_);
        state_ = std::make_shared<crocoddyl::StateMultibody>(std::make_shared<pinocchio::Model>(model_));
    }

    actuation_ = std::make_shared<crocoddyl::ActuationModelFloatingBase>(state_);
    running_cost_model_ = std::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());
    running_contact_models_ = std::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());
    terminal_contact_models_ = std::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());

    terminal_cost_model_ = std::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());

    //### Default problem formulation parameters ###
    dt_ = 0.02;
    N_horizon_ = 2;
    max_iter_ = 100;
    T_ = 5e3;  // number of trials
    //###############################################

    //### Default class member init ###
    mu_ = 0.9;

    RH_rotation_ = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    LH_rotation_ = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix();
    //FOOT_rotation = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    cost_weights_ = cost_weights;
    x0_ = Eigen::VectorXd::Zero(state_->get_nx());
    //#################################

    config_path_ = THIS_COM "config/g1/sim/mujoco/ihwbc/crocoddyl_params.yaml";
    params_ = YAML::LoadFile(config_path_);
    loadInitialConfiguration();
    loadContactFrames();
    loadRegularizationWeights();
    loadBoundWeights();
    loadCoMWeights();
    loadTrackingFramesWeights();

    initializeSolver();

}

void HumanoidMulticontactTracker::loadInitialConfiguration() {
    
    std::vector<std::string> joint_names;
    util::ReadParameter(params_, "joint_names", joint_names);
    for(size_t i=0; i < joint_names.size(); i++){
        std::cout << "joint_names[" << i << "]: " << joint_names[i] << std::endl;
    }

    Eigen::VectorXd q0 = Eigen::VectorXd::Zero(joint_names.size());
    for (int i = 0; i < joint_names.size(); i++) {
        double joint_value = 0.0;
        util::ReadParameter(params_["initial_config"], joint_names[i], joint_value);
        q0[i] = joint_value;
    }
    // std::cout << "THE ONE IM PASSING -- q0: " << q0.transpose() << std::endl;
    setInitialJointConfiguration(q0);

}

void HumanoidMulticontactTracker::loadContactFrames(){

    std::vector<std::string> contact_frames;
    util::ReadParameter(params_, "contact_frames", contact_frames);

    setFrames(contact_frames);

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

    xreg_weights_ = Eigen::VectorXd::Ones(2 * state_->get_nv());
    for (int i = 0; i < 3; ++i) xreg_weights_(i) = pow(base_pos_weights[i],2);
    for (int i = 0; i < 3; ++i) xreg_weights_(3 + i) = pow(base_rot_weights[i],2);
    for (int i = 6; i < nv; ++i) xreg_weights_(i) = pow(joint_pos_weights,2);
    for (int i = 0; i < nv; ++i) xreg_weights_(nv + i) = pow(joint_vel_weights,2);

    std::cout << "xreg_weights_: " << xreg_weights_.transpose() << std::endl;
    std::cout <<"total xreg_weight: " << xreg_weight_ << std::endl;

    util::ReadParameter(params_["running_costs"]["uReg"], "w", ureg_weight_);
    std::cout << "ureg_weight_: " << ureg_weight_ << std::endl;

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

    std::cout << "terminal_xreg_weights_: " << terminal_xreg_weights_.transpose() << std::endl;
    std::cout <<"terminal_total xreg_weight: " << terminal_xreg_weight_ << std::endl;

    util::ReadParameter(params_["terminal_costs"]["uReg"], "w", terminal_ureg_weight_);
    std::cout << "terminal_ureg_weight_: " << terminal_ureg_weight_ << std::endl;

    // old native eigen pop version
    // xreg_weights_ = Eigen::VectorXd::Ones(2 * state_->get_nv());
    // xreg_weights_.head<3>().fill(0.);
    // xreg_weights_.segment<3>(3).fill(pow(500., 2));
    // xreg_weights_.segment(6, state_->get_nv() - 6).fill(pow(0.01, 2));
    // xreg_weights_.segment(state_->get_nv(), state_->get_nv()).fill(pow(10., 2));

}

void HumanoidMulticontactTracker::loadBoundWeights(){
    util::ReadParameter(params_["running_costs"]["xBound"], "w", xbound_weight_);
    std::cout << "xbound_weight_: " << xbound_weight_ << std::endl;

    util::ReadParameter(params_["terminal_costs"]["xBound"], "w", terminal_xbound_weight_);
    std::cout << "terminal_xbound_weight_: " << terminal_xbound_weight_ << std::endl;

}

void HumanoidMulticontactTracker::loadCoMWeights(){
    util::ReadParameter(params_["running_costs"]["CoM"], "w", com_tracking_weight_);
    std::cout << "com_tracking_weight_: " << com_tracking_weight_ << std::endl;

    util::ReadParameter(params_["terminal_costs"]["CoM"], "w", terminal_com_tracking_weight_);
    std::cout << "terminal_com_tracking_weight_: " << terminal_com_tracking_weight_ << std::endl;

}

void HumanoidMulticontactTracker::loadTrackingFramesWeights(){
    std::vector<std::string> track_frame_names;
    util::ReadParameter(params_, "tracking_frames", track_frame_names);
    for (const auto& frame_name : track_frame_names) {
        double w_frame;
        util::ReadParameter(params_["running_costs"]["tracking_frames"], frame_name, w_frame);
        frame_targets_[frame_name] = w_frame;
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

void HumanoidMulticontactTracker::addCoMCost(const double com_tracking_weight = 1e4, const mpc_utils::Phase phase = mpc_utils::Phase::Running){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_ : terminal_cost_model_;
    pinocchio::Data data(model_full_);
 
    com_reference_ << pinocchio::centerOfMass(model_full_, data, x0_.head(state_->get_nq())); // get only q0_  // 0, 0, 0.8; 
    // std::cout<<"DESIRED COM: "<< com_reference_.transpose() << std::endl; //0, 0, 0.8; //

    com_residual_ = std::make_shared<crocoddyl::ResidualModelCoMPosition>(state_, com_reference_, actuation_->get_nu());

    std::shared_ptr<crocoddyl::ActivationModelAbstract> com_activation = std::make_shared<crocoddyl::ActivationModelQuad>(3);
    std::shared_ptr<crocoddyl::CostModelAbstract> com_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, com_activation, com_residual_);

    cost_model->addCost("CoMTracking", com_cost, com_tracking_weight);

    // std::shared_ptr<crocoddyl::CostModelAbstract> comCost = std::make_shared<crocoddyl::CostModelResidual>(state_, std::make_shared<crocoddyl::ResidualModelCoMPosition>(state_, Eigen::Vector3d::Zero(), actuation_->get_nu()));

}

void HumanoidMulticontactTracker::addFrameTrackingCost(const std::string& frame_name, const mpc_utils::Phase phase = mpc_utils::Phase::Running){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_ : terminal_cost_model_;

    pinocchio::Data data_full_(model_full_);
    Eigen::VectorXd q = x0_.head(state_->get_nq());
    pinocchio::forwardKinematics(model_full_, data_full_, q);
    pinocchio::updateFramePlacements(model_full_, data_full_);

    // pinocchio::SE3 current_pose = data_full_.oMf[model_full_.getFrameId(frame_name)];
    // std::cout << "current_pose which will be desired is: " << current_pose.translation().transpose() << std::endl;
    // std::cout << "current_pose rotation is: " << current_pose.rotation().eulerAngles(0, 1, 2).transpose() << std::endl;

    Eigen::Vector3d torso_pos = {0.0340706, -8.68116e-05, 0.696563};
    Eigen::Vector3d torso_rot = {0, 0, 0};

    pinocchio::SE3 current_pose = pinocchio::SE3(Eigen::AngleAxisd(torso_rot[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(torso_rot[1], Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(torso_rot[2], Eigen::Vector3d::UnitZ()), torso_pos);

    std::shared_ptr<crocoddyl::CostModelAbstract> goal_tracking_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, std::make_shared<crocoddyl::ResidualModelFramePlacement>(state_, model_full_.getFrameId(frame_name), current_pose, actuation_->get_nu()));
    cost_model->addCost("frame_"+frame_name, goal_tracking_cost, frame_targets_[frame_name]);
}

void HumanoidMulticontactTracker::addXBoundCost(const double x_bound_weight = 50000.0, const mpc_utils::Phase phase = mpc_utils::Phase::Running) {

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_ : terminal_cost_model_;

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

void HumanoidMulticontactTracker::addRegularizationCosts(const Eigen::VectorXd& x_weights, const double xreg_weight = 5e-2, const double ureg_weight = 1e-4, const mpc_utils::Phase phase = mpc_utils::Phase::Running) {

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_ : terminal_cost_model_;

    xreg_activation_ = std::make_shared<crocoddyl::ActivationModelWeightedQuad>(x_weights); //NOTE: power is computed in the weights not here
    xreg_cost_ = std::make_shared<crocoddyl::CostModelResidual>(state_, xreg_activation_, std::make_shared<crocoddyl::ResidualModelState>(state_, x0_, actuation_->get_nu()));
    ureg_cost_ = std::make_shared<crocoddyl::CostModelResidual>(state_, std::make_shared<crocoddyl::ResidualModelControl>(state_, actuation_->get_nu()));
    
    cost_model->addCost("xReg", xreg_cost_, xreg_weight);
    cost_model->addCost("uReg", ureg_cost_, ureg_weight);

}

void HumanoidMulticontactTracker::addContactCosts(const std::vector<std::string>& frame_names, const mpc_utils::Phase phase = mpc_utils::Phase::Running){

    auto& cost_model = (phase == mpc_utils::Phase::Running) ? running_cost_model_ : terminal_cost_model_;
    auto& contact_model = (phase == mpc_utils::Phase::Running) ? running_contact_models_ : terminal_contact_models_;

    for(size_t i = 0; i < frame_names.size(); i++){

        std::string frame_name = frame_names[i];
        std::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model6D =
        std::make_shared<crocoddyl::ContactModel6D>(state_, model_full_.getFrameId(frame_name), pinocchio::SE3::Identity(), actuation_->get_nu(), Eigen::Vector2d(10., 50.0)); //NOTE: this croc version doesn't have LOCAL_WORLD_ALIGNED
        contact_model->addContact(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_contact", support_contact_model6D);

        //TODO: add hand contact rotation if needed

        Eigen::Matrix3d rotation;
        if(frame_name.find("RH") != std::string::npos){
            rotation = RH_rotation_;
        }
        else if(frame_name.find("LH") != std::string::npos){
            rotation = LH_rotation_;
        }
        else{

            // Eigen::Matrix3d R_flip;
            // R_flip = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX());
            // std::cout<<"Rotating Z- with Z+ for "<< frame_name << std::endl;
            // rotation = R_flip;//
            rotation = Eigen::Matrix3d::Identity();
        }
        crocoddyl::FrictionCone surf_cone(rotation, mu_, 4, false);
        crocoddyl::ActivationBounds bounds(surf_cone.get_lb(), surf_cone.get_ub());
        std::shared_ptr<crocoddyl::ActivationModelAbstract> surf_activation_friction = std::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
        std::shared_ptr<crocoddyl::ResidualModelAbstract> surf_residual = std::make_shared<crocoddyl::ResidualModelContactFrictionCone>(state_, model_full_.getFrameId(frame_name), surf_cone, actuation_->get_nu());
        std::shared_ptr<crocoddyl::CostModelAbstract> surf_cost = std::make_shared<crocoddyl::CostModelResidual>(state_, surf_activation_friction, surf_residual);
        cost_model->addCost(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", surf_cost, 1e1);


        // w_frame_ = mpc_utils::getFrameGain(frame_name, cost_weights_);
        //TODO: set frame pose, this will come from the trajectory in bezier curve form at time t @carlos
        // pinocchio::SE3 fr_Mref = SE3::Identity();

    }

}

std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> HumanoidMulticontactTracker::createMultiFrameActionModel(const std::vector<std::string>& frame_names){

    // Debug for feet frames
    Eigen::VectorXd q = pinocchio::neutral(model_full_);
    pinocchio::Data data(model_full_);
    std::string frame_name = "l_foot_contact";
    pinocchio::FrameIndex frame_id = model_full_.getFrameId(frame_name);
    pinocchio::forwardKinematics(model_full_, data, q);
    pinocchio::updateFramePlacement(model_full_, data, frame_id);
    Eigen::Matrix3d R = data.oMf[frame_id].rotation();
    std::cout << "Orientation of frame '" << frame_name << "':\n" << R << std::endl;
    std::cout << "Z-axis in world frame: " << R.col(2).transpose() << std::endl;

    addContactCosts(frame_names);
    // addCoMCost(com_tracking_weight_);
    for(const auto frame_names : frame_targets_){
        addFrameTrackingCost(frame_names.first);
    }
    addXBoundCost(xbound_weight_);
    addRegularizationCosts(xreg_weights_, xreg_weight_, ureg_weight_);

    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> runningDAM = std::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, running_contact_models_, running_cost_model_);
    return runningDAM;
}

std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> HumanoidMulticontactTracker::createMultiFrameTerminalActionModel(const std::vector<std::string>& frame_names){

    addContactCosts(frame_names, mpc_utils::Phase::Terminal);
    // addCoMCost(terminal_com_tracking_weight_, mpc_utils::Phase::Terminal);
    for(const auto frame_names : frame_targets_){
        addFrameTrackingCost(frame_names.first, mpc_utils::Phase::Terminal);
    }
    addXBoundCost(terminal_xbound_weight_, mpc_utils::Phase::Terminal);
    addRegularizationCosts(terminal_xreg_weights_, terminal_xreg_weight_, terminal_ureg_weight_, mpc_utils::Phase::Terminal);
    
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

    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> running_DAM = createMultiFrameActionModel(frame_names_);
    std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminal_DAM = createMultiFrameTerminalActionModel(frame_names_);

    std::shared_ptr<crocoddyl::ActionModelAbstract> runningModelWithEuler = std::make_shared<crocoddyl::IntegratedActionModelEuler>(running_DAM, dt_);
    std::shared_ptr<crocoddyl::ActionModelAbstract> terminalModelWithEuler = std::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_DAM, dt_);

    std::vector<std::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
    for(std::size_t i = 0; i < N_horizon_; ++i) {
        running_models.push_back(runningModelWithEuler);
    }

    problem_ = std::make_shared<crocoddyl::ShootingProblem>(x0_, running_models, terminalModelWithEuler);
    fddp_ = std::make_shared<crocoddyl::SolverFDDP>(problem_);
}

void HumanoidMulticontactTracker::solveOneStep(std::vector<Eigen::VectorXd>& xs_out, std::vector<Eigen::VectorXd>& us_out, const Eigen::Vector3d& desired_com = Eigen::Vector3d(0., 0., 0.6)){

    static bool first_iteration = true;
    const std::size_t N = fddp_->get_problem()->get_T();

    if(first_iteration){
        
        std::vector<Eigen::VectorXd> xs(N, x0_);
        std::vector<Eigen::VectorXd> us = problem_->quasiStatic_xs(xs);
        xs.push_back(x0_);

        fddp_->solve(xs, us, max_iter_);

        us_out = fddp_->get_us(); // return the control computed, ill only use us_[0]
        xs_out = fddp_->get_xs(); // return the state computed
        first_iteration = false;
        // for (std::size_t i = 0; i < xs.size(); ++i) {
        // const Eigen::VectorXd& x = xs[i];
        // const Eigen::VectorXd& q = x.head(model_full_.nq);
        // pinocchio::Data data(model_full_);
        // pinocchio::centerOfMass(model_full_, data, q);
        // Eigen::Vector3d com = data.com[i];
        
        // Eigen::Vector3d com_error = com - com_reference_;
        
        // std::cout << "t[" << i << "]: CoM error = "
        //         << com_error.transpose()
        //         << " (norm = " << com_error.norm() << ")" << std::endl;
        // }
    }
    else{

        std::vector<Eigen::VectorXd> xs(N, xs_out[0]);
        std::vector<Eigen::VectorXd> us = us_out;
        xs.push_back(xs_out[0]);
        // com_residual_->set_reference(desired_com);
        problem_->set_x0(xs[0]);
        // fddp_->setCallbacks({std::make_shared<crocoddyl::CallbackVerbose>()});
        fddp_->solve(xs, us, max_iter_);

        xs_out = fddp_->get_xs(); // return the state computed
        us_out = fddp_->get_us(); // return the control computed, ill only use us_[0]

        // for (std::size_t i = 0; i < xs.size(); ++i) {
        //     const Eigen::VectorXd& x = xs[i];
        //     const Eigen::VectorXd& q = x.head(model_full_.nq);
        //     pinocchio::Data data(model_full_);
        //     pinocchio::centerOfMass(model_full_, data, q);
        //     Eigen::Vector3d com = data.com[i];
            
        //     Eigen::Vector3d com_error = com - com_reference_;
            
        //     std::cout << "t[" << i << "]: CoM error = "
        //             << com_error.transpose()
        //             << " (norm = " << com_error.norm() << ")" << std::endl;
        // }

        Eigen::VectorXd torso_expected;
        torso_expected.resize(3);
        torso_expected << 0, 0, 0.704;
        for (std::size_t i = 0; i < xs_out.size(); ++i) {
            const Eigen::VectorXd& x = xs_out[i];
            const Eigen::VectorXd& q = x.head(model_full_.nq);
            pinocchio::Data data_iter(model_full_);
            pinocchio::forwardKinematics(model_full_, data_iter, q);
            pinocchio::updateFramePlacements(model_full_, data_iter);

            pinocchio::SE3 current_pose_iter = data_iter.oMf[model_full_.getFrameId("torso_link")];
            Eigen::Vector3d position_error_iter = current_pose_iter.translation() - torso_expected;
            std::cout << "Iteration " << i << " - Position error: " << position_error_iter.transpose() << std::endl;
        }
    }
}
