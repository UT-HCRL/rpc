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

#include <pinocchio/algorithm/model.hpp>

HumanoidMulticontactTracker::HumanoidMulticontactTracker(const std::string& robot_path, const std::unordered_map<std::string, mpc_utils::Weights>& cost_weights, const std::vector<int>& locked_joints_list = {0}) : locked_joints_list_(locked_joints_list) {

    pinocchio::urdf::buildModel(robot_path, pinocchio::JointModelFreeFlyer(), model_full_);

    if(locked_joints_list_.empty()) {
        // std::cout<<"No locked joints, using full model\n";
        reduced_model_ = false;
        boost::shared_ptr<crocoddyl::StateMultibody> state = boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model_full_));
        state_ = boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model_full_));
    }

    else {
        reduced_model_ = true;
        locked_joints_.reserve(locked_joints_list_.size());
        for (const auto& joint : locked_joints_list_) {
            locked_joints_.push_back(joint);
        }
        
        pinocchio::buildReducedModel(model_full_, locked_joints_, Eigen::VectorXd::Zero(model_full_.nq), model_);
        state_ = boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model_));
    }

    actuation_ = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state_);
    running_cost_model_ = boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());
    running_contact_models_ = boost::make_shared<crocoddyl::ContactModelMultiple>(state_, actuation_->get_nu());

    //### Default problem formulation parameters ###
    dt_ = 0.01;
    N_horizon_ = 5;
    max_iter_ = 100;
    T_ = 5e3;  // number of trials
    //###############################################

    //### Default class member init ###
    mu_ = 0.5;

    RH_rotation_ = Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix(); //TODO: Check its the same as util.util.euler_to_rot
    LH_rotation_ = Eigen::AngleAxisd(M_PI / 2, Eigen::Vector3d::UnitX()).toRotationMatrix(); //Check its the same as util.util.euler_to_rot
    cost_weights_ = cost_weights;
    x0_ = Eigen::VectorXd::Zero(state_->get_nx());
    //#################################

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

void HumanoidMulticontactTracker::addCoMCost(const double com_tracking_weight = 1e4){

    pinocchio::Data data(model_full_);
    pinocchio::centerOfMass(model_full_, data, x0_.head(state_->get_nq())); // get only q0_

    Eigen::Vector3d com_reference = data.com[0]; // [0] assume multiple CoM, use initial state as desired

    boost::shared_ptr<crocoddyl::ResidualModelCoMPosition> com_residual = boost::make_shared<crocoddyl::ResidualModelCoMPosition>(state_, com_reference, actuation_->get_nu());

    boost::shared_ptr<crocoddyl::ActivationModelAbstract> com_activation = boost::make_shared<crocoddyl::ActivationModelQuad>(3);
    boost::shared_ptr<crocoddyl::CostModelAbstract> com_cost = boost::make_shared<crocoddyl::CostModelResidual>(state_, com_activation, com_residual);

    running_cost_model_->addCost("CoMTracking", com_cost, com_tracking_weight);

    // boost::shared_ptr<crocoddyl::CostModelAbstract> comCost = boost::make_shared<crocoddyl::CostModelResidual>(state_, boost::make_shared<crocoddyl::ResidualModelCoMPosition>(state_, Eigen::Vector3d::Zero(), actuation_->get_nu()));
    
    // BELOW THERES THE IMPLEM FROM THE BENCHMARK CPP FILES
    // boost::shared_ptr<crocoddyl::CostModelAbstract> goalTrackingCost = 
    //       boost::make_shared<CostModelResidual>(
    //           state, boost::make_shared<ResidualModelFramePlacement>(
    //                      state, model.getFrameId(robotNames.ee_name),
    //                      pinocchio::SE3Tpl<Scalar>(
    //                          Matrix3s::Identity(),
    //                          Vector3s(Scalar(.0), Scalar(.0), Scalar(.4))),
    //                      actuation->get_nu()));
    //   boost::shared_ptr<CostModelAbstract> xRegCost =
    //       boost::make_shared<CostModelResidual>(
    //           state, boost::make_shared<ResidualModelState>(state, default_state,
    //                                                       actuation->get_nu()));
    //   boost::shared_ptr<CostModelAbstract> uRegCost =
    //       boost::make_shared<CostModelResidual>(
    //           state,
    //           boost::make_shared<ResidualModelControl>(state, actuation->get_nu()));

}

void HumanoidMulticontactTracker::addXBoundCost(const double x_bound_weight = 50000.0) {
    Eigen::VectorXd x_lb(state_->get_lb().segment(1, state_->get_nv()).size() + state_->get_lb().tail(state_->get_nv()).size());
    Eigen::VectorXd x_ub(state_->get_ub().segment(1, state_->get_nv()).size() + state_->get_ub().tail(state_->get_nv()).size());
    x_lb << state_->get_lb().segment(1, state_->get_nv()), state_->get_lb().tail(state_->get_nv());
    x_ub << state_->get_ub().segment(1, state_->get_nv()), state_->get_ub().tail(state_->get_nv());
    crocoddyl::ActivationBounds x_bounds(x_lb, x_ub);

    boost::shared_ptr<crocoddyl::ActivationModelAbstract> x_bound_activation = boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(x_bounds);
    boost::shared_ptr<crocoddyl::ResidualModelAbstract> x_bound_residual = boost::make_shared<crocoddyl::ResidualModelState>(state_, actuation_->get_nu());
    boost::shared_ptr<crocoddyl::CostModelAbstract> x_bound_cost = boost::make_shared<crocoddyl::CostModelResidual>(state_, x_bound_activation, x_bound_residual);
    running_cost_model_->addCost("xBounds", x_bound_cost, x_bound_weight);
}

void HumanoidMulticontactTracker::addRegularizationCosts(const Eigen::VectorXd& x_weights, const double xreg_weight = 5e-2, const double ureg_weight = 1e-4) {

    xreg_activation_ = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(x_weights); //NOTE: for some reason in python is **2 (?)
    xreg_cost_ = boost::make_shared<crocoddyl::CostModelResidual>(state_, xreg_activation_, boost::make_shared<crocoddyl::ResidualModelState>(state_, x0_, actuation_->get_nu()));
    ureg_cost_ = boost::make_shared<crocoddyl::CostModelResidual>(state_, boost::make_shared<crocoddyl::ResidualModelControl>(state_, actuation_->get_nu()));
    
    running_cost_model_->addCost("xReg", xreg_cost_, xreg_weight);
    running_cost_model_->addCost("uReg", ureg_cost_, ureg_weight);

}

void HumanoidMulticontactTracker::addContactCosts(const std::vector<std::string>& frame_names){

    for(size_t i = 0; i < frame_names.size(); i++){

        std::string frame_name = frame_names[i];
        boost::shared_ptr<crocoddyl::ContactModelAbstract> support_contact_model6D =
        boost::make_shared<crocoddyl::ContactModel6D>(state_, model_full_.getFrameId(frame_name), pinocchio::SE3::Identity(), actuation_->get_nu(), Eigen::Vector2d(0., 50.)); //NOTE: removed LOCAL_WORLD_ALIGNED
        running_contact_models_->addContact(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_contact", support_contact_model6D);

        //TODO: add hand contact rotation (?)

        Eigen::Matrix3d rotation;
        if(frame_name.find("RH") != std::string::npos){
            rotation = RH_rotation_;
        }
        else if(frame_name.find("LH") != std::string::npos){
            rotation = LH_rotation_;
        }
        else{
            rotation = Eigen::Matrix3d::Identity();
        }

        crocoddyl::FrictionCone surf_cone(rotation, mu_, 4, true);
        crocoddyl::ActivationBounds bounds(surf_cone.get_lb(), surf_cone.get_ub());
        boost::shared_ptr<crocoddyl::ActivationModelAbstract> surf_activation_friction = boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(bounds);
        boost::shared_ptr<crocoddyl::ResidualModelAbstract> surf_residual = boost::make_shared<crocoddyl::ResidualModelContactFrictionCone>(state_, model_full_.getFrameId(frame_name), surf_cone, actuation_->get_nu());
        boost::shared_ptr<crocoddyl::CostModelAbstract> surf_cost = boost::make_shared<crocoddyl::CostModelResidual>(state_, surf_activation_friction, surf_residual);
        running_cost_model_->addCost(model_full_.frames[model_full_.getFrameId(frame_name)].name + "_friction_cone", surf_cost, 1e1);

        w_frame_ = mpc_utils::getFrameGain(frame_name, cost_weights_);
        //TODO: set frame pose, this will come from the trajectory in bezier curve form at time t @carlos
        // pinocchio::SE3 fr_Mref = SE3::Identity();

    }

}

boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> HumanoidMulticontactTracker::createMultiFrameActionModel(const std::vector<std::string>& frame_names){

    addContactCosts(frame_names);

    addCoMCost();//TODO: implement CoM cost

    Eigen::VectorXd xreg_weights = Eigen::VectorXd::Ones(2 * state_->get_nv()); //FIXME: MODIFY WEIGHTS this are taken from gepetto quadruped and prob dont work for G1
    xreg_weights.head<3>().fill(0.);
    xreg_weights.segment<3>(3).fill(pow(500., 2));
    xreg_weights.segment(6, state_->get_nv() - 6).fill(pow(0.01, 2));
    xreg_weights.segment(state_->get_nv(), state_->get_nv()).fill(pow(10., 2));
    addXBoundCost();

    addRegularizationCosts(xreg_weights);

    boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> runningDAM = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, running_contact_models_, running_cost_model_);
    return runningDAM;
}

boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> HumanoidMulticontactTracker::createMultiFrameTerminalActionModel(const std::vector<std::string>& frame_names){
    // TODO: Complete the split for the terminal cost setup
    boost::shared_ptr<crocoddyl::CostModelSum> terminalCostModel = boost::make_shared<crocoddyl::CostModelSum>(state_, actuation_->get_nu());


    boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminalDAM = boost::make_shared<crocoddyl::DifferentialActionModelContactFwdDynamics>(state_, actuation_, running_contact_models_, terminalCostModel);

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

void HumanoidMulticontactTracker::solveOneStep(std::vector<Eigen::VectorXd>& us_out){

    boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> running_DAM = createMultiFrameActionModel(frame_names_);
    boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> terminal_DAM = createMultiFrameTerminalActionModel(frame_names_);

    boost::shared_ptr<crocoddyl::ActionModelAbstract> runningModelWithEuler = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(running_DAM, dt_);
    boost::shared_ptr<crocoddyl::ActionModelAbstract> terminalModelWithEuler = boost::make_shared<crocoddyl::IntegratedActionModelEuler>(terminal_DAM, dt_);

    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> running_models;
    for(std::size_t i = 0; i < N_horizon_; ++i) {
        running_models.push_back(runningModelWithEuler);
    }

    boost::shared_ptr<crocoddyl::ShootingProblem> problem = boost::make_shared<crocoddyl::ShootingProblem>(x0_, running_models, terminalModelWithEuler);
    crocoddyl::SolverFDDP fddp(problem);
    
    const std::size_t N = fddp.get_problem()->get_T();
    std::vector<Eigen::VectorXd> xs(N, x0_);         //NOTE: in python was T+1 maybe to account for initial state, but it doesnt work here
    std::vector<Eigen::VectorXd> us = problem->quasiStatic_xs(xs);
    xs.push_back(x0_);

    std::cout << "NQ: "<< problem->get_terminalModel()->get_state()->get_nq()<< std::endl;
    std::cout << "Number of nodes: " << N << std::endl;

    // Solving the optimal control problem
    Eigen::ArrayXd duration(T_);
    for (unsigned int i = 0; i < T_; i++) {
        crocoddyl::Timer timer;
        fddp.solve(xs, us, max_iter_);
        duration[i] = timer.get_duration();
    }

    double avrg_duration = duration.sum() / T_;
    double min_duration = duration.minCoeff();
    double max_duration = duration.maxCoeff();
    std::cout << "  FDDP.solve [ms]: " << avrg_duration << " (" << min_duration
                << "-" << max_duration << ")" << std::endl;

    // Running calc
    for (unsigned int i = 0; i < T_; ++i) {
        crocoddyl::Timer timer;
        problem->calc(xs, us);
        duration[i] = timer.get_duration();
    }

    x0_ = fddp.get_xs().back(); //FIXME: i should't need this, just update the control input and pass it to the simulation

    us_out = fddp.get_us(); // return the control computed, ill only use us_[0]


}