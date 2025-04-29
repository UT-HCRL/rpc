#pragma once

#include <Eigen/Dense>

#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include <pinocchio/parsers/urdf.hpp>

#include "contact_sequence.hpp"
#include "mpc_utils.hpp"

class HumanoidMulticontactTracker{

    public:
        HumanoidMulticontactTracker(const std::string&, const std::unordered_map<std::string, mpc_utils::Weights>&, const std::vector<int>&);
        ~HumanoidMulticontactTracker() = default;

        void printModel() const;
        
        void setInitialJointConfiguration(const Eigen::VectorXd& q0);
        void setFrames(const std::vector<std::string>& frame_names);

        boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> createMultiFrameActionModel(const std::vector<std::string>& frame_names);
        boost::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> createMultiFrameTerminalActionModel(const std::vector<std::string>& frame_names);


        // BASE MPC OCP DEFS
        void addCoMCost(const double com_tracking_weight);
        void addXBoundCost(const double x_bound_weight);
        void addContactCosts(const std::vector<std::string>& frame_names);
        void addRegularizationCosts(const Eigen::VectorXd& xreg_weights, const double xreg_weight, const double ureg_weight);


        void solveOneStep(std::vector<Eigen::VectorXd>& us_out);
        
    private:

        //### HUMANOID MODEL ###
        bool reduced_model_;
        std::vector<std::string> frame_names_;
        std::vector<int> locked_joints_list_;
        double mu_;

        Eigen::Matrix3d RH_rotation_;
        Eigen::Matrix3d LH_rotation_;
        //######################


        //### PROBLEM FORMULATION ###
        ContactSequence contact_seqs_;
        Eigen::VectorXd q0_;
        Eigen::VectorXd x0_;
        mpc_utils::Weights w_frame_;

        unsigned int T_;
        double dt_;
        int N_horizon_;
        int max_iter_;
        std::unordered_map<std::string, mpc_utils::Weights> cost_weights_;
        std::unordered_map<std::string, mpc_utils::Weights> frame_targets_;

        boost::shared_ptr<crocoddyl::ActivationModelAbstract> xreg_activation_;
        boost::shared_ptr<crocoddyl::CostModelAbstract> xreg_cost_;
        boost::shared_ptr<crocoddyl::CostModelAbstract> ureg_cost_;

        boost::shared_ptr<crocoddyl::CostModelSum> running_cost_model_;
        boost::shared_ptr<crocoddyl::ContactModelMultiple> running_contact_models_;
        //###########################


        //### PINOCCHIO ###
        pinocchio::Model model_full_;
        pinocchio::Model model_;
        std::vector<pinocchio::JointIndex> locked_joints_;
        //#################


        //### CROCODDYL ###
        boost::shared_ptr<crocoddyl::ShootingProblem> problem_;
        boost::shared_ptr<crocoddyl::SolverFDDP> solver_;

        boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation_;
        boost::shared_ptr<crocoddyl::StateMultibody> state_;
        // FIXME: std::shared_ptr<crocoddyl::StateMultibody> state_; //Use this if bump works
        //#################

};
