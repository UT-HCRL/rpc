#pragma once

#include <Eigen/Dense>

#include <memory>
#include <memory>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>

#include "contact_sequence.hpp"
#include "mpc_utils.hpp"
#include "util/pkl_utils.hpp"
#include "util/util.hpp"

class CostRecorderCallback;

class HumanoidMulticontactTracker{

    public:
        HumanoidMulticontactTracker(const std::string&, const std::unordered_map<std::string, mpc_utils::Weights>&, const std::vector<int>& = {0}, const bool = false);
        ~HumanoidMulticontactTracker() = default;

        void printModel() const;
        void printWeights() const;
        void setConfigPath(const std::string& config_path){config_path_ = config_path;}

        void loadCostMask();
        void loadContactFrames();
        void loadInitialConfiguration();
        void loadRegularizationWeights();
        void loadBoundWeights();
        void loadCoMWeights();
        void loadTrackingFramesWeights();

        void setInitialJointConfiguration(const Eigen::VectorXd& q0);
        void setFrames(const std::vector<std::string>& frame_names);

        std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> createMultiFrameActionModel(const std::vector<std::string>& frame_names, const int horizon_index = 0);
        std::shared_ptr<crocoddyl::DifferentialActionModelContactFwdDynamics> createMultiFrameTerminalActionModel(const std::vector<std::string>& frame_names);

        double getDt() const { return dt_; }
        int getNhorizon() const { return N_horizon_; }
        int getMaxIter() const { return max_iter_; }

        int getQ0Size() const { return q0_.size(); }
        int getX0Size() const { return x0_.size(); }

        // BASE MPC OCP DEFS
        void addCoMCost(const double com_tracking_weight, const mpc_utils::Phase phase, const int horizon_index = 0);
        void addXBoundCost(const double x_bound_weight, const mpc_utils::Phase phase, const int horizon_index = 0);
        void addContactCosts(const std::vector<std::string>& frame_names, const mpc_utils::Phase phase, const int horizon_index = 0);
        void addRegularizationCosts(const Eigen::VectorXd& xreg_weights, const double xreg_weight, const double ureg_weight, const mpc_utils::Phase phase, const int horizon_index = 0);
        void addFrameTrackingCost(const std::string& frame_name, const mpc_utils::Phase phase, const int horizon_index = 0);

        void initializeSolver();
        void solveOneStep(std::vector<Eigen::VectorXd>& xs_out, std::vector<Eigen::VectorXd>& us_out, mpc_utils::MPCData& data_out, const Eigen::Vector3d& desired_com = Eigen::Vector3d(0., 0., 0.), std::vector<std::unordered_map<std::string, pinocchio::SE3>> desired_frames = {});
        std::vector<std::string> getTargetFrameNames() const {return track_frame_names_;}

    private:

        //### HUMANOID MODEL ###
        bool reduced_model_;
        std::vector<std::string> frame_names_;
        std::vector<std::string> track_frame_names_;
        std::vector<int> locked_joints_list_;
        double mu_;

        Eigen::Matrix3d RH_rotation_;
        Eigen::Matrix3d LH_rotation_;

        Eigen::Vector3d com_reference_;

        std::string config_path_;
        YAML::Node params_;
        //######################


        //### PROBLEM FORMULATION ###
        ContactSequence contact_seqs_;
        Eigen::VectorXd q0_;
        Eigen::VectorXd x0_;
        mpc_utils::Weights w_frame_;

        double dt_;
        int N_horizon_;
        int max_iter_;
        std::unordered_map<std::string, mpc_utils::Weights> cost_weights_; //FIXME: maybe unused, remove
        std::unordered_map<std::string, mpc_utils::Weights2D> contact_weights_;
        std::unordered_map<std::string, double> frame_targets_;
        std::unordered_map<std::string, double> frame_targets_terminal_; //Used for terminal cost frame tracking
    
        std::vector<int> cost_mask_;

        Eigen::VectorXd xreg_weights_;
        double xreg_weight_;
        double ureg_weight_;
        double xbound_weight_;
        double com_tracking_weight_;

        Eigen::VectorXd terminal_xreg_weights_;
        double terminal_xreg_weight_;
        double terminal_ureg_weight_;
        double terminal_xbound_weight_;
        double terminal_com_tracking_weight_;

        std::shared_ptr<crocoddyl::CostModelAbstract> xreg_cost_;
        std::shared_ptr<crocoddyl::CostModelAbstract> ureg_cost_;
        std::shared_ptr<crocoddyl::ActivationModelAbstract> xreg_activation_;

        std::vector<std::shared_ptr<crocoddyl::CostModelSum>> running_cost_model_;
        std::shared_ptr<crocoddyl::ContactModelMultiple> running_contact_models_;

        std::shared_ptr<crocoddyl::CostModelSum> terminal_cost_model_;
        std::shared_ptr<crocoddyl::ContactModelMultiple> terminal_contact_models_;

        std::vector<std::shared_ptr<crocoddyl::ResidualModelCoMPosition>> com_residual_;
        std::vector<std::unordered_map<std::string, std::shared_ptr<crocoddyl::ResidualModelFramePlacement>>> frame_residuals_;
        //###########################


        //### PINOCCHIO ###
        pinocchio::Model model_full_;
        pinocchio::Model model_;
        std::vector<pinocchio::JointIndex> locked_joints_;
        //#################


        //### CROCODDYL ###
        std::shared_ptr<crocoddyl::ShootingProblem> problem_;
        std::shared_ptr<crocoddyl::SolverFDDP> fddp_;

        std::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation_;
        std::shared_ptr<crocoddyl::StateMultibody> state_;
        bool enable_callbacks_;
        std::shared_ptr<CostRecorderCallback> cost_callback_;
        //#################

};
