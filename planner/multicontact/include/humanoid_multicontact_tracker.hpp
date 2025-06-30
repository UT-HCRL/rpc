#pragma once

#include <Eigen/Dense>

#include <memory>
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/multibody/model.hpp>

#include "mpc_utils.hpp"
#include "util/pkl_utils.hpp"
#include "util/util.hpp"

#include "contact_switching/core.hpp"

class CostRecorderCallback;

class HumanoidMulticontactTracker{

    public:
        HumanoidMulticontactTracker(const std::string&, const std::unordered_map<std::string, mpc_utils::Weights>&, const std::vector<int>& = {0}, const bool = false);
        ~HumanoidMulticontactTracker() = default;

        void printModel() const;
        void printWeights() const;
        void printContacts() const;
        void setConfigPath(const std::string& config_path){config_path_ = config_path;}

        void loadCostMask();
        void loadContactFrames();
        void loadInitialConfiguration();
        void loadRegularizationWeights();
        void loadBoundWeights();
        void loadCoMWeights();
        void loadTrackingFramesWeights();
        void loadMPCParams();

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
        void addFrameTrackingCost(const std::string& frame_name, const double frame_tracking_weight, const mpc_utils::Phase phase, const int horizon_index = 0);

        void deactivateContacts(const std::vector<std::string>& frame_names);
        void activateContacts(const std::vector<std::string>& frame_names);
        void switchContacts(const std::vector<std::string>& active_frames, const std::vector<std::string>& inactive_frames, std::vector<bool> & contact_mask, const std::vector<Eigen::VectorXd>& xs);

        std::vector<std::vector<std::map<std::string, pinocchio::Force>>> const getForceFromSolver();
        std::vector<std::map<std::string, Eigen::Matrix<double,6,1>>> const getEigenForceFromSolver();
    
        double getCostValue(const std::string& cost_name, const int horizon_index) const;

        bool isContactActive(const std::string& contact_name) const;

        void initializeSolver();

        /**
         * @brief Changes the rotational weights of the ActivationModelWeightedQuad @p cost_name.
         * If the mask has a zero, it uses the contact weight, otherwise it uses the terminal contact weight.
         * @param cost_name Name of the cost to change.
         * @param contact_mask Contact mask to apply.
         */
        void changeWeightedQuadRotWeight(const std::string cost_name, std::vector<bool>& contact_mask);

        void solveOneStep(std::vector<Eigen::VectorXd>& xs_out, std::vector<Eigen::VectorXd>& us_out, mpc_utils::MPCData& data_out, const std::vector<Eigen::Vector3d>& desired_com = {}, std::vector<std::unordered_map<std::string, pinocchio::SE3>> desired_frames = {}, const double time = 0.0);
        std::vector<std::string> getTargetFrameNames() const {return track_frame_names_;}
        
        // Auxiliary functions for DARE computation
        void computeDARE(const std::vector<Eigen::VectorXd>& xs_out, const std::vector<Eigen::VectorXd>& us_out);


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
        Eigen::VectorXd q0_;
        Eigen::VectorXd x0_;
        std::vector<VectorXd> x_prev_;
        std::vector<VectorXd> u_prev_;
        mpc_utils::Weights w_frame_;

        double dt_;
        int N_horizon_;
        int max_iter_;
        std::unordered_map<std::string, mpc_utils::Weights> cost_weights_; //FIXME: maybe unused, remove
        std::unordered_map<std::string, mpc_utils::Weights2D> contact_weights_;
        std::unordered_map<std::string, mpc_utils::Weights2D> terminal_contact_weights_;
        std::unordered_map<std::string, mpc_utils::Weights2D> frame_targets_;
        std::unordered_map<std::string, mpc_utils::Weights2D> frame_targets_terminal_; //Used for terminal cost frame tracking
    
        std::vector<int> cost_mask_;

        Eigen::VectorXd xreg_weights_;
        double xreg_weight_;
        double ureg_weight_;
        double xbound_weight_;
        double com_tracking_weight_;
        double frame_tracking_weight_;
        double tracking_contact_rot_weight_;
        double tracking_swing_rot_weight_;
        double friction_weight_;

        Eigen::VectorXd terminal_xreg_weights_;
        double terminal_xreg_weight_;
        double terminal_ureg_weight_;
        double terminal_xbound_weight_;
        double terminal_com_tracking_weight_;
        double terminal_frame_tracking_weight_;
        double terminal_tracking_contact_rot_weight_;
        double terminal_tracking_swing_rot_weight_;

        std::shared_ptr<crocoddyl::CostModelAbstract> xreg_cost_;
        std::shared_ptr<crocoddyl::CostModelAbstract> ureg_cost_;
        std::shared_ptr<crocoddyl::ActivationModelAbstract> xreg_activation_;

        std::vector<std::shared_ptr<crocoddyl::CostModelSum>> running_cost_model_;
        std::vector<std::shared_ptr<crocoddyl::ContactModelMultiple>> running_contact_models_;

        std::shared_ptr<crocoddyl::CostModelSum> terminal_cost_model_;
        std::shared_ptr<crocoddyl::ContactModelMultiple> terminal_contact_models_;

        std::vector<std::shared_ptr<crocoddyl::ResidualModelCoMPosition>> com_residual_;
        std::vector<std::unordered_map<std::string, std::shared_ptr<crocoddyl::ResidualModelFramePlacement>>> frame_residuals_;
        //###########################


        //### PINOCCHIO ###
        pinocchio::Model model_full_;
        pinocchio::Model model_;
        std::unique_ptr<pinocchio::Data> pinocchio_data_;
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

        //### CONTACT SWITCHING ###
        std::shared_ptr<ContactSwitchingManager> contact_switching_manager_;
        std::shared_ptr<TransitionDetector> detector_;
        std::unique_ptr<ContactTransitionCoordinator> coordinator_;
        ContactSwitchUtils::Context ctx_;

        std::unordered_map<std::string, std::vector<Eigen::Isometry3d>> future_poses_; //FIXME: this is redundant with data_out but need conversion between Isometry and PinocchioSE3
        std::vector<bool> contact_mask_;
        bool switch_trigger_ = false;
        //##########################

        //### DARE TEMP VARIABLES ###
        Eigen::MatrixXd K_DARE_;


        //### FUNC UTILS ###
        void printModelContacts() const;

};
