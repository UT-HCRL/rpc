#pragma once

#include <array>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <Eigen/Dense>

namespace mpc_utils {

    enum class Phase { Running, Terminal };

    using Weights = Eigen::Matrix<double, 6, 1>;
    using Weights2D = Eigen::Matrix<double, 2, 1>;
    using Vector6d = Eigen::Matrix<double, 6, 1>;

    inline Weights fromValues(double wp0, double wp1, double wp2, double wo1, double wo2, double wo3) {
        return Weights(wp0, wp1, wp2, wo1, wo2, wo3);
    }

    inline Weights2D from2DValues(double wp0, double wp1) {
        return Weights2D(wp0, wp1);
    }

    inline void printWeights(std::unordered_map<std::string, mpc_utils::Weights> gains){
        for (const auto& gain : gains) {
            std::cout << "Key: " << gain.first << "\nWeights:\n" << gain.second << "\n---\n\n";
        }
    }

    inline std::string matchKey(const std::string& frame_name) {
        using st = std::string;
        const std::array<std::pair<st, st>, 5> match_table = {{
            {"foot",   "feet"},
            {"hand",   "hands"},
            {"R_knee", "R_knee"},
            {"L_knee", "L_knee"},
            {"torso",  "torso"},
        }};

        for (const auto& [pattern, key] : match_table) {
            if (frame_name.find(pattern) != st::npos) {
                return key;
            }
        }
        throw std::runtime_error("No gain key matched for frame: " + frame_name);
    }

    inline Weights getGain(const std::string& key, const std::unordered_map<std::string, Weights>& gain_table) {
        auto it = gain_table.find(key);
        if (it != gain_table.end()) {
            return it->second;
        } else {
            throw std::runtime_error("No weight was set for frame: " + key);
        }
    }

    inline Weights getFrameGain(const std::string& frame_name, const std::unordered_map<std::string, Weights>& gain_table) {
        return getGain(matchKey(frame_name), gain_table);
    }

    struct MPCData{
        int total_iterations;
        std::vector<double> xReg_costs;
        std::vector<double> uReg_costs;
        std::vector<double> xBound_costs;
        std::vector<double> com_costs;
        std::vector<double> left_hand_frame_costs;
        std::vector<double> right_hand_frame_costs;
        std::vector<double> left_ankle_frame_costs;
        std::vector<double> right_ankle_frame_costs;
        std::vector<double> left_knee_frame_costs;
        std::vector<double> right_knee_frame_costs;
        std::vector<double> torso_link_frame_costs; //TODO: this could be a map between string and vector<double> to store costs for each frame
        std::unordered_map<std::string, std::vector<Eigen::Vector3d>> frame_des_pos;
        std::unordered_map<std::string, std::vector<Eigen::Vector3d>> frame_des_ori;
        std::unordered_map<std::string, Eigen::Vector3d> frame_ref_pos;
        std::unordered_map<std::string, Eigen::Vector3d> frame_ref_ori;
        std::unordered_map<std::string, std::vector<double>> frame_costs;
        std::vector<double> left_hand_contact_costs;
        std::vector<double> right_hand_contact_costs;
        std::vector<double> left_foot_contact_costs;
        std::vector<double> right_foot_contact_costs;
        bool b_fddp_feasible;
        float solve_duration;

        std::unordered_map<std::string, Eigen::Vector3d> frame_current_pos;

        Eigen::Vector3d com_ref_pos;
        Eigen::Vector3d com_curr_pos;

        std::unordered_map<std::string, Eigen::Vector3d> contact_forces;

    };

    inline std::pair<bool, Eigen::MatrixXd> calcDARE_old(const Eigen::Ref<const Eigen::MatrixXd>& A,const Eigen::Ref<const Eigen::MatrixXd>& B,const Eigen::Ref<const Eigen::MatrixXd>& Q,const Eigen::Ref<const Eigen::MatrixXd>& R){
        
        const int n = A.rows();
        const int m = B.cols();
        assert(A.cols() == n && Q.rows() == n && Q.cols() == n);
        assert(B.rows() == n && R.rows() == m && R.cols() == m);

        // Precompute B * R^-1 * B^T using LDLT for better stability
        Eigen::LDLT<Eigen::MatrixXd> R_ldlt(R);
        if (R_ldlt.info() != Eigen::Success) {
            std::cerr << "[DARE] ERROR: R is not positive definite.\n";
            return {false, Eigen::MatrixXd()};
        }
        const Eigen::MatrixXd BRBt = B * R_ldlt.solve(B.transpose());

        // Initialization
        Eigen::MatrixXd Ak = A;
        Eigen::MatrixXd Gk = BRBt;
        Eigen::MatrixXd Hk = Q;

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd V, Ak_next, Gk_next, Hk_next;

        constexpr int max_iter = 100;
        constexpr double tol = 1e-6;

        double rel_change = 1e9;
        int iter = 0;
        bool converged = false;

        while (iter < max_iter && rel_change > tol) {
            // V = (I + Gk * Hk)^-1
            V = (I + Gk * Hk).inverse();  // Sherman-Morrison if Gk*Hk is low-rank ??

            Ak_next = Ak * V * Ak;
            Gk_next = Gk + Ak * V * Gk * Ak.transpose();
            Hk_next = Hk + Ak.transpose() * Hk * V * Ak;

            // Use Frobenius norm for performance
            rel_change = (Hk_next - Hk).norm() / (Hk_next.norm() + 1e-12);  // add epsilon for safety

            Ak = Ak_next;
            Gk = Gk_next;
            Hk = Hk_next;
            iter++;
        }

        if (rel_change <= tol) {
            converged = true;
            if (iter < max_iter) {
                std::cout << "[DARE] Riccati solver converged in " << iter << " iterations.\n";
            } else {
                std::cerr << "[DARE] WARNING: Riccati solver reached max iterations. Rel. change = " << rel_change << "\n";
            }
        } else {
            std::cerr << "[DARE] WARNING: Riccati solver did not converge. Rel. change = " << rel_change << "\n";
        }
        
        // Eigen::MatrixXd K = (R + B.transpose() * Hk * B).inverse() * (B.transpose() * Hk * Ak);

        return {converged, Hk};
    }

    inline bool calcDARE(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd Q, Eigen::MatrixXd R, Eigen::MatrixXd& K){
        
        // Based on "Structure-Preserving Algorithms for Periodic Discrete-Time Algebraic Riccati Equations" [https://www.tandfonline.com/doi/full/10.1080/00207170410001714988#d1e289]
        Eigen::MatrixXd Ak = A;
        Eigen::MatrixXd Ak_nxt;
        Eigen::MatrixXd Gk = B * R.inverse() * B.transpose();
        Eigen::MatrixXd Gk_nxt;
        // Eigen::MatrixXd Hk = (K.size() > 0) ? K : Q;
        Eigen::MatrixXd Hk = Q;
        Eigen::MatrixXd Hk_nxt; // H converges quadratically to P

        // Check for NaN values in the initial matrices
        if (A.hasNaN() || B.hasNaN() || Q.hasNaN() || R.hasNaN() || Gk.hasNaN() || Hk.hasNaN()) {
            std::cerr << "[DARE] ERROR: Input matrices contain NaN values.\n";
            return false;
        }

        // Check if Q and R are positive definite
        Eigen::LLT<Eigen::MatrixXd> Q_llt(Q);
        Eigen::LLT<Eigen::MatrixXd> R_llt(R);

        if (Q_llt.info() != Eigen::Success) {
            std::cerr << "[DARE] ERROR: Q is not positive definite.\n";
            return false;
        }

        if (R_llt.info() != Eigen::Success) {
            std::cerr << "[DARE] ERROR: R is not positive definite.\n";
            return false;
        }

        Eigen::MatrixXd V;  // This is computed once to speed up computation, instead of recomputing three times

        int max_iter = 100;
        double tol = 1e-6;
        int curr_iter = 0;

        int rws = Ak.rows();
        int cls = Ak.cols();

        double norm = 1000;

        bool converged = false;

        while(curr_iter < max_iter && norm >= tol){
            
            // Compute next matrixes
            V = (Eigen::MatrixXd::Identity(rws, cls) + Gk * Hk).inverse();
            Ak_nxt = Ak * V *Ak;
            Gk_nxt = Gk + Ak * V * Gk * Ak.transpose();
            Hk_nxt = Hk + Ak.transpose() * Hk * V * Ak;

            // Update norm
            // norm = (Hk_nxt - Hk).lpNorm<Eigen::Infinity>() / Hk_nxt.lpNorm<Eigen::Infinity>();
            norm = (Hk_nxt - Hk).norm() / Hk_nxt.norm(); //Froebenius should be faster
            
            // Update Old
            Ak = Ak_nxt;
            Gk = Gk_nxt;
            Hk = Hk_nxt;
            curr_iter++;

        }

        if (curr_iter >= max_iter && norm > tol) {
            std::cerr << "[DARE] WARNING: Riccati solver did not converge (norm = " << norm << ")\n";
        }else{
            K = Hk;
            converged = true;
        }

        return converged;

    }

    inline std::pair<bool, Eigen::MatrixXd> calcDARE_cholensky(Eigen::MatrixXd A, Eigen::MatrixXd B, Eigen::MatrixXd Q, Eigen::MatrixXd R){

        Eigen::MatrixXd Ak = A;
        Eigen::MatrixXd Gk = B * R.llt().solve(B.transpose());
        Eigen::MatrixXd Hk;
        Eigen::MatrixXd Hk_nxt = Q;
        Eigen::MatrixXd V;
        Eigen::MatrixXd V1;
        Eigen::MatrixXd V2;

        int rws = Ak.rows();
        int cls = Ak.cols();
        int max_iter = 100;
        double tol = 1e-6;
        int curr_iter = 0;

        double norm = 1000;
        bool converged = false;
        
        do{

            Hk = Hk_nxt;

            V = (Eigen::MatrixXd::Identity(rws, cls) + Gk * Hk);
            auto V_solver = V.llt();
            V1 = V_solver.solve(Ak);
            V2 = V_solver.solve(Gk.transpose()).transpose();

            Gk += Ak * V2 * Ak.transpose();
            Hk_nxt = Hk + V1.transpose() * Hk * Ak;
            Ak *= V1;
            
            norm = (Hk_nxt - Hk).norm() / Hk_nxt.norm(); //Froebenius should be faster
            curr_iter++;

        }while(curr_iter < max_iter || norm >= tol);

        if (curr_iter >= max_iter && norm > tol) {
            std::cerr << "[DARE] WARNING: Riccati solver did not converge (norm = " << norm << ")\n";
        }else{
            converged = true;
        }

        return {converged, Hk_nxt};
    }

    inline Vector6d fromPinocchioForce(const pinocchio::Force& pinocchio_force) {

        Vector6d eigen_force;
        eigen_force.head<3>() = pinocchio_force.linear();
        eigen_force.tail<3>() = pinocchio_force.angular();
        return eigen_force;
    }

    //NOTE: these are defined here even if its a simple division, to enable future less naive implementations
    inline void normalize_weights(double& w, const int N){
        if (w < 0.0) {
            throw std::invalid_argument("Weight must be non-negative");
        }
        if (w == 0.0) {
            return; // No normalization needed for zero weight
        }
        if (N <= 0) {
            throw std::invalid_argument("Number of weights must be positive");
        }
        double alpha = 0.2;
        w = w / pow(N, alpha);
    }

    inline void normalize_weights(Weights& w, const int N){
        
        if ((w.array() < 0.0).all()) {
            throw std::invalid_argument("Weight must be non-negative");
        }
        else if((w.array() == 0.0).all()) {
            return;
        }
        
        if (N <= 0) {
            throw std::invalid_argument("Number of weights must be positive");
        }
        
        double alpha = 0.2;
        w = w / pow(N, alpha);
    }
    
    inline void normalize_weights(Eigen::VectorXd& w, const int N){ //TODO: template fun
        
        if ((w.array() < 0.0).all()) {
            throw std::invalid_argument("Weight must be non-negative");
        }
        else if((w.array() == 0.0).all()) {
            return;
        }
        
        if (N <= 0) {
            throw std::invalid_argument("Number of weights must be positive");
        }
        
        double alpha = 0.2;
        w = w / pow(N, alpha);
    }

} // namespace mpc_utils
