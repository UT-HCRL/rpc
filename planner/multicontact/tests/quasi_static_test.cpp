#include <gtest/gtest.h>
#include "humanoid_multicontact_tracker.hpp"

using Eigen::VectorXd;
using Eigen::Vector3d;
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/spatial/force.hpp>
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <numeric>
#include <iostream>

// --- Utility: skew symmetric matrix ---
namespace util {
Eigen::Matrix3d SkewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<     0, -v.z(),  v.y(),
          v.z(),     0, -v.x(),
         -v.y(),  v.x(),     0;
    return S;
}
}

// --- Test fixture ---
class QuasiStaticTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::string robot_path = THIS_COM "/robot_model/g1/g1_29dof_lock_waist.urdf";
        pinocchio::urdf::buildModel(robot_path, pinocchio::JointModelFreeFlyer(), model_full);
        data = pinocchio::Data(model_full);
    }

    pinocchio::Model model_full;
    pinocchio::Data data;
};

Eigen::VectorXd quasiStaticMultiContactSolution(
    const pinocchio::Model& model_full,
    pinocchio::Data& data,
    const Eigen::VectorXd& q_current,
    const Eigen::Vector3d& desired_com,
    const std::vector<std::string>& active_contacts,
    const std::vector<double>& alpha,
    const double mu)
{

    using namespace Eigen;
    VectorXd v = VectorXd::Zero(model_full.nv);
    VectorXd a = VectorXd::Zero(model_full.nv);
    double mass = 35.115;
    Eigen::Vector3d gravity(0, 0, -9.81);

    const double min_hand_normal = 120.0;                               // N
    const double min_foot_normal = 2.0 * min_hand_normal;               // N
    int n_contacts = active_contacts.size();
    int base_rows = 6;
    int cols = 3 * n_contacts;                                          // 3D force per contact
    int rows = base_rows + n_contacts;                                  // add bias rows

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
        pinocchio::FrameIndex fid = model_full.getFrameId(contact_name);
        Vector3d pos = data.oMi[model_full.frames[fid].parent].translation();
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

        if (fn < fn_min) {
            double need = fn_min - fn;
            f += need * n_world;
            fn = n_world.dot(f);
            ft = f - fn * n_world;
            ft_norm = ft.norm();
        }

        double tmax = mu * fn;
        if (ft_norm > 1e-12 && ft_norm > tmax) {
            ft *= (tmax / ft_norm);
            f = fn * n_world + ft;

            ft_norm = ft.norm();
        }

        f_guess.segment<3>(3*i) = f;
    }

    PINOCCHIO_ALIGNED_STD_VECTOR(pinocchio::Force) fext(model_full.joints.size(), pinocchio::Force::Zero());
    for (int i = 0; i < n_contacts; ++i) {
        auto fid = model_full.getFrameId(active_contacts[i]);
        auto jid = model_full.frames[fid].parent;  // joint ID corresponding to the frame
        Vector3d f_world = f_guess.segment<3>(3*i);

        // Transform world force to joint-local frame
        Eigen::Matrix3d R_world_to_joint = data.oMi[jid].rotation().transpose();
        Vector3d f_joint = R_world_to_joint * f_world;

        fext[jid] = pinocchio::Force(f_joint, Vector3d::Zero());
    }

    Eigen::VectorXd tau = rnea(model_full, data, q_current, v, a, fext);

    return f_guess;

}

TEST_F(QuasiStaticTest, ForcesAreInsideCone) {
    Eigen::VectorXd q_current = Eigen::VectorXd::Zero(model_full.nq);
    Eigen::Vector3d desired_com(0.0, 0.0, 0.9);

    pinocchio::forwardKinematics(model_full, data, q_current);
    pinocchio::updateFramePlacements(model_full, data);

    std::vector<std::string> active_contacts = {"l_foot_contact", "left_rubber_hand", "right_rubber_hand"};
    std::vector<double> alpha = {0.9, 0.05, 0.05};
    const double mu = 0.4;

    Eigen::VectorXd forces = quasiStaticMultiContactSolution(model_full, data, q_current, desired_com, active_contacts, alpha, mu);

    for (int i = 0; i < (int)active_contacts.size(); ++i) {
        Eigen::Vector3d f = forces.segment<3>(3*i);
        const std::string& cname = active_contacts[i];

        // Determine contact normal consistent with function
        Eigen::Vector3d n_world;
        if (cname.find("hand") != std::string::npos) {
            if (cname.find("left") != std::string::npos) n_world = Eigen::Vector3d(0.0, -1.0, 0.0);
            else n_world = Eigen::Vector3d(0.0, 1.0, 0.0);
        } else {
            n_world = Eigen::Vector3d::UnitZ();
        }

        double fn = std::abs(n_world.dot(f));
        Eigen::Vector3d ft_vec = f - fn * n_world;
        double ft = ft_vec.norm();

        std::cout << cname << " final f: " << f.transpose()
                  << " | fn: " << fn << ", ft: " << ft << ", mu*fn: " << mu*fn << "\n";

        // Friction cone check
        EXPECT_LE(ft, mu * fn + 1e-8);
    }
}