#include "humanoid_multicontact_tracker.hpp"
#include "util/util.hpp"

int main(){
    std::cout <<"---\nTEST START---\n\n";

    std::string r_file_path = THIS_COM "/robot_model/g1/g1_29dof_lock_waist.urdf";
    std::vector<int> locked_joints_list = {};
    const std::unordered_map<std::string, mpc_utils::Weights> gains = {
        {"torso",  mpc_utils::fromValues(1.0, 5., 0.5, 0.8, 0.8, 0.8)},
        {"feet",   mpc_utils::fromValues(8.0, 8.0, 8.0, 0.00001, 0.00001, 0.00001)},
        {"L_knee", mpc_utils::fromValues(4.0, 4.0, 4.0, 0.00001, 0.00001, 0.00001)},
        {"R_knee", mpc_utils::fromValues(4.0, 4.0, 4.0, 0.00001, 0.00001, 0.00001)},
        {"hands",  mpc_utils::fromValues(2.0, 2.0, 2.0, 0.00001, 0.00001, 0.00001)},
    };
    mpc_utils::printWeights(gains);
    
    std::vector<std::string> frame_names = {"l_foot_contact", "r_foot_contact"};

    HumanoidMulticontactTracker g1_test(r_file_path, gains, locked_joints_list);

    g1_test.printModel();
    g1_test.setInitialJointConfiguration(Eigen::VectorXd::Zero(34));
    g1_test.setFrames(frame_names);

    std::vector<Eigen::VectorXd> us_out;
    int sim_duration = 3;

    while(sim_duration>0){
        
        g1_test.solveOneStep(us_out);
        std::cout << "u: " << us_out[0].transpose() << std::endl;
        //TODO: use control input for simulation, get x0, update x0, recompute control input
        sim_duration--;

    }
    std::cout <<"\n---\nTEST END\n";
    return 0;
}