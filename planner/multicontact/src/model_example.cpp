#include "humanoid_multicontact_tracker.hpp"
#include "util/util.hpp"

#include "third_party/sciplot/sciplot.hpp"
using namespace sciplot;

void plotState(const std::vector<Eigen::VectorXd>& xs_out);
void plotControl(const std::vector<Eigen::VectorXd>& us_out);

int main(){
    std::cout <<"---\EXAMPLE START---\n\n";

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
    
    HumanoidMulticontactTracker g1_test(r_file_path, gains, locked_joints_list);
    g1_test.printModel();

    std::vector<Eigen::VectorXd> xs_out;
    std::vector<Eigen::VectorXd> us_out;
    std::vector<Eigen::VectorXd> xs_out_test;
    std::vector<Eigen::VectorXd> us_out_test;
    for(int i=0; i<100; i++){
        g1_test.solveOneStep(xs_out, us_out, Eigen::Vector3d(0., 0., 0.6));
        xs_out_test.push_back(xs_out.front());
        us_out_test.push_back(us_out.front());
    }

    plotState(xs_out_test);
    plotControl(us_out_test);

    std::cout << "u: " << us_out[0].transpose() << std::endl;
    std::cout << "last u:" << us_out[us_out.size()-1].transpose() << std::endl;
    std::cout <<"\n---\nEXAMPLE END\n";
    return 0;
}

void plotState(const std::vector<Eigen::VectorXd>& xs_out){
    
    std::vector<double> time;
    std::vector<std::vector<double>> joints_pos;
    std::vector<std::vector<double>> joints_vel;
    int pos_size = xs_out[0].size() / 2;
    int vel_size = pos_size;
    joints_pos.resize(pos_size);
    joints_vel.resize(vel_size);

    for(size_t i = 0; i < xs_out.size(); ++i) {
        time.push_back(i * 0.1);
    }

    for(size_t i = 0; i < xs_out.size(); ++i) {
        for(size_t j = 0; j < pos_size; ++j) {
            joints_pos[j].push_back(xs_out[i](j));
            joints_vel[j].push_back(xs_out[i](j + pos_size));
        }
    }

    Plot2D plotState;
    plotState.xlabel("time");
    plotState.ylabel("state");
    for(size_t i = 0; i < joints_pos.size(); ++i) {
        plotState.drawCurve(time, joints_pos[i]).label("joint pos " + std::to_string(i));
    }

    Figure fig = {{plotState}};
    Canvas canvas = {{fig}};
    canvas.size(750, 750);
    canvas.show();

}

void plotControl(const std::vector<Eigen::VectorXd>& us_out){
    std::vector<double> time;
    std::vector<std::vector<double>> joints_torques;
    int torque_size = us_out[0].size();
    joints_torques.resize(torque_size);

    for(size_t i = 0; i < us_out.size(); ++i) {
        time.push_back(i * 0.1);
    }

    for(size_t i = 0; i < us_out.size(); ++i) {
        for(size_t j = 0; j < torque_size; ++j) {
            joints_torques[j].push_back(us_out[i](j));
        }
    }

    Plot2D plotControl;
    plotControl.xlabel("time");
    plotControl.ylabel("control");
    for(size_t i = 0; i < joints_torques.size(); ++i) {
        plotControl.drawCurve(time, joints_torques[i]).label("joint torque " + std::to_string(i));
    }

    Figure fig = {{plotControl}};
    Canvas canvas = {{fig}};
    canvas.size(750, 750);
    canvas.show();
}
