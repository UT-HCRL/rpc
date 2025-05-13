#include "mujoco_utils.hpp"

namespace mjc_utils{

    void SetJointAndActuatorMaps(
        mjModel *m, 
        std::unordered_map<std::string, int> &mj_qpos_map,
        std::unordered_map<std::string, int> &mj_qvel_map,
        std::unordered_map<std::string, int> &mj_actuator_map,
        const std::unordered_map<std::string, int> &pin_joint_map,
        const std::unordered_map<std::string, int> &pin_actuator_map) 
    {
        // create mujoco joint map
        for (const auto &[pin_jnt_name, _] : pin_joint_map) {
            int jnt_idx = mj_name2id(m, mjOBJ_JOINT, pin_jnt_name.c_str());
            if (jnt_idx == -1) {
                std::cout << "[MuJoCo Utils] Can't find the joint name: " 
                          << pin_jnt_name << " in MuJoCo model" << '\n';
            }
            mj_qpos_map[pin_jnt_name] = m->jnt_qposadr[jnt_idx];
            mj_qvel_map[pin_jnt_name] = m->jnt_dofadr[jnt_idx];
        }
        // create mujoco actuator map
        for (const auto &[pin_act_name, _] : pin_actuator_map) {
            int act_idx = mj_name2id(m, mjOBJ_ACTUATOR, pin_act_name.c_str());
            if (act_idx == -1) {
                std::cout << "[MuJoCo Utils] Can't find the actuator name: " 
                          << pin_act_name << " in MuJoCo model" << '\n';
            }
            mj_actuator_map[pin_act_name] = act_idx;
        }
    }

    void ConfigureSensors(mjModel *m, int &imu_orientation_adr,
                      int &imu_ang_vel_adr, int &imu_lin_acc_adr,
                      int &imu_lin_vel_adr) {
            // imu orientation sensor
            const std::string &imu_orientation_sensor = "imu-orientation";
            int idx = mj_name2id(m, mjOBJ_SENSOR, imu_orientation_sensor.c_str());
            if (idx == -1)
                std::cout << "[MuJoCo Utils] Can't find the imu orientation sensor: "
                        << imu_orientation_sensor << " in MuJoCo model" << '\n';
            imu_orientation_adr = m->sensor_adr[idx];

            // imu angular velocity sensor (gyro)
            const std::string &imu_ang_vel_sensor = "imu-angular-velocity";
            idx = mj_name2id(m, mjOBJ_SENSOR, imu_ang_vel_sensor.c_str());
            if (idx == -1)
                std::cout << "[MuJoCo Utils] Can't find the imu angular velocity sensor: "
                        << imu_ang_vel_sensor << " in MuJoCo model" << '\n';
            imu_ang_vel_adr = m->sensor_adr[idx];
            // imu_ang_vel_noise_adr = m->sensor_noise[idx];

            // imu linear acceleration sensor (accelerometer)
            const std::string &imu_lin_acc_sensor = "imu-linear-acceleration";
            idx = mj_name2id(m, mjOBJ_SENSOR, imu_lin_acc_sensor.c_str());
            if (idx == -1)
                std::cout
                    << "[MuJoCo Utils] Can't find the imu linear acceleration sensor: "
                    << imu_lin_acc_sensor << " in MuJoCo model" << '\n';
            imu_lin_acc_adr = m->sensor_adr[idx];
            // imu_lin_acc_noise_adr = m->sensor_noise[idx];

            // imu linear velocity sensor (imu frame lin vel)
            const std::string &imu_lin_vel_sensor = "imu-linear-velocity";
            idx = mj_name2id(m, mjOBJ_SENSOR, imu_lin_vel_sensor.c_str());
            if (idx == -1)
                std::cout << "[MuJoCo Utils] Can't find the imu linear velocity sensor: "
                        << imu_lin_vel_sensor << " in MuJoCo model" << '\n';
            imu_lin_vel_adr = m->sensor_adr[idx];
            // imu_frame_lin_acc_noise_adr = m->sensor_noise[idx];
            }
}
