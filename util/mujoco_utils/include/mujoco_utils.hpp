#pragma once

#include <mujoco/mujoco.h>
#include <unordered_map>
#include <string>
#include <iostream>

namespace mjc_utils{

    /**
     * @brief Maps joint and actuator names from a pinocchio
     *        based model to their corresponding indices in 
     *        the MuJoCo model.
     */
    void SetJointAndActuatorMaps(
        mjModel *m, 
        std::unordered_map<std::string, int> &mj_qpos_map,
        std::unordered_map<std::string, int> &mj_qvel_map,
        std::unordered_map<std::string, int> &mj_actuator_map,
        const std::unordered_map<std::string, int> &pin_joint_map,
        const std::unordered_map<std::string, int> &pin_actuator_map);

    /**
     * @brief Configures the IMU sensors in the MuJoCo model.
     */
    void ConfigureSensors(mjModel *m, int &imu_orientation_adr,
                      int &imu_ang_vel_adr, int &imu_lin_acc_adr,
                      int &imu_lin_vel_adr);


}
