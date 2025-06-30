#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <Eigen/Geometry>

namespace ContactSwitchUtils {

    /**
     * @brief Context structure for contact switching.
     * 
     * This structure holds the current time, poses of frames, and contact statuses.
     * It is used to evaluate transition conditions and manage contact switching logic.
     */
    struct Context {
        double time;
        std::unordered_map<std::string, Eigen::Isometry3d> pose;
        std::vector<bool> contact_status;
    };

    /**
     * @brief Structure representing a transition event.
     */
    struct TransitionEvent {
        int triggered_knot;
        std::string condition_name;
        std::string frame_name;
        double time;
    };

}