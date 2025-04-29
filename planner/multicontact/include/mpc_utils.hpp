#pragma once

#include <array>
#include <string_view>
#include <stdexcept>
#include <unordered_map>
#include <Eigen/Dense>

namespace mpc_utils {

    using Weights = Eigen::Matrix<double, 6, 1>;

    inline Weights fromValues(double wp0, double wp1, double wp2, double wo1, double wo2, double wo3) {
        return Weights(wp0, wp1, wp2, wo1, wo2, wo3);
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

} // namespace mpc_utils
