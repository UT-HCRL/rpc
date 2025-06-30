#pragma once

#include <conditions/base.hpp>
#include <iostream>

/**
 * @brief Condition that checks if a frame has passed a defined plane.
 * 
 * This condition evaluates whether a specified frame's position has crossed
 * a defined plane, which is represented by a pose and a normal vector.
 * The condition checks if the dot product of the frame's relative position
 * to the plane's position and the plane's normal vector is non-negative,
 * indicating that the frame is on or beyond the plane in the direction of the normal.
 */
class PlanePassCondition : public TransitionCondition {
public:
    PlanePassCondition(const std::string& name, const Eigen::Isometry3d& plane_pose, 
                      const Eigen::Vector3d& plane_normal, const std::string& frame_name)
        : plane_pose_(plane_pose), plane_normal_(plane_normal), frame_name_(frame_name) {
        setName(name);
    }

    bool evaluate(const Context& ctx) const override {
        auto it = ctx.pose.find(frame_name_);
        if (it == ctx.pose.end()) {
            std::cerr << "Frame " << frame_name_ << " not found in context." << std::endl;
            return false;
        }
        
        Eigen::Vector3d frame_position = it->second.translation();
        Eigen::Vector3d plane_position = plane_pose_.translation();
        Eigen::Vector3d relative_position = frame_position - plane_position;
        
        return relative_position.dot(plane_normal_) >= 0;
    }

    void update(const Eigen::Isometry3d& new_plane_pose, const Eigen::Vector3d& new_plane_normal) {
        plane_pose_ = new_plane_pose;
        plane_normal_ = new_plane_normal;
    }

    std::string getName() const override { return name_; }
    std::string getFrameName() const { return frame_name_; }

private:
    Eigen::Isometry3d plane_pose_;
    Eigen::Vector3d plane_normal_;
    std::string frame_name_;
};