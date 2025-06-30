#pragma once

#include <vector>
#include <unordered_map>
#include <string>

#include "cs_utils.hpp"

using namespace ContactSwitchUtils;

/**
 * @brief Transition detector that evaluates conditions against a context.
 * 
 * This class manages a collection of transition conditions and provides methods
 * to detect transitions based on the current context. It can also look ahead
 * to future contexts to detect transitions based on future contexts.
 */
class TransitionDetector {
public:
    void addCondition(std::shared_ptr<TransitionCondition> condition) {
        conditions_.push_back(condition);
    }

    void clearConditions() {
        conditions_.clear();
    }

    std::vector<std::shared_ptr<TransitionCondition>> getConditions() const {
        return conditions_;
    }

    std::shared_ptr<TransitionCondition> getCondition(const std::string& name) const {
        for (const auto& condition : conditions_) {
            if (condition->getName() == name) {
                return condition;
            }
        }
        return nullptr; // Not found
    }

    // Check current conditions
    std::vector<TransitionEvent> detectTransitions(const Context& ctx) const {
        std::vector<TransitionEvent> events;
        
        for (const auto& condition : conditions_) {
            if (condition->evaluate(ctx)) {
                TransitionEvent event;
                event.triggered_knot = 0;
                event.condition_name = condition->getName();
                event.time = ctx.time;
                
                // Extract frame name if it's a PlanePassCondition
                if (auto plane_condition = std::dynamic_pointer_cast<PlanePassCondition>(condition)) {
                    event.frame_name = plane_condition->getFrameName();
                }
                
                events.push_back(event);
            }
        }
        
        return events;
    }

    // Transition detection for MPC knots
    std::vector<TransitionEvent> detectFutureTransitions(
        const Context& current_ctx,
        const std::unordered_map<std::string, std::vector<Eigen::Isometry3d>>& future_poses,
        double dt, int num_knots) const {
        
        std::vector<TransitionEvent> events;
        
        for (const auto& [frame, poses] : future_poses) {
            if (poses.size() != num_knots) {
                throw std::invalid_argument("Size of future poses for frame " + frame + " does not match the number of knots.");
            }
        }

        // Check each knot
        for (int knot = 0; knot < num_knots; ++knot) {
            bool condition_trigger = false;
            Context future_ctx;
            future_ctx.time = current_ctx.time + knot * dt;
            future_ctx.contact_status = current_ctx.contact_status;
            
            // Update poses for this time step
            for (const auto& [frame, poses] : future_poses) {
                future_ctx.pose[frame] = poses[knot];
            }
            
            // Check conditions
            for (const auto& condition : conditions_) {
                if (condition->evaluate(future_ctx)) {
                    condition_trigger = true;
                    TransitionEvent event;
                    event.triggered_knot = knot;
                    event.condition_name = condition->getName();
                    event.time = future_ctx.time;
                    
                    if (auto plane_condition = std::dynamic_pointer_cast<PlanePassCondition>(condition)) {
                        event.frame_name = plane_condition->getFrameName();
                    }
                    
                    events.push_back(event);
                    break; // Only first trigger per knot
                }
            }
            if(condition_trigger) {
                break; // Stop checking further conditions if one is triggered, as we only want the first knot that triggers transition
            }
        }

        return events;
    }

private:
    std::vector<std::shared_ptr<TransitionCondition>> conditions_;
};