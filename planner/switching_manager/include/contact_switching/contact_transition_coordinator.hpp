#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>

#include "contact_switching_manager.hpp"
#include "transition_detector.hpp"

/**
 * @brief Coordinator for managing contact transitions.
 * 
 * This class coordinates the contact switching logic and transition detection.
 * It allows setting custom transition callbacks and processes transitions
 * based on the current context or future contexts.
 * 
 */
class ContactTransitionCoordinator {
public:
    using TransitionCallback = std::function<void(const TransitionEvent&, ContactSwitchingManager&)>;

    ContactTransitionCoordinator(std::shared_ptr<ContactSwitchingManager> csm,
                                std::shared_ptr<TransitionDetector> detector)
        : csm_(csm), detector_(detector) {
        
        // Set default transition callback
        setDefaultTransitionCallback();
    }

    void setTransitionCallback(TransitionCallback callback) {
        transition_callback_ = callback;
    }

    // Process current transitions
    bool processTransitions(const Context& ctx) {
        auto events = detector_->detectTransitions(ctx);
        
        if (events.empty()) {
            return false;
        }
        
        // Process first event
        transition_callback_(events[0], *csm_);
        return true;
    }

    // Process transitions while looking into future contexts
    bool processFutureTransitions(const Context& current_ctx,
                                 const std::unordered_map<std::string, std::vector<Eigen::Isometry3d>>& future_poses,
                                 double dt) {
        auto events = detector_->detectFutureTransitions(current_ctx, future_poses, dt, csm_->getKnots());
        
        if (events.empty()) {
            return false;
        }
        
        // Process first event
        transition_callback_(events[0], *csm_);
        return events[0].triggered_knot == 0;
    }

    std::shared_ptr<ContactSwitchingManager> getContactSwitchingManager() const {
        return csm_;
    }

    std::shared_ptr<TransitionDetector> getTransitionDetector() const {
        return detector_;
    }

private:
    std::shared_ptr<ContactSwitchingManager> csm_;
    std::shared_ptr<TransitionDetector> detector_;
    TransitionCallback transition_callback_;

    void setDefaultTransitionCallback() {
        transition_callback_ = [](const TransitionEvent& event, ContactSwitchingManager& csm) {
            // Default behavior: activate switching from the triggered knot
            csm.setSwitchingMask(event.triggered_knot);
        };
    }
};