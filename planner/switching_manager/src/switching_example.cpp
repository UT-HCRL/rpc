#include <iostream>

#include <contact_switching/core.hpp>

int main() {
    try {
        // Create contact switching manager
        auto contact_switching_manager = std::make_shared<ContactSwitchingManager>(
            std::unordered_map<std::string, bool>{
                {"left_foot", true},
                {"right_foot", true},
                {"left_hand", false},
                {"right_hand", false}
            }, 
            5 // Assuming 5 knots
        );

        // Create transition detector
        auto detector = std::make_shared<TransitionDetector>();
        
        // Create coordinator
        auto coordinator = std::make_unique<ContactTransitionCoordinator>(
            contact_switching_manager, detector);

        // Add a transition condition for time elapsed
        auto time_condition = std::make_shared<TimeElapsedCondition>("time_elapsed", 3.0, true);
        detector->addCondition(time_condition);

        // Create a context
        ContactSwitchUtils::Context ctx;
        ctx.time = 0.0;
        
        // Simulate time passing and check phase trigger
        std::cout << "=== Testing Time-based Transition ===" << std::endl;
        while(true) {
            ctx.time += 0.01; // Simulate time increment
            if (coordinator->processTransitions(ctx)) {
                std::cout << "Phase triggered at time: " << ctx.time << std::endl;
                break;
            }
        }

        // Update time condition threshold
        time_condition->update();

        // Reset switching mask and simulate more time passing
        contact_switching_manager->resetSwitchingMask();
        while(true) {
            ctx.time += 0.01; // Simulate time increment
            if (coordinator->processTransitions(ctx)) {
                std::cout << "Phase triggered at time: " << ctx.time << std::endl;
                break;
            }
        }

        time_condition->update();

        // Test the transition manager with a plane pass condition
        std::cout << "\n=== Testing Plane Pass Condition ===" << std::endl;
        Eigen::Isometry3d plane_pose = Eigen::Isometry3d::Identity();
        plane_pose.translation() = Eigen::Vector3d(0.0, 0.3, 0.0);
        Eigen::Vector3d plane_normal(0.0, 1.0, 0.0); // Normal pointing in the positive y direction
        auto plane_condition = std::make_shared<PlanePassCondition>("plane_pass", plane_pose, plane_normal, "left_hand");
        detector->addCondition(plane_condition);

        // Reset switching mask for plane test
        contact_switching_manager->resetSwitchingMask();
        
        double pos_y = 0.2;
        while(true) {
            ctx.time += 0.01;   // Simulate time increment
            pos_y += 0.01;      // Simulate left hand position increment
            ctx.pose["left_hand"] = Eigen::Isometry3d::Identity();
            ctx.pose["left_hand"].translation() = Eigen::Vector3d(0.0, pos_y, 0.0); 
            
            if (coordinator->processTransitions(ctx)) {
                std::cout << "Plane pass condition triggered at time: " << ctx.time 
                          << ", left hand position: " << ctx.pose["left_hand"].translation().transpose() << std::endl;
                break;
            }
        }

        // Check the contact switching mask before prediction
        std::cout << "\n=== Testing Predictive Transitions ===" << std::endl;
        contact_switching_manager->resetSwitchingMask();

        auto contact_mask = contact_switching_manager->getSwitchingMask();
        std::cout << "Contact switching mask before triggerPhasePrediction:" << std::endl;
        for (size_t i = 0; i < contact_mask.size(); ++i) {
            std::cout << "Contact " << i << ": " << (contact_mask[i] ? "Active" : "Inactive") << std::endl;
        }

        // Test the predictive transition detection
        std::unordered_map<std::string, std::vector<Eigen::Isometry3d>> future_poses;
        future_poses["left_hand"].resize(5);
        
        // Create future poses
        std::vector<double> y_positions = {0.2, 0.3, 0.4, 0.5, 0.6};
        for (int i = 0; i < 5; ++i) {
            future_poses["left_hand"][i] = Eigen::Isometry3d::Identity();
            future_poses["left_hand"][i].translation() = Eigen::Vector3d(0.0, y_positions[i], 0.0);
        }

        // Update context for prediction
        ctx.time = 0.0; // Reset to a different time for prediction test
        ctx.pose["left_hand"] = Eigen::Isometry3d::Identity();
        ctx.pose["left_hand"].translation() = Eigen::Vector3d(0.0, 0.2, 0.0);

        if (coordinator->processFutureTransitions(ctx, future_poses, 0.01)) {
            std::cout << "Phase triggered by future poses." << std::endl;
        } else {
            std::cout << "No phase trigger by future poses." << std::endl;
        }

        contact_mask = contact_switching_manager->getSwitchingMask();
        std::cout << "Contact switching mask after triggerPhasePrediction:" << std::endl;
        for (size_t i = 0; i < contact_mask.size(); ++i) {
            std::cout << "Contact " << i << ": " << (contact_mask[i] ? "Active" : "Inactive") << std::endl;
        }

        // Demonstrate custom transition callback
        std::cout << "\n=== Testing Custom Transition Callback ===" << std::endl;
        contact_switching_manager->resetSwitchingMask();
        
        // Set a custom callback that provides more detailed handling
        coordinator->setTransitionCallback([](const TransitionEvent& event, ContactSwitchingManager& csm) {
            std::cout << "Custom callback triggered!" << std::endl;
            std::cout << "  - Condition: " << event.condition_name << std::endl;
            std::cout << "  - Frame: " << event.frame_name << std::endl;
            std::cout << "  - Knot: " << event.triggered_knot << std::endl;
            std::cout << "  - Time: " << event.time << std::endl;
            
            // Custom logic: activate switching and set contact status
            csm.setSwitchingMask(event.triggered_knot);
            if (!event.frame_name.empty()) {
                csm.setContactStatus(event.frame_name, true);
                std::cout << "  - Set " << event.frame_name << " contact to active" << std::endl;
            }
        });

        // Test the custom callback with the plane condition
        pos_y = 0.25;
        ctx.time = 15.0;
        while(pos_y < 0.35) {
            ctx.time += 0.01;
            pos_y += 0.01;
            ctx.pose["left_hand"] = Eigen::Isometry3d::Identity();
            ctx.pose["left_hand"].translation() = Eigen::Vector3d(0.0, pos_y, 0.0);
            
            if (coordinator->processTransitions(ctx)) {
                break;
            }
        }

        // Show final contact status
        std::cout << "\nFinal contact status:" << std::endl;
        std::cout << "left_hand contact: " << (contact_switching_manager->getContactStatus("left_hand") ? "Active" : "Inactive") << std::endl;

        // Demonstrate event-based detection without automatic switching
        std::cout << "\n=== Testing Direct Event Detection ===" << std::endl;
        auto detector_only = std::make_shared<TransitionDetector>();
        auto simple_condition = std::make_shared<TimeElapsedCondition>("simple_time", 20.0, false);
        detector_only->addCondition(simple_condition);
        
        Context test_ctx;
        test_ctx.time = 19.5;
        auto events = detector_only->detectTransitions(test_ctx);
        std::cout << "Events detected at time 19.5: " << events.size() << std::endl;
        
        test_ctx.time = 20.5;
        events = detector_only->detectTransitions(test_ctx);
        std::cout << "Events detected at time 20.5: " << events.size() << std::endl;
        if (!events.empty()) {
            std::cout << "  - Event condition: " << events[0].condition_name << std::endl;
            std::cout << "  - Event time: " << events[0].time << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}