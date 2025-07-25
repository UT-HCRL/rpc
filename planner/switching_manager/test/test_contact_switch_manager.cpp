#include <gtest/gtest.h>
#include <memory>
#include <unordered_map>
#include <Eigen/Geometry>
#include <contact_switching/core.hpp>

class ContactSwitchManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        const std::vector<std::string> frame_names = {
            "left_foot", "right_foot", "left_hand", "right_hand"
        };
        // Create contact switching manager
        auto contact_switching_manager = std::make_shared<ContactSwitchingManager>(
            frame_names,
            5 // Assuming 5 knots
        );

        contact_switching_manager->addContactPhase({true, true, false, false}); // Initial phase with left and right foot active

        // Initialize transition detector
        detector = std::make_shared<TransitionDetector>();

        // Initialize coordinator
        coordinator = std::make_unique<ContactTransitionCoordinator>(
            contact_switching_manager, detector
        );

        // Initialize context
        ctx.time = 0.0;
    }

    std::shared_ptr<ContactSwitchingManager> contact_switching_manager;
    std::shared_ptr<TransitionDetector> detector;
    std::unique_ptr<ContactTransitionCoordinator> coordinator;
    ContactSwitchUtils::Context ctx;
};

TEST_F(ContactSwitchManagerTest, TimeBasedTransition) {
    auto time_condition = std::make_shared<TimeElapsedCondition>("time_elapsed", 3.0, true);
    detector->addCondition(time_condition);

    while (true) {
        ctx.time += 0.01; // Simulate time increment
        if (coordinator->processTransitions(ctx)) {
            EXPECT_FLOAT_EQ(ctx.time, 3.0);
            break;
        }
    }
}

TEST_F(ContactSwitchManagerTest, PlanePassCondition) {
    Eigen::Isometry3d plane_pose = Eigen::Isometry3d::Identity();
    plane_pose.translation() = Eigen::Vector3d(0.0, 0.3, 0.0);
    Eigen::Vector3d plane_normal(0.0, 1.0, 0.0);
    auto plane_condition = std::make_shared<PlanePassCondition>("plane_pass", plane_pose, plane_normal, "left_hand");
    detector->addCondition(plane_condition);

    contact_switching_manager->resetSwitchingMask();
    double pos_y = 0.2;
    while (true) {
        ctx.time += 0.01;
        pos_y += 0.01;
        ctx.pose["left_hand"] = Eigen::Isometry3d::Identity();
        ctx.pose["left_hand"].translation() = Eigen::Vector3d(0.0, pos_y, 0.0);

        if (coordinator->processTransitions(ctx)) {
            EXPECT_FLOAT_EQ(pos_y, 0.3);
            break;
        }
    }
}

TEST_F(ContactSwitchManagerTest, PredictiveTransitions) {
    contact_switching_manager->resetSwitchingMask();

    std::unordered_map<std::string, std::vector<Eigen::Isometry3d>> future_poses;
    future_poses["left_hand"].resize(5);

    std::vector<double> y_positions = {0.2, 0.3, 0.4, 0.5, 0.6};
    for (int i = 0; i < 5; ++i) {
        future_poses["left_hand"][i] = Eigen::Isometry3d::Identity();
        future_poses["left_hand"][i].translation() = Eigen::Vector3d(0.0, y_positions[i], 0.0);
    }

    ctx.time = 0.0;
    ctx.pose["left_hand"] = Eigen::Isometry3d::Identity();
    ctx.pose["left_hand"].translation() = Eigen::Vector3d(0.0, 0.2, 0.0);

    bool triggered = coordinator->processFutureTransitions(ctx, future_poses, 0.01);
    EXPECT_FALSE(triggered);
}

TEST_F(ContactSwitchManagerTest, CustomTransitionCallback) {
    contact_switching_manager->resetSwitchingMask();

    coordinator->setTransitionCallback([](const TransitionEvent& event, ContactSwitchingManager& csm) {
        EXPECT_EQ(event.condition_name, "plane_pass");
        csm.setSwitchingMask(event.triggered_knot);
        if (!event.frame_name.empty()) {
            csm.setContactStatus(event.frame_name, true);
        }
    });

    Eigen::Isometry3d plane_pose = Eigen::Isometry3d::Identity();
    plane_pose.translation() = Eigen::Vector3d(0.0, 0.3, 0.0);
    Eigen::Vector3d plane_normal(0.0, 1.0, 0.0);
    auto plane_condition = std::make_shared<PlanePassCondition>("plane_pass", plane_pose, plane_normal, "left_hand");
    detector->addCondition(plane_condition);

    double pos_y = 0.25;
    ctx.time = 15.0;
    while (pos_y < 0.35) {
        ctx.time += 0.01;
        pos_y += 0.01;
        ctx.pose["left_hand"] = Eigen::Isometry3d::Identity();
        ctx.pose["left_hand"].translation() = Eigen::Vector3d(0.0, pos_y, 0.0);

        if (coordinator->processTransitions(ctx)) {
            break;
        }
    }

    EXPECT_TRUE(contact_switching_manager->getContactStatus("left_hand"));
}

TEST_F(ContactSwitchManagerTest, DirectEventDetection) {
    auto detector_only = std::make_shared<TransitionDetector>();
    auto simple_condition = std::make_shared<TimeElapsedCondition>("simple_time", 20.0, false);
    detector_only->addCondition(simple_condition);

    ContactSwitchUtils::Context test_ctx;
    test_ctx.time = 19.5;
    auto events = detector_only->detectTransitions(test_ctx);
    EXPECT_EQ(events.size(), 0);

    test_ctx.time = 20.5;
    events = detector_only->detectTransitions(test_ctx);
    EXPECT_EQ(events.size(), 1);
    EXPECT_EQ(events[0].condition_name, "simple_time");
    EXPECT_EQ(events[0].time, 20.5);
}

TEST_F(ContactSwitchManagerTest, SwitchingMaskReset) {
    contact_switching_manager->resetSwitchingMask();
    auto mask = contact_switching_manager->getSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (const auto& value : mask) {
        EXPECT_FALSE(value);
    }

    contact_switching_manager->setSwitchingMask(2);
    mask = contact_switching_manager->getSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (size_t i = 2; i < mask.size(); ++i) {
        EXPECT_TRUE(mask[i]);
    }

    contact_switching_manager->resetSwitchingMask();
    mask = contact_switching_manager->getSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (const auto& value : mask) {
        EXPECT_FALSE(value);
    }
}

TEST_F(ContactSwitchManagerTest, PopSwitchingMask) {
    contact_switching_manager->resetSwitchingMask();

    contact_switching_manager->setSwitchingMask(3);
    auto mask = contact_switching_manager->getSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (size_t i = 3; i < mask.size(); ++i) {
        EXPECT_TRUE(mask[i]);
    }
    
    mask = contact_switching_manager->popSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (size_t i = 3; i < mask.size(); ++i) {
        EXPECT_TRUE(mask[i]);
    }

    mask = contact_switching_manager->getSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (size_t i = 3; i < mask.size(); ++i) {
        EXPECT_FALSE(mask[i]);
    }
}

TEST_F(ContactSwitchManagerTest, SwitchingMaskWithTransitions) {
    contact_switching_manager->resetSwitchingMask();
    auto mask = contact_switching_manager->getSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (size_t i = 0; i < mask.size(); ++i) {
        EXPECT_FALSE(mask[i]);
    }

    auto time_condition = std::make_shared<TimeElapsedCondition>("time_elapsed", 2.0, true);
    detector->addCondition(time_condition);

    ctx.time = 0.0;
    while (ctx.time < 3.0) {
        ctx.time += 0.01;
        bool transition_enabled = coordinator->processTransitions(ctx);
        mask = contact_switching_manager->getSwitchingMask();
        if (transition_enabled) {
            for (size_t i = 0; i < mask.size(); ++i) {
                EXPECT_TRUE(mask[i]);
            }
            break;
        }else{
            for (size_t i = 0; i < mask.size(); ++i) {
                EXPECT_FALSE(mask[i]);
            }
        }
    }

}

TEST_F(ContactSwitchManagerTest, SwitchingMaskWithPredictedTransitions){
    contact_switching_manager->resetSwitchingMask();

    auto plane_condition = std::make_shared<PlanePassCondition>(
        "plane_pass", 
        Eigen::Isometry3d(Eigen::Translation3d(0.0, 0.5, 0.0)), 
        Eigen::Vector3d(0.0, 1.0, 0.0), 
        "left_hand"
    );
    detector->addCondition(plane_condition);

    std::unordered_map<std::string, std::vector<Eigen::Isometry3d>> future_poses;
    future_poses["left_hand"].resize(5);

    std::vector<double> y_positions = {0.2, 0.3, 0.4, 0.5, 0.6};
    for (int i = 0; i < 5; ++i) {
        future_poses["left_hand"][i] = Eigen::Isometry3d::Identity();
        future_poses["left_hand"][i].translation() = Eigen::Vector3d(0.0, y_positions[i], 0.0);
    }

    ctx.time = 0.0;
    ctx.pose["left_hand"] = Eigen::Isometry3d::Identity();
    ctx.pose["left_hand"].translation() = Eigen::Vector3d(0.0, 0.2, 0.0);

    bool triggered = coordinator->processFutureTransitions(ctx, future_poses, 0.01);
    EXPECT_FALSE(triggered);

    auto mask = contact_switching_manager->getSwitchingMask();
    EXPECT_EQ(mask.size(), 5);
    for (size_t i = 3; i < mask.size(); ++i) {
        EXPECT_TRUE(mask[i]);
    }
}