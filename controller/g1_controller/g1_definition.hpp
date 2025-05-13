#pragma once

namespace g1_link {
constexpr int pelvis = 2;
constexpr int imu_in_pelvis = 4;
constexpr int left_hip_pitch_link = 6;
constexpr int left_hip_roll_link = 8;
constexpr int left_hip_yaw_link = 10;
constexpr int left_knee_link = 12;
constexpr int left_ankle_pitch_link = 14;
constexpr int left_ankle_roll_link = 16;
constexpr int l_foot_contact = 18;
constexpr int pelvis_contour_link = 20;
constexpr int right_hip_pitch_link = 22;
constexpr int right_hip_roll_link = 24;
constexpr int right_hip_yaw_link = 26;
constexpr int right_knee_link = 28;
constexpr int right_ankle_pitch_link = 30;
constexpr int right_ankle_roll_link = 32;
constexpr int r_foot_contact = 34;
constexpr int waist_yaw_link = 36;
constexpr int waist_roll_link = 38;
constexpr int torso_link = 40;
constexpr int d435_link = 42;
constexpr int head_link = 44;
constexpr int imu_in_torso = 46;
constexpr int left_shoulder_pitch_link = 48;
constexpr int left_shoulder_roll_link = 50;
constexpr int left_shoulder_yaw_link = 52;
constexpr int left_elbow_link = 54;
constexpr int left_wrist_roll_link = 56;
constexpr int left_wrist_pitch_link = 58;
constexpr int left_wrist_yaw_link = 60;
constexpr int left_rubber_hand = 62;
constexpr int logo_link = 64;
constexpr int mid360_link = 66;
constexpr int right_shoulder_pitch_link = 68;
constexpr int right_shoulder_roll_link = 70;
constexpr int right_shoulder_yaw_link = 72;
constexpr int right_elbow_link = 74;
constexpr int right_wrist_roll_link = 76;
constexpr int right_wrist_pitch_link = 78;
constexpr int right_wrist_yaw_link = 80;
constexpr int right_rubber_hand = 82;
constexpr int torso_com_link = 84;
constexpr int waist_support_link = 86;
} // namespace g1_link

namespace g1_joint {
constexpr int left_hip_pitch_joint = 0;
constexpr int left_hip_roll_joint = 1;
constexpr int left_hip_yaw_joint = 2;
constexpr int left_knee_joint = 3;
constexpr int left_ankle_pitch_joint = 4;
constexpr int left_ankle_roll_joint = 5;
constexpr int right_hip_pitch_joint = 6;
constexpr int right_hip_roll_joint = 7;
constexpr int right_hip_yaw_joint = 8;
constexpr int right_knee_joint = 9;
constexpr int right_ankle_pitch_joint = 10;
constexpr int right_ankle_roll_joint = 11;
constexpr int waist_yaw_joint = 12;
constexpr int left_shoulder_pitch_joint = 13;
constexpr int left_shoulder_roll_joint = 14;
constexpr int left_shoulder_yaw_joint = 15;
constexpr int left_elbow_joint = 16;
constexpr int left_wrist_roll_joint = 17;
constexpr int left_wrist_pitch_joint = 18;
constexpr int left_wrist_yaw_joint = 19;
constexpr int right_shoulder_pitch_joint = 20;
constexpr int right_shoulder_roll_joint = 21;
constexpr int right_shoulder_yaw_joint = 22;
constexpr int right_elbow_joint = 23;
constexpr int right_wrist_roll_joint = 24;
constexpr int right_wrist_pitch_joint = 25;
constexpr int right_wrist_yaw_joint = 26;
} // namespace g1_joint

namespace g1 {
constexpr int n_qdot = 33;
constexpr int n_adof = 27;
} // namespace g1
