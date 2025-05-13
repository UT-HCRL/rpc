import numpy as np


class Config(object):
    CONTROLLER_DT = 0.00125
    N_SUBSTEP = 1

    INITIAL_BASE_JOINT_POS = [0, 0, 0.74]
    INITIAL_BASE_JOINT_QUAT = [0, 0, 0, 1]

    PRINT_ROBOT_INFO = True

    MEASURE_COMPUTATION_TIME = False

    VIDEO_RECORD = False
    RECORD_FREQ = 50
    RENDER_WIDTH = 1920
    RENDER_HEIGHT = 1080

    ##TODO:
    USE_MESHCAT = False


class G1LinkIdx(object):
    pelvis = -1
    pelvis_contour_link = 0
    left_hip_pitch_link = 1
    left_hip_roll_link = 2
    left_hip_yaw_link = 3
    left_knee_link = 4
    left_ankle_pitch_link = 5
    left_ankle_roll_link = 6
    l_foot_contact = 7
    right_hip_pitch_link = 8
    right_hip_roll_link = 9
    right_hip_yaw_link = 10
    right_knee_link = 11
    right_ankle_pitch_link = 12
    right_ankle_roll_link = 13
    r_foot_contact = 14
    waist_yaw_link = 15
    waist_roll_link = 16
    torso_link = 17
    torso_com_link = 18
    logo_link = 19
    head_link = 20
    waist_support_link = 21
    imu_in_torso = 22
    d435_link = 23
    mid360_link = 24
    left_shoulder_pitch_link = 25
    left_shoulder_roll_link = 26
    left_shoulder_yaw_link = 27
    left_elbow_link = 28
    left_wrist_roll_link = 29
    left_wrist_pitch_link = 30
    left_wrist_yaw_link = 31
    left_rubber_hand = 32
    right_shoulder_pitch_link = 33
    right_shoulder_roll_link = 34
    right_shoulder_yaw_link = 35
    right_elbow_link = 36
    right_wrist_roll_link = 37
    right_wrist_pitch_link = 38
    right_wrist_yaw_link = 39
    right_rubber_hand = 40
    imu_in_pelvis = 41


class G1JointIdx(object):
    left_hip_pitch_joint = 1
    left_hip_roll_joint = 2
    left_hip_yaw_joint = 3
    left_knee_joint = 4
    left_ankle_pitch_joint = 5
    left_ankle_roll_joint = 6
    right_hip_pitch_joint = 8
    right_hip_roll_joint = 9
    right_hip_yaw_joint = 10
    right_knee_joint = 11
    right_ankle_pitch_joint = 12
    right_ankle_roll_joint = 13
    waist_yaw_joint = 15
    left_shoulder_pitch_joint = 25
    left_shoulder_roll_joint = 26
    left_shoulder_yaw_joint = 27
    left_elbow_joint = 28
    left_wrist_roll_joint = 29
    left_wrist_pitch_joint = 30
    left_wrist_yaw_joint = 31
    right_shoulder_pitch_joint = 33
    right_shoulder_roll_joint = 34
    right_shoulder_yaw_joint = 35
    right_elbow_joint = 36
    right_wrist_roll_joint = 37
    right_wrist_pitch_joint = 38
    right_wrist_yaw_joint = 39


# ========= G1 29 dof version =========
# class G1LinkIdx(object):
#     pelvis = -1
#     pelvis_contour_link = 0
#     left_hip_pitch_link = 1
#     left_hip_roll_link = 2
#     left_hip_yaw_link = 3
#     left_knee_link = 4
#     left_ankle_pitch_link = 5
#     left_ankle_roll_link = 6
#     l_foot_contact = 7
#     right_hip_pitch_link = 8
#     right_hip_roll_link = 9
#     right_hip_yaw_link = 10
#     right_knee_link = 11
#     right_ankle_pitch_link = 12
#     right_ankle_roll_link = 13
#     r_foot_contact = 14
#     torso_link = 15
#     torso_com_link = 16
#     head_link = 17
#     left_shoulder_pitch_link = 18
#     left_shoulder_roll_link = 19
#     left_shoulder_yaw_link = 20
#     left_elbow_pitch_link = 21
#     left_elbow_roll_link = 22
#     left_palm_link = 23
#     left_zero_link = 24
#     left_one_link = 25
#     left_two_link = 26
#     left_three_link = 27
#     left_four_link = 28
#     left_five_link = 29
#     left_six_link = 30
#     right_shoulder_pitch_link = 31
#     right_shoulder_roll_link = 32
#     right_shoulder_yaw_link = 33
#     right_elbow_pitch_link = 34
#     right_elbow_roll_link = 35
#     right_palm_link = 36
#     right_zero_link = 37
#     right_one_link = 38
#     right_two_link = 39
#     right_three_link = 40
#     right_four_link = 41
#     right_five_link = 42
#     right_six_link = 43
#     logo_link = 44
#     imu_link = 45

# class G1JointIdx(object):
#     left_hip_pitch_joint = 1
#     left_hip_roll_joint = 2
#     left_hip_yaw_joint = 3
#     left_knee_joint = 4
#     left_ankle_pitch_joint = 5
#     left_ankle_roll_joint = 6
#     l_foot_contact = 7
#     right_hip_pitch_joint = 8
#     right_hip_roll_joint = 9
#     right_hip_yaw_joint = 10
#     right_knee_joint = 11
#     right_ankle_pitch_joint = 12
#     right_ankle_roll_joint = 13
#     r_foot_contact = 14
#     torso_joint = 15
#     torso_com_link = 16
#     head_link = 17
#     left_shoulder_pitch_joint = 18
#     left_shoulder_roll_joint = 19
#     left_shoulder_yaw_joint = 20
#     left_elbow_pitch_joint = 21
#     left_elbow_roll_joint = 22
#     left_palm_link = 23
#     left_zero_joint = 24
#     left_one_joint = 25
#     left_two_joint = 26
#     left_three_joint = 27
#     left_four_joint = 28
#     left_five_joint = 29
#     left_six_joint = 30
#     right_shoulder_pitch_joint = 31
#     right_shoulder_roll_joint = 32
#     right_shoulder_yaw_joint = 33
#     right_elbow_pitch_joint = 34
#     right_elbow_roll_joint = 35
#     right_palm_link = 36
#     right_zero_joint = 37
#     right_one_joint = 38
#     right_two_joint = 39
#     right_three_joint = 40
#     right_four_joint = 41
#     right_five_joint = 42
#     right_six_joint = 43

class G1ManipulationLinkIdx(object):
    torso_link = -1
    torso_com_link = 0
    neck_pitch_link = 1
    l_shoulder_fe_link = 2
    l_shoulder_aa_link = 3
    l_shoulder_ie_link = 4
    l_elbow_fe_link = 5
    l_wrist_ps_link = 6
    l_wrist_pitch_link = 7
    l_sake_gripper_link = 8
    l_hand_contact = 9
    left_ezgripper_palm_link = 10
    left_ezgripper_finger_L1_1 = 11
    left_ezgripper_finger_L2_1 = 12
    left_ezgripper_finger_L1_2 = 13
    left_ezgripper_finger_L2_2 = 14
    r_shoulder_fe_link = 15
    r_shoulder_aa_link = 16
    r_shoulder_ie_link = 17
    r_elbow_fe_link = 18
    r_wrist_ps_link = 19
    r_wrist_pitch_link = 20
    r_sake_gripper_link = 21
    r_hand_contact = 22
    right_ezgripper_palm_link = 23
    right_ezgripper_finger_L1_1 = 24
    right_ezgripper_finger_L2_1 = 25
    right_ezgripper_finger_L1_2 = 26
    right_ezgripper_finger_L2_2 = 27
    l_hip_ie_link = 28
    l_hip_aa_link = 29
    l_hip_fe_link = 30
    l_knee_fe_lp = 31
    l_knee_adj_link = 32
    l_knee_fe_ld = 33
    l_ankle_fe_link = 34
    l_ankle_ie_link = 35
    l_foot_contact = 36
    r_hip_ie_link = 37
    r_hip_aa_link = 38
    r_hip_fe_link = 39
    r_knee_fe_lp = 40
    r_knee_adj_link = 41
    r_knee_fe_ld = 42
    r_ankle_fe_link = 43
    r_ankle_ie_link = 44
    r_foot_contact = 45
    torso_imu = 46


class G1ManipulationJointIdx(object):
    neck_pitch = 1
    l_shoulder_fe = 2
    l_shoulder_aa = 3
    l_shoulder_ie = 4
    l_elbow_fe = 5
    l_wrist_ps = 6
    l_wrist_pitch = 7
    r_shoulder_fe = 15
    r_shoulder_aa = 16
    r_shoulder_ie = 17
    r_elbow_fe = 18
    r_wrist_ps = 19
    r_wrist_pitch = 20
    l_hip_ie = 28
    l_hip_aa = 29
    l_hip_fe = 30
    l_knee_fe_jp = 31
    l_knee_fe_jd = 33
    l_ankle_fe = 34
    l_ankle_ie = 35
    r_hip_ie = 37
    r_hip_aa = 38
    r_hip_fe = 39
    r_knee_fe_jp = 40
    r_knee_fe_jd = 42
    r_ankle_fe = 43
    r_ankle_ie = 44


class JointGains(object):
    kp = 5. * np.ones(27)
    kd = 0. * np.ones(27)
    # kp = np.array(
    #     [
    #         10,
    #         10,
    #         10,
    #         10,
    #         10,
    #         10,
    #         10,
    #         5,
    #         5,
    #         5,
    #         5,
    #         5,
    #         5,
    #         5,
    #         10,
    #         10,
    #         10,
    #         10,
    #         10,
    #         10,
    #         10,
    #         5,
    #         5,
    #         5,
    #         5,
    #         5,
    #         5,
    #     ]
    # )
    # kd = np.array(
    #     [
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.001,
    #         0.001,
    #         0.001,
    #         0.001,
    #         0.001,
    #         0.001,
    #         0.01,
    #         0.01,
    #         0.001,
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.001,
    #         0.001,
    #         0.001,
    #         0.001,
    #         0.001,
    #         0.001,
    #     ]
    # )
