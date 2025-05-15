import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

matplotlib.rcParams['figure.dpi'] = 300
plt.style.use("fivethirtyeight")

def set_fig_opt(ax):
    """
    Sets figure options for a given axis.
    """
    ax.grid(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='major', labelsize=20)

def plot_phase(ax, time, state, phase_color_map):
    """
    Plots phase background colors on a given axis.
    """
    time = time.flatten()
    state = state.flatten()
    
    unique_states = np.unique(state)
    for s in unique_states:
        if s in phase_color_map:
            # Find contiguous regions of the same state
            is_state = (state == s).astype(int)
            state_changes = np.diff(np.concatenate(([0], is_state, [0])))
            starts = np.where(state_changes == 1)[0]
            ends = np.where(state_changes == -1)[0]

            for start_idx, end_idx in zip(starts, ends):
                if end_idx > start_idx:
                    ax.axvspan(time[start_idx], time[end_idx-1], 
                               facecolor=phase_color_map[s], alpha=0.2, edgecolor='none')

def quat_to_euler(q):
    """
    Converts a quaternion to Euler angles.
    """
    # q is expected to be [x, y, z, w]
    # convert to [w, x, y, z] for some conventions if needed, but scipy takes [x, y, z, w]
    # The MATLAB file seems to have it as [x,y,z,w] in des_..._pos and converts to [w,x,y,z] for quat2eul
    # So we need to swap them before passing to scipy if the input is [w,x,y,z]
    # The matlab code does:
    # torso_ori_des_quat(1, :) = des_torso_ori_pos(4, :); -> w
    # torso_ori_des_quat(2:4, :) = des_torso_ori_pos(1:3, :); -> x,y,z
    # This means des_torso_ori_pos is [x,y,z,w]
    # Scipy's from_quat takes [x,y,z,w]
    quat = np.array([0, 0, 0, 1])[None, :]
    safe_q = np.where(np.linalg.norm(q, axis=-1, keepdims=True) == 0, quat.repeat(q.shape[0], axis=0), q)
    return Rotation.from_quat(safe_q).as_euler('xyz')

def save_and_close(fig, name):
    plt.savefig(os.path.join(outdir, f"{name}.png"))
    plt.close(fig)

# ==============================================================================
# Data Loading
# ==============================================================================
exp_data_location = '/tmp'
b_include_lhand_task = False
outdir = os.path.abspath("experiment_data/graphs/g1")
os.makedirs(outdir, exist_ok=True)

def find_latest_mat_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    latest_file = max(files, key=os.path.getmtime)
    print(f'Loading {latest_file}')
    return h5py.File(latest_file, "r")

try:
    controller_data = find_latest_mat_file(os.path.join(exp_data_location, 'g1_controller_data*.mat'))
    estimator_data = find_latest_mat_file(os.path.join(exp_data_location, 'g1_state_estimator_kf_data*.mat'))
    # icp_data = find_latest_mat_file(os.path.join(exp_data_location, 'g1_icp_data*.mat'))
except FileNotFoundError as e:
    print(e)
    exit()

print(list(controller_data.keys()))
print("")
print(list(estimator_data.keys()))

data = {k: np.array(v) for d in [controller_data, estimator_data] for k, v in d.items() if not k.startswith('__')}

# ==============================================================================
# Labels and Colors
# ==============================================================================
g1_joint_label = [
    'l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll',
    'r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll'
] # Placeholder G1 joint names

g1_lf_label = ['l_hip_yaw', 'l_hip_roll', 'l_hip_pitch', 'l_knee', 'l_ankle_pitch', 'l_ankle_roll', 'l_ankle_ie']
g1_rf_label = ['r_hip_yaw', 'r_hip_roll', 'r_hip_pitch', 'r_knee', 'r_ankle_pitch', 'r_ankle_roll', 'r_ankle_ie']

g1_lf_idx = [0, 1, 2, 3, 4, 5, 6] # Placeholder indices for G1
g1_rf_idx = [7, 8, 9, 10, 11, 12, 13] # Placeholder indices for G1

phase_color = {
    1: [0.8, 0.8, 0.8],  # Double Support 1
    2: [0.6, 0.8, 0.6],  # Right Swing
    3: [0.8, 0.8, 0.8],  # Double Support 2
    4: [0.8, 0.6, 0.6],  # Left Swing
}

xy_label = ['x', 'y']
xyz_label = ['x', 'y', 'z']
rpy_label = ['roll', 'pitch', 'yaw']
xy_dot_label = ['x_dot', 'y_dot']
xyz_dot_label = ['x_dot', 'y_dot', 'z_dot']
rf_label = ['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz']
fb_qddot_label = ['x_ddot', 'y_ddot', 'z_ddot', 'roll_ddot', 'pitch_ddot', 'yaw_ddot']
task_names = ["delta_qddot", "delta_rf", "xc_ddot"]
g1_wbc_cost_labels = task_names

# ==============================================================================
# Data Post-processing
# ==============================================================================
num_wbc_data = np.array(data['rf_rf_cmd']).shape[0]
wbc_time = data["time"].flatten()[-num_wbc_data:]

# data = {k: v[-num_wbc_data:] for k, v in data.items()}
time = data['time']
state = data['state']

# ==============================================================================
# Plotting
# ==============================================================================

# # ICP XY Task
# fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
# fig.suptitle('ICP XY Task', fontsize=30)
# for i in range(2):
#     axes[i].plot(wbc_time, data['des_icp'][:, i], linewidth=1, label='Desired')
#     axes[i].plot(wbc_time, data['icp_est'][:, i], linewidth=2, label='Actual')
#     axes[i].set_ylabel(xy_label[i])
#     set_fig_opt(axes[i])
#     plot_phase(axes[i], time, state, phase_color)
# axes[-1].set_xlabel('time')
# axes[0].legend()

# # ICP XY Task in LOCAL
# fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
# fig.suptitle('ICP XY Task in LOCAL', fontsize=30)
# for i in range(2):
#     axes[i].plot(wbc_time, data['local_des_icp'][:, i], linewidth=1, label='Desired')
#     axes[i].plot(wbc_time, data['local_act_icp'][:, i], linewidth=2, label='Actual')
#     axes[i].set_ylabel(xy_label[i])
#     set_fig_opt(axes[i])
#     plot_phase(axes[i], time, state, phase_color)
# axes[-1].set_xlabel('time')
# axes[0].legend()

# CoM XY Task
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
fig.suptitle('CoM XY Task', fontsize=30)
for i in range(2):
    # Position
    axes[i, 0].plot(wbc_time, data['act_com_xy_pos'][:, i], linewidth=2, label='Actual')
    axes[i, 0].plot(wbc_time, data['des_com_xy_pos'][:, i], linewidth=1, label='Desired')
    axes[i, 0].set_ylabel(f'pos {xy_label[i]}')
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)

    # Velocity
    axes[i, 1].plot(wbc_time, data['act_com_xy_vel'][:, i], linewidth=2, label='Actual')
    axes[i, 1].plot(wbc_time, data['des_com_xy_vel'][:, i], linewidth=1, label='Desired')
    axes[i, 1].set_ylabel(f'vel {xy_label[i]}')
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)

axes[0,0].legend()
axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "com_xy")

# CoM Z Task
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('CoM Z Task', fontsize=30)
axes[0].plot(wbc_time, data['des_com_z_pos'].flatten(), linewidth=1, label='Desired')
axes[0].plot(wbc_time, data['act_com_z_pos'].flatten(), linewidth=2, label='Actual')
axes[0].set_ylabel('z pos')
axes[0].set_xlabel('time')
set_fig_opt(axes[0])
plot_phase(axes[0], time, state, phase_color)
axes[0].legend()

axes[1].plot(wbc_time, data['des_com_z_vel'].flatten(), linewidth=1, label='Desired')
axes[1].plot(wbc_time, data['act_com_z_vel'].flatten(), linewidth=2, label='Actual')
axes[1].set_ylabel('z vel')
axes[1].set_xlabel('time')
set_fig_opt(axes[1])
plot_phase(axes[1], time, state, phase_color)
axes[1].legend()

save_and_close(fig, "com_z")

# Centroidal Angular Momentum
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
fig.suptitle('Centroidal Angular Momentum', fontsize=30)
for i in range(3):
    axes[i].plot(wbc_time, data['act_cam'][:, i], linewidth=2, label='Actual')
    axes[i].plot(wbc_time, data['des_cam'][:, i], linewidth=1, label='Desired')
    axes[i].set_ylabel(xyz_label[i])
    set_fig_opt(axes[i])
    plot_phase(axes[i], time, state, phase_color)
axes[0].legend()
axes[-1].set_xlabel('time')

save_and_close(fig, "ang_momentum")

# Torso ori task
des_torso_ori_euler_xyz = quat_to_euler(data['des_torso_ori_pos'])
act_torso_ori_euler_xyz = quat_to_euler(data['act_torso_ori_pos'])

fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
fig.suptitle('Torso ori task', fontsize=30)
for i in range(3):
    # Position
    axes[i, 0].plot(wbc_time, act_torso_ori_euler_xyz[:, i], linewidth=2, label='Actual')
    axes[i, 0].plot(wbc_time, des_torso_ori_euler_xyz[:, i], linewidth=1, label='Desired')
    axes[i, 0].set_ylabel(f'pos {rpy_label[i]}')
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)

    # Velocity
    axes[i, 1].plot(wbc_time, data['act_torso_ori_vel'][:, i], linewidth=2, label='Actual')
    axes[i, 1].plot(wbc_time, data['des_torso_ori_vel'][:, i], linewidth=1, label='Desired')
    axes[i, 1].set_ylabel(f'vel {xyz_dot_label[i]}')
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)

axes[0,0].legend()
axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "torso_ori")

if b_include_lhand_task:
    # Left hand pos task
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
    fig.suptitle('Left Hand Task', fontsize=30)
    for i in range(3):
        axes[i, 0].plot(wbc_time, data['act_lh_pos'][:, i], linewidth=2, label='Actual')
        axes[i, 0].plot(wbc_time, data['des_lh_pos'][:, i], linewidth=1, label='Desired')
        axes[i, 0].set_ylabel(f'pos {xyz_label[i]}')
        set_fig_opt(axes[i, 0])
        plot_phase(axes[i, 0], time, state, phase_color)

        axes[i, 1].plot(wbc_time, data['act_lh_vel'][:, i], linewidth=2, label='Actual')
        axes[i, 1].plot(wbc_time, data['des_lh_vel'][:, i], linewidth=1, label='Desired')
        axes[i, 1].set_ylabel(f'vel {xyz_dot_label[i]}')
        set_fig_opt(axes[i, 1])
        plot_phase(axes[i, 1], time, state, phase_color)
    axes[0,0].legend()
    axes[-1, 0].set_xlabel('time')
    axes[-1, 1].set_xlabel('time')

    save_and_close(fig, "lh")

    # Left hand ori task
    des_lh_ori_euler_xyz = quat_to_euler(data['des_lh_ori'])
    act_lh_ori_euler_xyz = quat_to_euler(data['act_lh_ori'])
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
    fig.suptitle('Left Hand ori task', fontsize=30)
    for i in range(3):
        axes[i, 0].plot(wbc_time, act_lh_ori_euler_xyz[:, i], linewidth=2, label='Actual')
        axes[i, 0].plot(wbc_time, des_lh_ori_euler_xyz[:, i], linewidth=1, label='Desired')
        axes[i, 0].set_ylabel(f'pos {rpy_label[i]}')
        set_fig_opt(axes[i, 0])
        plot_phase(axes[i, 0], time, state, phase_color)

        axes[i, 1].plot(wbc_time, data['act_lh_ori_vel'][:, i], linewidth=2, label='Actual')
        axes[i, 1].plot(wbc_time, data['des_lh_ori_vel'][:, i], linewidth=1, label='Desired')
        axes[i, 1].set_ylabel(f'vel {xyz_dot_label[i]}')
        set_fig_opt(axes[i, 1])
        plot_phase(axes[i, 1], time, state, phase_color)
    axes[0,0].legend()
    axes[-1, 0].set_xlabel('time')
    axes[-1, 1].set_xlabel('time')

    save_and_close(fig, "lh_ori")

# Left foot pos task
fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
fig.suptitle('Left Foot Task', fontsize=30)
for i in range(3):
    axes[i, 0].plot(wbc_time, data['act_lf_pos'][:, i], linewidth=2, label='Actual')
    axes[i, 0].plot(wbc_time, data['des_lf_pos'][:, i], linewidth=1, label='Desired')
    axes[i, 0].set_ylabel(f'pos {xyz_label[i]}')
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)

    axes[i, 1].plot(wbc_time, data['act_lf_vel'][:, i], linewidth=2, label='Actual')
    axes[i, 1].plot(wbc_time, data['des_lf_vel'][:, i], linewidth=1, label='Desired')
    axes[i, 1].set_ylabel(f'vel {xyz_dot_label[i]}')
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
axes[0,0].legend()
axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "lfoot")

# Left foot ori task
des_lf_ori_euler_xyz = quat_to_euler(data['des_lf_ori'])
act_lf_ori_euler_xyz = quat_to_euler(data['act_lf_ori'])
fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
fig.suptitle('Left foot ori task', fontsize=30)
for i in range(3):
    axes[i, 0].plot(wbc_time, act_lf_ori_euler_xyz[:, i], linewidth=2, label='Actual')
    axes[i, 0].plot(wbc_time, des_lf_ori_euler_xyz[:, i], linewidth=1, label='Desired')
    axes[i, 0].set_ylabel(f'pos {rpy_label[i]}')
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)

    axes[i, 1].plot(wbc_time, data['act_lf_ori_vel'][:, i], linewidth=2, label='Actual')
    axes[i, 1].plot(wbc_time, data['des_lf_ori_vel'][:, i], linewidth=1, label='Desired')
    axes[i, 1].set_ylabel(f'vel {xyz_dot_label[i]}')
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
axes[0,0].legend()
axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "lfoot_ori")

# Right foot pos task
fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
fig.suptitle('Right Foot Task', fontsize=30)
for i in range(3):
    axes[i, 0].plot(wbc_time, data['act_rf_pos'][:, i], linewidth=2, label='Actual')
    axes[i, 0].plot(wbc_time, data['des_rf_pos'][:, i], linewidth=1, label='Desired')
    axes[i, 0].set_ylabel(f'pos {xyz_label[i]}')
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)

    axes[i, 1].plot(wbc_time, data['act_rf_vel'][:, i], linewidth=2, label='Actual')
    axes[i, 1].plot(wbc_time, data['des_rf_vel'][:, i], linewidth=1, label='Desired')
    axes[i, 1].set_ylabel(f'vel {xyz_dot_label[i]}')
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
axes[0,0].legend()
axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "rfoot")

# Right foot ori task
des_rf_ori_euler_xyz = quat_to_euler(data['des_rf_ori'])
act_rf_ori_euler_xyz = quat_to_euler(data['act_rf_ori'])
fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
fig.suptitle('Right foot ori task', fontsize=30)
for i in range(3):
    axes[i, 0].plot(wbc_time, act_rf_ori_euler_xyz[:, i], linewidth=2, label='Actual')
    axes[i, 0].plot(wbc_time, des_rf_ori_euler_xyz[:, i], linewidth=1, label='Desired')
    axes[i, 0].set_ylabel(f'pos {rpy_label[i]}')
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)

    axes[i, 1].plot(wbc_time, data['act_rf_ori_vel'][:, i], linewidth=2, label='Actual')
    axes[i, 1].plot(wbc_time, data['des_rf_ori_vel'][:, i], linewidth=1, label='Desired')
    axes[i, 1].set_ylabel(f'vel {xyz_dot_label[i]}')
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
axes[0,0].legend()
axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "rfoot_ori")

# Reaction force cmd
fig, axes = plt.subplots(6, 2, figsize=(16, 30), sharex=True)
fig.suptitle('Reaction Force Command (Local)', fontsize=30)
for i in range(6):
    # Left foot
    axes[i, 0].plot(wbc_time, data['lf_rf_cmd'][:, i], linewidth=2, label='Command')
    axes[i, 0].plot(wbc_time, data['des_rf_lfoot'][:, i], 'k', linewidth=1, label='Desired')
    axes[i, 0].set_ylabel(rf_label[i])
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)
    if i == 0:
        axes[i, 0].set_title('Left Foot')
        axes[i, 0].legend()

    # Right foot
    axes[i, 1].plot(wbc_time, data['rf_rf_cmd'][:, i], linewidth=2, label='Command')
    axes[i, 1].plot(wbc_time, data['des_rf_rfoot'][:, i], 'k', linewidth=1, label='Desired')
    axes[i, 1].set_ylabel(rf_label[i])
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
    if i == 0:
        axes[i, 1].set_title('Right Foot')
        axes[i, 1].legend()

axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "rf")

# Floating base qddot cmd
fig, axes = plt.subplots(6, 1, figsize=(12, 30), sharex=True)
fig.suptitle('Floating base qddot cmd', fontsize=30)
for i in range(6):
    axes[i].plot(wbc_time, data['fb_qddot_cmd'][:, i], linewidth=2)
    axes[i].set_ylabel(fb_qddot_label[i])
    set_fig_opt(axes[i])
    plot_phase(axes[i], time, state, phase_color)
axes[-1].set_xlabel('time')

save_and_close(fig, "fb_qddot_cmd")

# Foot qddot cmd
fig, axes = plt.subplots(7, 2, figsize=(16, 35), sharex=True)
fig.suptitle('Foot qddot cmd', fontsize=30)
for i in range(7):
    # Left foot
    axes[i, 0].plot(wbc_time, data['joint_acc_cmd'][:, g1_lf_idx[i]], linewidth=2)
    axes[i, 0].set_ylabel(g1_lf_label[i])
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)
    if i == 0:
        axes[i, 0].set_title('Left Foot')

    # Right foot
    axes[i, 1].plot(wbc_time, data['joint_acc_cmd'][:, g1_rf_idx[i]], linewidth=2)
    axes[i, 1].set_ylabel(g1_rf_label[i])
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
    if i == 0:
        axes[i, 1].set_title('Right Foot')

axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "foot_qddot_cmd")

# Foot torque cmd
fig, axes = plt.subplots(7, 2, figsize=(16, 35), sharex=True)
fig.suptitle('Foot torque cmd', fontsize=30)
for i in range(7):
    # Left foot
    axes[i, 0].plot(wbc_time, data['joint_trq_cmd'][:, g1_lf_idx[i]], linewidth=2)
    axes[i, 0].set_ylabel(g1_lf_label[i])
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)
    if i == 0:
        axes[i, 0].set_title('Left Foot')

    # Right foot
    axes[i, 1].plot(wbc_time, data['joint_trq_cmd'][:, g1_rf_idx[i]], linewidth=2)
    axes[i, 1].set_ylabel(g1_rf_label[i])
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
    if i == 0:
        axes[i, 1].set_title('Right Foot')

axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "foot_trq_cmd")

# Lower body joint position
l_joint_lim_low = np.array([-0.872665, -0.2618, -1.52716, -0.0872665, -0.0872665, -1.5708, -0.523599])
l_joint_lim_up = np.array([0.872665, 0.7854, 0.5236, 1.52716, 1.52716, 1.0472, 0.523599])
r_joint_lim_low = np.array([-0.872665, -0.7854, -1.52716, -0.0872665, -0.0872665, -1.5708, -0.523599])
r_joint_lim_up = np.array([0.872665, 0.2618, 0.5236, 1.52716, 1.52716, 1.0472, 0.523599])

fig, axes = plt.subplots(7, 2, figsize=(16, 35), sharex=True)
fig.suptitle('Lower body joint position', fontsize=30)
for i in range(7):
    # Left foot
    axes[i, 0].plot(time.flatten(), data['joint_pos_cmd'][:, g1_lf_idx[i]], linewidth=2, label='Command')
    # axes[i, 0].plot(time.flatten(), data['joint_pos_act'][:, g1_lf_idx[i]], linewidth=2, label='Actual')
    axes[i, 0].plot(time.flatten(), np.full_like(time.flatten(), l_joint_lim_low[i]), '--k', linewidth=2)
    axes[i, 0].plot(time.flatten(), np.full_like(time.flatten(), l_joint_lim_up[i]), '--k', linewidth=2)
    axes[i, 0].set_ylabel(g1_lf_label[i])
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)
    if i == 0:
        axes[i, 0].set_title('Left Foot')
        axes[i, 0].legend()

    # Right foot
    axes[i, 1].plot(time.flatten(), data['joint_pos_cmd'][:, g1_rf_idx[i]], linewidth=2, label='Command')
    # axes[i, 1].plot(time.flatten(), data['joint_pos_act'][:, g1_rf_idx[i]], linewidth=2, label='Actual')
    axes[i, 1].plot(time.flatten(), np.full_like(time.flatten(), r_joint_lim_low[i]), '--k', linewidth=2)
    axes[i, 1].plot(time.flatten(), np.full_like(time.flatten(), r_joint_lim_up[i]), '--k', linewidth=2)
    axes[i, 1].set_ylabel(g1_rf_label[i])
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
    if i == 0:
        axes[i, 1].set_title('Right Foot')
        axes[i, 1].legend()

axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "joint_pos")

# Lower body joint velocity
fig, axes = plt.subplots(7, 2, figsize=(16, 35), sharex=True)
fig.suptitle('Lower body joint velocity', fontsize=30)
for i in range(7):
    # Left foot
    axes[i, 0].plot(time.flatten(), data['joint_vel_cmd'][:, g1_lf_idx[i]], linewidth=2, label='Command')
    # axes[i, 0].plot(time.flatten(), data['joint_vel_act'][:, g1_lf_idx[i]], linewidth=2, label='Actual')
    axes[i, 0].set_ylabel(g1_lf_label[i])
    set_fig_opt(axes[i, 0])
    plot_phase(axes[i, 0], time, state, phase_color)
    if i == 0:
        axes[i, 0].set_title('Left Foot')
        axes[i, 0].legend()

    # Right foot
    axes[i, 1].plot(time.flatten(), data['joint_vel_cmd'][:, g1_rf_idx[i]], linewidth=2, label='Command')
    # axes[i, 1].plot(time.flatten(), data['joint_vel_act'][:, g1_rf_idx[i]], linewidth=2, label='Actual')
    axes[i, 1].set_ylabel(g1_rf_label[i])
    set_fig_opt(axes[i, 1])
    plot_phase(axes[i, 1], time, state, phase_color)
    if i == 0:
        axes[i, 1].set_title('Right Foot')
        axes[i, 1].legend()

axes[-1, 0].set_xlabel('time')
axes[-1, 1].set_xlabel('time')

save_and_close(fig, "joint_vel")

# WBC Unweighted task costs
num_tasks = len(task_names)
num_cols = min(4, num_tasks)
num_rows = int(np.ceil(num_tasks / num_cols))
fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 5), sharex=True)
fig.suptitle('WBC Unweighted task costs', fontsize=30)
axes = axes.flatten()
for i, task_name in enumerate(task_names):
    cost_data = data.get(f'{task_name}_cost', None)
    if cost_data is not None:
        axes[i].plot(wbc_time[4:], cost_data.flatten()[4:], linewidth=2)
        axes[i].set_ylabel(g1_wbc_cost_labels[i])
        set_fig_opt(axes[i])
        plot_phase(axes[i], wbc_time, state.flatten()[-num_wbc_data:], phase_color)
        if i >= num_tasks - num_cols:
            axes[i].set_xlabel('time')

for i in range(num_tasks, len(axes)):
    axes[i].axis('off')

save_and_close(fig, "wbc_cost")

# # WBC Weighted task costs
# fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, num_rows * 5), sharex=True)
# fig.suptitle('WBC Weighted task costs', fontsize=30)
# axes = axes.flatten()
# for i, task_name in enumerate(task_names):
#     cost_data = data.get(f'wbc_cost_w_{task_name}', None)
#     print(data.keys())
#     if cost_data is not None:
#         axes[i].plot(wbc_time[4:], cost_data.flatten()[4:], linewidth=2)
#         axes[i].set_ylabel(g1_wbc_cost_labels[i])
#         set_fig_opt(axes[i])
#         plot_phase(axes[i], wbc_time, state.flatten()[-num_wbc_data:], phase_color)
#         if i >= num_tasks - num_cols:
#             axes[i].set_xlabel('time')

# for i in range(num_tasks, len(axes)):
#     axes[i].axis('off')

# save_and_close(fig, "wbc_cost_w")

# Early contact detection (Left foot)
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
fig.suptitle('Early contact detection (Left foot)', fontsize=30)

# Foot pos (z)
# axes[0].plot(wbc_time, data['act_lf_pos'][:, 2], linewidth=2, label='Actual')
axes[0].plot(wbc_time, data['des_lf_pos'][:, 2], linewidth=1, label='Desired')
axes[0].set_ylabel('Foot pos (z)')
set_fig_opt(axes[0])
plot_phase(axes[0], time, state, phase_color)
axes[0].legend()

# Reaction force (z-local)
# axes[1].plot(wbc_time, data['act_rf_z_lfoot'].flatten(), linewidth=2, label='Actual')
axes[1].plot(wbc_time, data['des_rf_lfoot'][:, 5], linewidth=1, label='Desired')
axes[1].set_ylabel('Reaction force (z-local)')
set_fig_opt(axes[1])
plot_phase(axes[1], time, state, phase_color)
axes[1].legend()

save_and_close(fig, "ecd_lfoot")

# Contact Sensor
# axes[2].plot(wbc_time, data['b_lf_contact_touchdown'].flatten(), linewidth=2, label='FT sensor')
# axes[2].plot(wbc_time, data['b_lf_heel_toe_touchdown'].flatten(), linewidth=2, label='Foot height')
# axes[2].plot(wbc_time, data['b_lf_contact'].flatten(), 'k', linewidth=2, label='Controller')
# axes[2].set_ylabel('Contact Sensor')
# axes[2].set_xlabel('time')
# set_fig_opt(axes[2])
# plot_phase(axes[2], time, state, phase_color)
# axes[2].legend()

# Early contact detection (Right foot)
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
fig.suptitle('Early contact detection (Right foot)', fontsize=30)

# Foot pos (z)
axes[0].plot(wbc_time, data['act_rf_pos'][:, 2], linewidth=2, label='Actual')
axes[0].plot(wbc_time, data['des_rf_pos'][:, 2], linewidth=1, label='Desired')
axes[0].set_ylabel('Foot pos (z)')
set_fig_opt(axes[0])
plot_phase(axes[0], time, state, phase_color)
axes[0].legend()

# Reaction force (z-local)
# axes[1].plot(wbc_time, data['act_rf_z_rfoot'].flatten(), linewidth=2, label='Actual')
axes[1].plot(wbc_time, data['des_rf_rfoot'][:, 5], linewidth=1, label='Desired')
axes[1].set_ylabel('Reaction force (z-local)')
set_fig_opt(axes[1])
plot_phase(axes[1], time, state, phase_color)
axes[1].legend()

save_and_close(fig, "ecd_rfoot")

# Contact Sensor
# axes[2].plot(wbc_time, data['b_rf_contact_touchdown'].flatten(), linewidth=2, label='FT sensor')
# axes[2].plot(wbc_time, data['b_rf_heel_toe_touchdown'].flatten(), linewidth=2, label='Foot height')
# axes[2].plot(wbc_time, data['b_rf_contact'].flatten(), 'k', linewidth=2, label='Controller')
# axes[2].set_ylabel('Contact Sensor')
# axes[2].set_xlabel('time')
# set_fig_opt(axes[2])
# plot_phase(axes[2], time, state, phase_color)
# axes[2].legend()

# Upper body task position
num_ub_joints = data['des_upper_body_pos'].shape[1]
fig, axes = plt.subplots(num_ub_joints, 1, figsize=(12, num_ub_joints * 3), sharex=True)
if num_ub_joints == 1:
    axes = [axes]
fig.suptitle('Upper body task position', fontsize=30)
for i in range(num_ub_joints):
    axes[i].plot(wbc_time, data['act_upper_body_pos'][:, i], linewidth=2, label='Actual')
    axes[i].plot(wbc_time, data['des_upper_body_pos'][:, i], linewidth=1, label='Desired')
    set_fig_opt(axes[i])
    plot_phase(axes[i], time, state, phase_color)
    # axes[i].set_ylabel(f'Joint {i}') # Add labels if available
axes[0].legend()
axes[-1].set_xlabel('time')

save_and_close(fig, "upper_pos")

# Upper body task velocity
fig, axes = plt.subplots(num_ub_joints, 1, figsize=(12, num_ub_joints * 3), sharex=True)
if num_ub_joints == 1: axes = [axes]
fig.suptitle('Upper body task velocity', fontsize=30)
for i in range(num_ub_joints):
    axes[i].plot(wbc_time, data['act_upper_body_vel'][:, i], linewidth=2, label='Actual')
    axes[i].plot(wbc_time, data['des_upper_body_vel'][:, i], linewidth=1, label='Desired')
    set_fig_opt(axes[i])
    plot_phase(axes[i], time, state, phase_color)
    # axes[i].set_ylabel(f'Joint {i} vel') # Add labels if available
axes[0].legend()
axes[-1].set_xlabel('time')

save_and_close(fig, "upper_vel")
