import os
import glob
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# Configuration & Helpers
# ==============================================================================
EXP_DATA_LOCATION = '/tmp'
OUTDIR = os.path.abspath("experiment_data/graphs/mpc")
os.makedirs(OUTDIR, exist_ok=True)

def set_fig_opt(ax):
    """
    Sets figure options to match convex_mpc.m styling.
    """
    ax.grid(True)
    # Match Matlab's 'LineWidth', 3 on gca (axes spines)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    # Match Matlab's 'FontSize', 30
    ax.tick_params(axis='both', which='major', labelsize=30)
    # Match Matlab's 'TickLabelInterpreter', 'latex' (matplotlib uses mathtext by default or via rcParams)

def save_and_close(fig, name):
    plt.savefig(os.path.join(OUTDIR, f"{name}.png"))
    plt.close(fig)

def find_latest_mat_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    latest_file = max(files, key=os.path.getmtime)
    print(f'Loading {latest_file}')
    return h5py.File(latest_file, "r")

# ==============================================================================
# Data Loading
# ==============================================================================
try:
    # Match 'convex_mpc*.mat' pattern
    mpc_data_file = find_latest_mat_file(os.path.join(EXP_DATA_LOCATION, 'convex_mpc*.mat'))
except FileNotFoundError as e:
    print(e)
    exit()

# Extract data into a dictionary of numpy arrays
# Note: h5py reads Matlab v7.3 arrays with transposed dimensions (e.g. Matlab [3, N] -> Python [N, 3])
data = {}
for k in mpc_data_file.keys():
    if not k.startswith('__'):
        data[k] = np.array(mpc_data_file[k])

mpc_data_file.close()

# ==============================================================================
# Data Processing
# ==============================================================================

# Time manipulation
if 'time' in data:
    # Matlab: [row, col] = size(time); -> Python shape: (col, row)
    # In Matlab: for i = 2:col, time(1:row, i) = time(1:row, i) + time(end, i-1);
    # This implies time is cumulative across MPC steps (columns in Matlab, rows in Python).
    
    # We copy the data to avoid issues if h5py returned a read-only view or similar
    time_mat = data['time'].copy() 
    
    # Iterate through MPC steps (Python dim 0) starting from index 1
    for i in range(1, time_mat.shape[0]):
        time_mat[i, :] += time_mat[i-1, -1]
    
    # Reshape to vector
    # Matlab: reshape(time, [row*col, 1]) -> Column-major reshape
    # Python: time_mat is (C, R). Flattening (C-order) gives row 0, row 1...
    # Since Python rows correspond to Matlab columns (transposed), 
    # Python's flatten() on (C, R) effectively concatenates the horizons in order.
    # Check: Python row 0 is Matlab col 1 (transposed). Flatten puts Python row 0 then row 1.
    # Matlab reshape puts Matlab col 1 then col 2.
    # So flattened time_mat matches Matlab's time_vec.
    time_vec = time_mat.flatten()
    
    # Reaction Force Time
    # Matlab: time_rf = time(2:end, 1:col); -> Rows 2 to end.
    # Python time_mat is (C, R). Matlab rows are Python cols.
    # So we want Python columns 1 to end (skipping column 0).
    time_rf_mat = time_mat[:, 1:] 
    time_rf_vec = time_rf_mat.flatten()
else:
    print("Error: 'time' variable not found in data.")
    exit()

# ==============================================================================
# Plotting
# ==============================================================================

# 1. MPC Solve Time
if 'mpc_solve_time' in data:
    # Matlab: mpc_solve_time(2:end)
    solve_times = data['mpc_solve_time'].flatten()[1:]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(solve_times, linewidth=2, label='Raw')
    
    # Moving average
    window_size = 100
    if len(solve_times) >= window_size:
        # Simple moving average to match movmean behavior roughly
        filtered_solve_times = np.convolve(solve_times, np.ones(window_size)/window_size, mode='same')
        # Matlab's movmean might handle edges differently, but 'same' keeps shape
        ax.plot(filtered_solve_times, linewidth=2, color='y', label='Filtered')
        
    ax.set_title('MPC Solve Times', fontsize=20)
    ax.set_xlabel('MPC Iterations', fontsize=20)
    ax.set_ylabel('Solve Time (s)', fontsize=20)
    set_fig_opt(ax)
    ax.legend(fontsize=20)
    save_and_close(fig, "mpc_solve_time")

# 2. MPC CoM Trajectory
# Matlab plots com_pos(j, :) vs time_vec.
# com_pos in Matlab is likely (3, TotalSteps). In Python: (TotalSteps, 3).
if 'com_pos' in data and 'des_com_pos' in data and 'com_vel' in data and 'des_com_vel' in data:
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
    fig.suptitle('com states', fontsize=30)
    
    labels = ['x', 'y', 'z']
    
    for i in range(3):
        # Position (Left Column)
        ax = axes[i, 0]
        # Data check: Ensure length matches time_vec
        if data['com_pos'].shape[0] != time_vec.shape[0]:
             # If shapes mismatch (e.g. if com_pos wasn't (3, Total)), handle appropriately.
             # Assuming standard h5py transpose behavior where Matlab (3, N) -> Python (N, 3)
             pass 

        ax.plot(time_vec, data['com_pos'][:, i], 'b', linewidth=3, label='Actual')
        ax.plot(time_vec, data['des_com_pos'][:, i], 'r', linewidth=3, label='Desired')
        set_fig_opt(ax)
        ax.set_ylabel(f'pos {labels[i]}', fontsize=30)
        
        # Velocity (Right Column)
        ax = axes[i, 1]
        ax.plot(time_vec, data['com_vel'][:, i], 'b', linewidth=3, label='Actual')
        ax.plot(time_vec, data['des_com_vel'][:, i], 'r', linewidth=3, label='Desired')
        set_fig_opt(ax)
        ax.set_ylabel(f'vel {labels[i]}', fontsize=30)

    axes[-1, 0].set_xlabel('time', fontsize=30)
    axes[-1, 1].set_xlabel('time', fontsize=30)
    save_and_close(fig, "mpc_com")

# 3. MPC Ori Trajectory
if 'euler_ang' in data and 'des_euler_ang' in data and 'ang_vel' in data and 'des_ang_vel' in data:
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), sharex=True)
    fig.suptitle('ori states', fontsize=30)
    
    labels = ['x', 'y', 'z']
    
    for i in range(3):
        # Orientation (Left Column)
        ax = axes[i, 0]
        ax.plot(time_vec, data['euler_ang'][:, i], 'b', linewidth=3, label='Actual')
        ax.plot(time_vec, data['des_euler_ang'][:, i], 'r', linewidth=3, label='Desired')
        set_fig_opt(ax)
        ax.set_ylabel(f'euler {labels[i]}', fontsize=30)
        
        # Angular Velocity (Right Column)
        ax = axes[i, 1]
        ax.plot(time_vec, data['ang_vel'][:, i], 'b', linewidth=3, label='Actual')
        ax.plot(time_vec, data['des_ang_vel'][:, i], 'r', linewidth=3, label='Desired')
        set_fig_opt(ax)
        ax.set_ylabel(f'ang vel {labels[i]}', fontsize=30)

    axes[-1, 0].set_xlabel('time', fontsize=30)
    axes[-1, 1].set_xlabel('time', fontsize=30)
    save_and_close(fig, "mpc_ori")

# 4. MPC Reaction Wrench Trajectory
# Matlab loops i=1:12. Odd -> LF, Even -> RF.
# force_LF/RF in Matlab likely (6, N). Python (N, 6).
# time_rf_vec corresponds to the force data.
if 'force_LF' in data and 'force_RF' in data:
    fig, axes = plt.subplots(6, 2, figsize=(16, 30), sharex=True)
    fig.suptitle('reaction force', fontsize=30)
    
    rf_labels = ['Mx', 'My', 'Mz', 'Fx', 'Fy', 'Fz']
    
    for i in range(6):
        # Left Foot (Left Column)
        ax = axes[i, 0]
        ax.plot(time_rf_vec, data['force_LF'][:, i], 'r', linewidth=3)
        set_fig_opt(ax)
        ax.set_ylabel(f'LF {rf_labels[i]}', fontsize=30)
        
        # Right Foot (Right Column)
        ax = axes[i, 1]
        ax.plot(time_rf_vec, data['force_RF'][:, i], 'r', linewidth=3)
        set_fig_opt(ax)
        ax.set_ylabel(f'RF {rf_labels[i]}', fontsize=30)

    axes[-1, 0].set_xlabel('time', fontsize=30)
    axes[-1, 1].set_xlabel('time', fontsize=30)
    save_and_close(fig, "mpc_force")

# 5. MPC Reference Trajectory
if 'des_lf_pos' in data:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    des_lf_pos = data['des_lf_pos'] # Shape (N, 3)
    n = des_lf_pos.shape[0]
    
    # Color map
    cmap = matplotlib.colormaps['viridis']
    
    # Loop through segments of 10 steps
    # Matlab: i = 1:10:n-1. Python: range(0, n, 10)
    for i in range(0, n, 10):
        end_index = min(i+10, n)
        idx = np.arange(i, end_index)
        
        # Matlab plots des_lf_pos(1, i:endIndex) -> X component
        segment = des_lf_pos[i:end_index, 0] 
        
        # Color based on index
        color = cmap(i / n)
        
        ax.plot(idx, segment, '-', linewidth=2, color=color)
        
    ax.set_title('Robot Base Trajectory with Color Variation', fontsize=30)
    ax.set_xlabel('Step Index', fontsize=30)
    ax.set_ylabel('X Position', fontsize=30)
    set_fig_opt(ax)
    save_and_close(fig, "mpc_ref_traj")
