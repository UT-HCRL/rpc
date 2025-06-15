import os
import sys
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append("/home/alesof/workspaces/UT/02_bumpcroc/rpc/")
matplotlib.use("TkAgg")

st_idx = 10

# read pkl data & save the data in containers
time = []
phase = []

joint_pos = []
joint_vel = []
joint_pos_des = []
joint_vel_des = []
joint_trq_des = []
com_des = []

with open("/home/alesof/workspaces/UT/02_bumpcroc/rpc/data_example/g1_step_over_knee_knocker_latest_fix.pkl", "rb") as file:
    while True:
        try:
            data = pickle.load(file)
            # time.append(data["time"])
            # phase.append(data["phase"])
            com_des.append(data["center_of_mass"])

            # joint_pos.append(data["joint_positions"])
            # joint_vel.append(data["joint_vel_des"])
            # joint_pos_des.append(data["joint_pos_des"])
            # joint_vel_des.append(data["joint_vel_des"])
            # joint_trq_des.append(data["joint_trq_des"])
        except EOFError:
            break

# time = np.array(time)[st_idx:]
# phase = np.array(phase)[st_idx:]

# joint_pos = np.stack(joint_pos, axis=0)[st_idx:, :]
# joint_vel = np.stack(joint_vel, axis=0)[st_idx:, :]
# joint_pos_des = np.stack(joint_pos_des, axis=0)[st_idx:, :]
# joint_vel_des = np.stack(joint_vel_des, axis=0)[st_idx:, :]
# joint_trq_des = np.stack(joint_trq_des, axis=0)[st_idx:, :]
# com_des = np.stack(com_des, axis=0)[st_idx:, :]

# Print the data for inspection
print("Center of Mass Desired (com_des):")
print(com_des)

# plt.show()
