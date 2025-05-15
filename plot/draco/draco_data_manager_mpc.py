import zmq
import sys
import os

cwd = os.getcwd()
sys.path.append(cwd + "/build")
sys.path.append(cwd)

import pickle
import time
from typing import Optional

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

import ruamel.yaml as yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from messages.draco_pb2 import *
from plot.data_saver import DataSaver
from plot.protobuf_to_dict import protobuf_to_dict
from plot.helper import load_pkl_data


import meshcat
import meshcat_shapes
from plot import meshcat_utils as vis_tools

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--b_visualize", action="store_true")
parser.add_argument("--b_use_plotjuggler", action="store_true")
parser.add_argument("--b_play", action="store_true")
parser.add_argument("--filename", type=str, default=None)
args = parser.parse_args()

# ==========================================================================
# Misc
# ==========================================================================
des_com_traj_label = []
des_com_traj_proj_label = []
des_ori_traj_label = []
des_lf_pos_traj_label = []
des_rf_pos_traj_label = []
des_lf_ori_traj_label = []
des_rf_ori_traj_label = []

# ==========================================================================
# Socket initialize
# ==========================================================================
context = zmq.Context()
socket = context.socket(zmq.SUB)

b_using_kf_estimator = False
b_using_non_kf_estimator = False

SAVE_FREQ = 50  # hz


def check_if_kf_estimator(kf_pos, est_pos):
    global b_using_kf_estimator, b_using_non_kf_estimator

    # check if we have already set either the KF or non-KF flag to True
    if b_using_kf_estimator or b_using_non_kf_estimator:
        return

    # if both kf_pos and est_pos data are zero's, we have not entered standup
    if not (np.any(kf_pos) or np.any(est_pos)):
        return

    # otherwise, we can infer from the current kf_pos and est_pos data
    if np.any(kf_pos):
        b_using_kf_estimator = True
    else:
        b_using_non_kf_estimator = True


def add_com_velocities(msg_dict, last_msg_dict) -> None:
    ori_key = "kf_base_joint_ori" if b_using_kf_estimator else "est_base_joint_ori"
    if (
        (not b_using_kf_estimator and not b_using_non_kf_estimator)
        or ori_key not in msg_dict
        or "time" not in last_msg_dict
    ):
        return
    cur_ori = R.from_quat(msg_dict[ori_key])
    dt = msg_dict["time"] - last_msg_dict["time"]
    # add linvel velocity field
    if "act_com_pos" in last_msg_dict:
        world_vel = ((np.array(msg_dict["act_com_pos"]) - np.array(last_msg_dict["act_com_pos"])) / dt).tolist()
        msg_dict["com_lin_vel"] = cur_ori.apply(world_vel, inverse=True).tolist()
    if ori_key in last_msg_dict:
        prev_ori = R.from_quat(last_msg_dict[ori_key]).as_euler("xyz")
        # clamp to range [-pi, pi]
        ori_diff = (cur_ori.as_euler("xyz") - prev_ori + np.pi) % (2 * np.pi) - np.pi
        msg_dict["com_ang_vel"] = (ori_diff / dt).tolist()


def process_data(msg, last_msg_dict) -> dict:
    msg_dict = protobuf_to_dict(msg)
    add_com_velocities(msg_dict, last_msg_dict)
    # add CoM velocity field
    # if "act_com_pos" in last_msg_dict and "time" in last_msg_dict and base_ori is not None:
    #     dt = msg_dict["time"] - last_msg_dict["time"]
    #     world_vel = ((np.array(msg_dict["act_com_pos"]) - np.array(last_msg_dict["act_com_pos"])) / dt).tolist()
    #     msg_dict["com_vel"] = R.from_quat(base_ori).apply(world_vel, inverse=True).tolist()

    data_saver.history = msg_dict
    data_saver.advance()

    return msg_dict


def plot_trajectory(
    msg_dict: dict,
    pos_key: str,
    label: str,
    has_initialized: bool,
    *,
    ori_key: Optional[str] = None,
    project_z: bool = False,
    color: int = vis_tools.Color.RED,
) -> bool:
    use_ori = ori_key is not None
    if pos_key not in msg_dict or len(msg_dict[pos_key]) == 0:
        return
    if use_ori and (ori_key not in msg_dict or len(msg_dict[ori_key]) == 0):
        return
    if use_ori:
        assert len(msg_dict[pos_key]) == len(msg_dict[ori_key]), (
            f"Expected {pos_key} and {ori_key} to be the same length, got {len(msg_dict[pos_key])} and {len(msg_dict[ori_key])}"
        )
    for i, pos in enumerate(msg_dict[pos_key]):
        name = f"{label}_{i}" if not use_ori else f"{label}_pos_{i}"
        ori_name = f"{label}_ori_{i}"
        if not has_initialized:
            meshcat_shapes.point(
                viz.viewer[name],
                color=color,
                opacity=0.8,
                radius=0.01,
            )
            if use_ori:
                meshcat_shapes.frame(
                    viz.viewer[ori_name],
                    axis_length=0.1,
                    axis_thickness=0.005,
                    opacity=0.8,
                    origin_radius=0.01,
                )
        z = pos["z"] if not project_z else 0.0
        trans = meshcat.transformations.translation_matrix([pos["x"], pos["y"], z])
        viz.viewer[name].set_transform(trans)
        if use_ori:
            ori = msg_dict[ori_key][i]
            rot = meshcat.transformations.euler_matrix(ori["x"], ori["y"], ori["z"], axes="sxyz")
            viz.viewer[ori_name].set_transform(trans.dot(rot))
    # return true to indicate initialization is complete
    return True


if __name__ == "__main__":
    # YAML parse
    with open("config/draco/sim/pybullet/wbic/pnc.yaml", "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
            ip_address = config["ip_address"]
        except yaml.YAMLError as exc:
            print(exc)

    socket.connect(ip_address)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    if args.b_use_plotjuggler:
        pj_context = zmq.Context()
        pj_socket = pj_context.socket(zmq.PUB)
        pj_socket.bind("tcp://*:9872")

    # meshcat visualizer
    if args.b_visualize:
        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            "robot_model/draco/draco_modified.urdf",
            "robot_model/draco",
            pin.JointModelFreeFlyer(),
        )
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        try:
            viz.initViewer(open=True)
        except ImportError as err:
            print("Error while initializing the viewer. It seems you should install python meshcat")
            print(err)
            exit()
        viz.loadViewerModel(rootNodeName="draco3")
        vis_q = pin.neutral(model)

        # add other visualizations to viewer
        meshcat_shapes.point(viz.viewer["com_des"], color=vis_tools.Color.RED, opacity=1.0, radius=0.01)

        meshcat_shapes.point(viz.viewer["com"], color=vis_tools.Color.BLUE, opacity=1.0, radius=0.01)

        meshcat_shapes.point(
            viz.viewer["com_projected"],
            color=vis_tools.Color.BLACK,
            opacity=1.0,
            radius=0.01,
        )

        # add arrows visualizers to viewer
        vis_tools.add_arrow(viz.viewer, "grf_lf", color=[0, 0, 1])  #  BLUE
        vis_tools.add_arrow(viz.viewer, "grf_rf", color=[1, 0, 0])  #  RED

    if args.b_play:
        data = load_pkl_data(args.filename if args.filename is not None else "pnc.pkl")

    msg = pnc_msg()
    msg_dict = {}
    if not args.b_play:
        data_saver = DataSaver()

    b_com_pos_init = False
    b_com_proj_init = False
    b_lf_traj_init = False
    b_rf_traj_init = False

    while not args.b_play or len(data) > 0:
        if args.b_play:
            msg_dict = data.pop(0)
        else:
            # receive msg trough socket
            encoded_msg = socket.recv()
            msg.ParseFromString(encoded_msg)
            msg_dict = process_data(msg, msg_dict)

        #  publish back to plot juggler
        # if args.b_use_plotjuggler:
        # pj_socket.send_string(json.dumps(data_saver.history))

        if args.b_visualize:
            check_if_kf_estimator(msg_dict["kf_base_joint_pos"], msg_dict["est_base_joint_pos"])

            base_pos = msg_dict["kf_base_joint_pos" if b_using_kf_estimator else "est_base_joint_pos"]
            base_ori = msg_dict["kf_base_joint_ori" if b_using_kf_estimator else "est_base_joint_ori"]

            vis_q[0:3] = np.array(base_pos)
            vis_q[3:7] = np.array(base_ori)  # quaternion [x,y,z,w]
            vis_q[7:] = np.array(msg_dict["joint_positions"])

            viz.display(vis_q)

            trans = meshcat.transformations.translation_matrix(
                [msg_dict["des_com_pos"][0], msg_dict["des_com_pos"][1], msg_dict["des_com_pos"][2]]
            )
            viz.viewer["com_des"].set_transform(trans)

            trans = meshcat.transformations.translation_matrix(
                [msg_dict["act_com_pos"][0], msg_dict["act_com_pos"][1], msg_dict["act_com_pos"][2]]
            )
            viz.viewer["com"].set_transform(trans)

            trans = meshcat.transformations.translation_matrix(
                [msg_dict["des_com_pos"][0], msg_dict["des_com_pos"][1], 0.0]
            )
            viz.viewer["com_projected"].set_transform(trans)

            # visualize GRFs
            if msg_dict["phase"] != 1:
                vis_tools.grf_display(
                    viz.viewer["grf_lf"], msg_dict["lfoot_pos"], msg_dict["lfoot_ori"], msg_dict["lfoot_rf_cmd"]
                )
                vis_tools.grf_display(
                    viz.viewer["grf_rf"], msg_dict["rfoot_pos"], msg_dict["rfoot_ori"], msg_dict["rfoot_rf_cmd"]
                )

            b_com_pos_init = plot_trajectory(
                msg_dict,
                "des_com_traj",
                "des_com",
                b_com_pos_init,
                ori_key="des_torso_ori_traj",
                color=vis_tools.Color.GREEN,
            )
            b_com_proj_init = plot_trajectory(
                msg_dict,
                "des_com_traj",
                "des_com_projected",
                b_com_proj_init,
                project_z=True,
                color=vis_tools.Color.BLUE,
            )
            b_lf_traj_init = plot_trajectory(
                msg_dict,
                "des_lf_pos_traj",
                "des_lf",
                b_lf_traj_init,
                ori_key="des_lf_ori_traj",
                color=vis_tools.Color.CYAN,
            )
            b_rf_traj_init = plot_trajectory(
                msg_dict,
                "des_rf_pos_traj",
                "des_rf",
                b_rf_traj_init,
                ori_key="des_rf_ori_traj",
                color=vis_tools.Color.YELLOW,
            )

            if args.b_play:
                if SAVE_FREQ > 0:
                    time.sleep(1.0 / SAVE_FREQ)
                else:
                    # timestep on Enter
                    input()
