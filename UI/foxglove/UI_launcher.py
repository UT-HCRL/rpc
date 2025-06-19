import os
import sys
import zmq
import time
import ruamel.yaml as yaml
import numpy as np

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/build")

from plot.data_saver import *
import pinocchio as pin
import json
import argparse
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("--b_use_plotjuggler", type=bool, default=False)
parser.add_argument(
    "--visualizer", choices=["none", "meshcat", "foxglove"], default="none"
)
parser.add_argument(
    "--robot", choices=["draco", "g1", "fixed_draco", "manipulator"], default="g1"
)
parser.add_argument("--hw_or_sim", choices=["hw", "sim"], default="sim")
args = parser.parse_args()

crocoddyl_forces = ["l_foot_rf", "r_foot_rf", "l_hand_rf", "r_hand_rf"]

viz_des_trajectories = {
    "left_ankle_roll_des_pos": ([1, 0, 1, 1], [0.03, 0.03, 0.03]),
    "right_ankle_roll_des_pos": ([1, 0, 1, 1], [0.03, 0.03, 0.03]),
    "left_rubber_hand_des_pos": ([1, 0, 1, 1], [0.03, 0.03, 0.03]),
    "right_rubber_hand_des_pos": ([1, 0, 1, 1], [0.03, 0.03, 0.03]),
    "torso_des_pos": ([1, 0, 1, 1], [0.03, 0.03, 0.03]),
    "left_knee_des_pos": ([1, 0, 1, 1], [0.03, 0.03, 0.03]),
    "right_knee_des_pos": ([1, 0, 1, 1], [0.03, 0.03, 0.03]),
}

viz_curr_trajectories = {
    "left_ankle_roll_curr_pos": ([1, 0, 0, 1], [0.03, 0.03, 0.03]),
    "right_ankle_roll_curr_pos": ([1, 0, 0, 1], [0.03, 0.03, 0.03]),
    "left_rubber_hand_curr_pos": ([1, 0, 0, 1], [0.03, 0.03, 0.03]),
    "right_rubber_hand_curr_pos": ([1, 0, 0, 1], [0.03, 0.03, 0.03]),
    "torso_curr_pos": ([1, 0, 0, 1], [0.03, 0.03, 0.03]),
    "left_knee_curr_pos": ([1, 0, 0, 1], [0.03, 0.03, 0.03]),
    "right_knee_curr_pos": ([1, 0, 0, 1], [0.03, 0.03, 0.03])
}

def rot_to_quat(rot):
    """
    Parameters
    ----------
    rot (np.array): SO3

    Returns
    -------
    quat (np.array): scalar last quaternion

    """
    return np.copy(R.from_matrix(rot).as_quat())

if args.visualizer == "meshcat":
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat
    from plot import meshcat_utils as vis_tools
elif args.visualizer == "foxglove":
    # Foxglove dependencies
    import UI.foxglove.control_widgets as foxglove_ctrl
    import asyncio
    import threading
    from base64 import b64encode
    from foxglove_websocket.server import FoxgloveServer
    from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
    from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
    from mcap_protobuf.schema import build_file_descriptor_set

    # local tools to manage Foxglove scenes
    from plot.foxglove_utils import SceneChannel, ShapeScene
    from UI.visualization_toolbox import update_robot_transform

    scene_schema = b64encode(
        build_file_descriptor_set(SceneUpdate).SerializeToString()
    ).decode("ascii")
    frame_schema = b64encode(
        build_file_descriptor_set(FrameTransform).SerializeToString()
    ).decode("ascii")

##==========================================================================
##Socket initialize
##==========================================================================
context = zmq.Context()
socket = context.socket(zmq.SUB)

b_using_kf_estimator = False
b_using_non_kf_estimator = False

##==========================================================================
## Load config file
##==========================================================================
with open("config/" + args.robot + "/INTERFACE.yaml", "r") as iface_yaml:
    try:
        config = yaml.safe_load(iface_yaml)
        env = config["test_env_name"]
        wbc = config["whole_body_controller"]
    except yaml.YAMLError as exc:
        print(exc)

pnc_path = (
    "config/" + args.robot + "/" + args.hw_or_sim + "/" + env + "/" + wbc + "/pnc.yaml"
)

async def main():
    async with FoxgloveServer(
        "0.0.0.0",
        8765,
        "Visualization server",
        capabilities=["parameters", "parametersSubscribe"],
    ) as server:
        
        tf_chan_id = await SceneChannel(
            False,
            "transforms",
            "protobuf",
            FrameTransform.DESCRIPTOR.full_name,
            frame_schema,
        ).add_chan(server)
        
        arrows_scene = ShapeScene()
        shape_params = ["arrows", [0, 0, 1, 0.5], [0.03, 0.1, 0.08]]
        for name in crocoddyl_forces:
            arrows_scene.add_shape(name, *shape_params)

        des_traj_scene = ShapeScene()
        for frame_name, (rgba_color, scale) in viz_des_trajectories.items():
            des_traj_scene.add_shape(
                frame_name, "spheres", rgba_color, scale
            )
        
        curr_traj_scene = ShapeScene()
        for frame_name, (rgba_color, scale) in viz_curr_trajectories.items():
            curr_traj_scene.add_shape(
                frame_name, "spheres", rgba_color, scale
            )

        mpc_horizon = 2 #FIXME: can this be dynamic or should we load the horizon from yaml?

        # MPC costs
        mpc_tot_iter = await SceneChannel(
            True, "total_iterations", "json", "total_iterations", ["value"]
        ).add_chan(server)
        mpc_xreg_costs = await SceneChannel(
            True, "xreg_costs", "json", "xreg_costs", [f"N_{i}" for i in range(mpc_horizon)] 
        ).add_chan(server)
        mpc_ureg_costs = await SceneChannel(
            True, "ureg_costs", "json", "ureg_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_xbound_costs = await SceneChannel(
            True, "xbound_costs", "json", "xbound_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_com_costs = await SceneChannel(
            True, "com_costs", "json", "com_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_left_hand_contact_costs = await SceneChannel(
            True, "left_hand_contact_costs", "json", "left_hand_contact_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_right_hand_contact_costs = await SceneChannel(
            True, "right_hand_contact_costs", "json", "right_hand_contact_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_left_foot_contact_costs = await SceneChannel(
            True, "left_foot_contact_costs", "json", "left_foot_contact_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_right_foot_contact_costs = await SceneChannel(
            True, "right_foot_contact_costs", "json", "right_foot_contact_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)

        # MPC desired positions
        mpc_torso_des_pos = await SceneChannel(
            True, "torso_des_pos", "json", "torso_des_pos", [f"N_0_{axis}" for axis in ["x", "y", "z"]]
        ).add_chan(server)
        mpc_right_hand_des_pos = await SceneChannel(
            True, "right_rubber_hand_des_pos", "json", "right_rubber_hand_des_pos", [f"N_0_{axis}" for axis in ["x", "y", "z"]]
        ).add_chan(server)
        mpc_left_hand_des_pos = await SceneChannel(
            True, "left_rubber_hand_des_pos", "json", "left_rubber_hand_des_pos", [f"N_0_{axis}" for axis in ["x", "y", "z"]]
        ).add_chan(server)
        
        # MPC frame costs
        mpc_right_hand_frame_costs = await SceneChannel(
            True, "right_hand_frame_costs", "json", "right_hand_frame_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_left_hand_frame_costs = await SceneChannel(
            True, "left_hand_frame_costs", "json", "left_hand_frame_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_left_ankle_frame_costs = await SceneChannel(
            True, "left_ankle_frame_costs", "json", "left_ankle_frame_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_right_ankle_frame_costs = await SceneChannel(
            True, "right_ankle_frame_costs", "json", "right_ankle_frame_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_left_knee_frame_costs = await SceneChannel(
            True, "left_knee_frame_costs", "json", "left_knee_frame_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_right_knee_frame_costs = await SceneChannel(
            True, "right_knee_frame_costs", "json", "right_knee_frame_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)
        mpc_torso_link_frame_costs = await SceneChannel(
            True, "torso_link_frame_costs", "json", "torso_link_frame_costs", [f"N_{i}" for i in range(mpc_horizon)]
        ).add_chan(server)

        mpc_fddp_feasible = await SceneChannel(
            True, "fddp_feasible", "json", "fddp_feasible", ["value"]
        ).add_chan(server)

        grfs_chan_id = await SceneChannel(
            True, "GRFs", "json", "normal",
            [
                "l_foot_rf_x",
                "l_foot_rf_y",
                "l_foot_rf_z",
                "r_foot_rf_x",
                "r_foot_rf_y",
                "r_foot_rf_z",
                "l_hand_rf_x",
                "l_hand_rf_y",
                "l_hand_rf_z",
                "r_hand_rf_x",
                "r_hand_rf_y",
                "r_hand_rf_z",
            ]
        ).add_chan(server)
        normS_chan_id = await SceneChannel(
            False,
            "normal_viz",
            "protobuf",
            SceneUpdate.DESCRIPTOR.full_name,
            scene_schema,
        ).add_chan(server)
        
        des_traj_chan_id = await SceneChannel(
            False,
            "des_traj",
            "protobuf",
            SceneUpdate.DESCRIPTOR.full_name,
            scene_schema,
        ).add_chan(server)

        lf_pos_chan_id = await SceneChannel(
            True, "lf_pos", "json", "lf_pos", ["x", "y", "z"]
        ).add_chan(server)
        rf_pos_chan_id = await SceneChannel(
            True, "rf_pos", "json", "rf_pos", ["x", "y", "z"]
        ).add_chan(server)
        lh_pos_chan_id = await SceneChannel(
            True, "lh_pos", "json", "lh_pos", ["x", "y", "z"]
        ).add_chan(server)
        rh_pos_chan_id = await SceneChannel(
            True, "rh_pos", "json", "rh_pos", ["x", "y", "z"]
        ).add_chan(server)

        # Send the FrameTransform every frame to update the model's position
        transform = FrameTransform()

        print("foxglove websocket initiated")

        while True:
            tasks = []  # Scenes to synchronously update

            # receive msg trough socket
            encoded_msg = socket.recv()
            msg.ParseFromString(encoded_msg)

            await asyncio.sleep(0.01)
            now = time.time_ns()
            transform.timestamp.FromNanoseconds(now)

            # Get mesh pose.
            if b_using_kf_estimator:
                base_pos = msg.kf_base_joint_pos
                base_ori = msg.kf_base_joint_ori
            else:
                base_pos = msg.est_base_joint_pos
                base_ori = msg.est_base_joint_ori
            vis_q = pin.neutral(model)
            vis_q[0:3] = np.array(base_pos)
            vis_q[3:7] = np.array(base_ori)  # quaternion [x,y,z,w]
            vis_q[7:] = np.array(msg.joint_positions)

            json_bytes = json.dumps(
                {
                    "x": list(msg.lfoot_pos)[0],
                    "y": list(msg.lfoot_pos)[1],
                    "z": list(msg.lfoot_pos)[2],
                }
            ).encode("utf8")
            await server.send_message(lf_pos_chan_id, now, json_bytes)

            await server.send_message(
                rf_pos_chan_id,
                now,
                json.dumps(
                    {
                        "x": list(msg.rfoot_pos)[0],
                        "y": list(msg.rfoot_pos)[1],
                        "z": list(msg.rfoot_pos)[2],
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                lh_pos_chan_id,
                now,
                json.dumps(
                    {
                        "x": list(msg.lhand_pos)[0],
                        "y": list(msg.lhand_pos)[1],
                        "z": list(msg.lhand_pos)[2],
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                rh_pos_chan_id,
                now,
                json.dumps(
                    {
                        "x": list(msg.rhand_pos)[0],
                        "y": list(msg.rhand_pos)[1],
                        "z": list(msg.rhand_pos)[2],
                    }
                ).encode("utf8"),
            )

            # TODO: Create a dictionary to shorten the code avoiding repetition of await
            await server.send_message(
                mpc_tot_iter,
                now,
                json.dumps(
                    {
                        "value": msg.total_iterations,
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_xreg_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.xreg_costs[i] for i in range(len(msg.xreg_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_ureg_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.ureg_costs[i] for i in range(len(msg.ureg_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_xbound_costs,
                now,
                json.dumps(
                    {
                        **{
                            f"N_{i}": msg.xbound_costs[i]
                            for i in range(len(msg.xbound_costs))
                        },
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_com_costs,
                now,
                json.dumps(
                    { 
                        **{
                            f"N_{i}": msg.com_costs[i]
                            for i in range(len(msg.com_costs))
                        },
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_left_hand_contact_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.left_hand_contact_costs[i] for i in range(len(msg.left_hand_contact_costs))},
                    }
                ).encode("utf8")
            )

            await server.send_message(
                mpc_left_foot_contact_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.left_foot_contact_costs[i] for i in range(len(msg.left_foot_contact_costs))},
                    }
                ).encode("utf8")
            )

            await server.send_message(
                mpc_right_foot_contact_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.right_foot_contact_costs[i] for i in range(len(msg.right_foot_contact_costs))},
                    }
                ).encode("utf8")
            )

            await server.send_message(
                mpc_right_hand_contact_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.right_hand_contact_costs[i] for i in range(len(msg.right_hand_contact_costs))},
                    }
                ).encode("utf8")
            )

            await server.send_message(
                mpc_right_hand_frame_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.right_hand_frame_costs[i] for i in range(len(msg.right_hand_frame_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_left_hand_frame_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.left_hand_frame_costs[i] for i in range(len(msg.left_hand_frame_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_left_ankle_frame_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.left_ankle_frame_costs[i] for i in range(len(msg.left_ankle_frame_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_right_ankle_frame_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.right_ankle_frame_costs[i] for i in range(len(msg.right_ankle_frame_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_left_knee_frame_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.left_knee_frame_costs[i] for i in range(len(msg.left_knee_frame_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_right_knee_frame_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.right_knee_frame_costs[i] for i in range(len(msg.right_knee_frame_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_torso_link_frame_costs,
                now,
                json.dumps(
                    {
                        **{f"N_{i}": msg.torso_link_frame_costs[i] for i in range(len(msg.torso_link_frame_costs))},
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_torso_des_pos,
                now,
                json.dumps(
                    {
                        "N_0_x": msg.torso_des_pos.x,
                        "N_0_y": msg.torso_des_pos.y,
                        "N_0_z": msg.torso_des_pos.z,
                    }
                ).encode("utf8"),
            )

            await server.send_message(
                mpc_fddp_feasible,
                now,
                json.dumps(
                    {
                        "value": msg.b_fddp_feasible,
                    }
                ).encode("utf8"),
            )

            for hand, channel in [
                (msg.right_rubber_hand_des_pos, mpc_right_hand_des_pos),
                (msg.left_rubber_hand_des_pos, mpc_left_hand_des_pos),
            ]:
                await server.send_message(
                    channel,
                    now,
                    json.dumps(
                        {
                            "N_0_x": hand.x,
                            "N_0_y": hand.y,
                            "N_0_z": hand.z,
                        }
                    ).encode("utf8"),
                )

            if hasattr(msg, "l_foot_rf") and hasattr(msg, "r_foot_rf") and hasattr(msg, "l_hand_rf") and hasattr(msg, "r_hand_rf") and msg.l_foot_rf and msg.r_foot_rf and msg.l_hand_rf and msg.r_hand_rf:
                await server.send_message(
                    grfs_chan_id,
                    now,
                    json.dumps(
                        {
                            "l_foot_rf_x": msg.l_foot_rf.x,
                            "l_foot_rf_y": msg.l_foot_rf.y,
                            "l_foot_rf_z": msg.l_foot_rf.z,
                            "r_foot_rf_x": msg.r_foot_rf.x,
                            "r_foot_rf_y": msg.r_foot_rf.y,
                            "r_foot_rf_z": msg.r_foot_rf.z,
                            "l_hand_rf_x": msg.l_hand_rf.x,
                            "l_hand_rf_y": msg.l_hand_rf.y,
                            "l_hand_rf_z": msg.l_hand_rf.z,
                            "r_hand_rf_x": msg.r_hand_rf.x,
                            "r_hand_rf_y": msg.r_hand_rf.y,
                            "r_hand_rf_z": msg.r_hand_rf.z,
                        }
                    ).encode("utf8"),
                )

            # update mesh positions
            pin.forwardKinematics(model, data, vis_q)
            pin.updateGeometryPlacements(model, data, visual_model, visual_data)
            for visual in visual_model.geometryObjects:
                update_robot_transform(visual, visual_data, visual_model, transform)
                await server.send_message(
                    tf_chan_id, now, transform.SerializeToString()
                )
                transform.rotation.Clear()
                transform.translation.Clear()
            
            pin.updateFramePlacements(model, data)
            for fid, m_frame in enumerate(model.frames):
                frame_name = model.frames[fid].name
                transform.parent_frame_id = "world"
                transform.child_frame_id = frame_name
                transform.translation.x = data.oMf[fid].translation[0]
                transform.translation.y = data.oMf[fid].translation[1]
                transform.translation.z = data.oMf[fid].translation[2]
                rot = data.oMf[fid].rotation
                q = rot_to_quat(rot)
                transform.rotation.x = q[0]
                transform.rotation.y = q[1]
                transform.rotation.z = q[2]
                transform.rotation.w = q[3]
                await server.send_message(
                    tf_chan_id, now, transform.SerializeToString()
                )
                transform.rotation.Clear()
                transform.translation.Clear()

            Ry = R.from_euler("y", -np.pi / 2).as_matrix()

            if hasattr(msg, "l_foot_rf") and hasattr(msg, "r_foot_rf") and msg.l_foot_rf and msg.r_foot_rf:
                for obj in crocoddyl_forces:
                    transform.parent_frame_id = "world"
                    transform.child_frame_id = obj
                    transform.timestamp.FromNanoseconds(now)
                    # show aligned at the center of respective foot sole
                    if obj == "l_foot_rf":
                        if np.linalg.norm(msg.lfoot_ori) == 0:
                            lfoot_ori = [0, 0, 0, 1] 
                        else:
                            lfoot_ori = msg.lfoot_ori
                        R_foot = R.from_quat(lfoot_ori).as_matrix()
                        transform.translation.x = msg.lfoot_pos[0]
                        transform.translation.y = msg.lfoot_pos[1]
                        transform.translation.z = msg.lfoot_pos[2]
                    elif obj == "r_foot_rf":
                        if np.linalg.norm(msg.rfoot_ori) == 0:
                            rfoot_ori = [0, 0, 0, 1]
                        else:
                            rfoot_ori = msg.rfoot_ori
                        R_foot = R.from_quat(rfoot_ori).as_matrix()
                        transform.translation.x = msg.rfoot_pos[0]
                        transform.translation.y = msg.rfoot_pos[1]
                        transform.translation.z = msg.rfoot_pos[2]
                    elif obj == "l_hand_rf":
                        if np.linalg.norm(msg.lhand_ori) == 0:
                            lhand_ori = [0, 0, 0, 1]
                        else:
                            lhand_ori = msg.lhand_ori
                        R_foot = R.from_quat(lhand_ori).as_matrix()
                        transform.translation.x = msg.lhand_pos[0]
                        transform.translation.y = msg.lhand_pos[1]
                        transform.translation.z = msg.lhand_pos[2]
                    elif obj == "r_hand_rf":
                        if np.linalg.norm(msg.rhand_ori) == 0:
                            rhand_ori = [0, 0, 0, 1]
                        else:
                            rhand_ori = msg.rhand_ori
                        R_foot = R.from_quat(rhand_ori).as_matrix()
                        transform.translation.x = msg.rhand_pos[0]
                        transform.translation.y = msg.rhand_pos[1]
                        transform.translation.z = msg.rhand_pos[2]

                    # rotate transform since arrow points in +x direction
                    q_cmd_arrow = rot_to_quat(Ry)
                    transform.rotation.x = q_cmd_arrow[0]
                    transform.rotation.y = q_cmd_arrow[1]
                    transform.rotation.z = q_cmd_arrow[2]
                    transform.rotation.w = q_cmd_arrow[3]

                    if obj in crocoddyl_forces:
                        force_dir = np.array([getattr(msg, obj).x, getattr(msg, obj).y, getattr(msg, obj).z])

                        if np.all(force_dir != 0.0):
                            force_norm = np.linalg.norm(force_dir)

                            # compute axis and angle of rotation to align z with force direction
                            force_dir /= force_norm
                            rot_ang = np.arccos(force_dir.dot(np.array([0, 0, 1])))
                            rot_ax = np.cross(force_dir, np.array([0, 0, 1]))
                            rot_ax /= np.linalg.norm(rot_ax)
                            ax_hat = np.array(
                                [
                                    [0, -rot_ax[2], rot_ax[1]],
                                    [rot_ax[2], 0, -rot_ax[0]],
                                    [-rot_ax[1], rot_ax[0], 0],
                                ]
                            )
                            R_rot_force = (
                                np.eye(3)
                                + np.sin(rot_ang) * ax_hat
                                + (1 - np.cos(rot_ang)) * ax_hat @ ax_hat
                            )
                            quat_force = rot_to_quat(R_rot_force)

                            # force scale
                            force_magnitude = force_norm / 1200.0
                
                            arrows_scene.scale(obj, quat_force, force_magnitude, now)
                            tasks.append(
                                server.send_message(tf_chan_id, now, transform.SerializeToString())
                            )
                            await server.send_message(
                                normS_chan_id, now, arrows_scene.serialized_msg(obj)
                            )
            
            for frame_name in viz_des_trajectories.keys():
                # Build the transform message
                transform.parent_frame_id = "world"
                transform.child_frame_id = frame_name
                transform.timestamp.FromNanoseconds(now)
                
                pos_msg = getattr(msg, frame_name)
                transform.translation.x = pos_msg.x
                transform.translation.y = pos_msg.y
                transform.translation.z = pos_msg.z
                
                # Update the pose for this frame in the scene
                des_traj_scene.update(frame_name, now)

                # Send messages
                tasks.append(server.send_message(tf_chan_id, now, transform.SerializeToString()))
                tasks.append(server.send_message(des_traj_chan_id, now, des_traj_scene.serialized_msg(frame_name)))
            
            for frame_name in viz_curr_trajectories.keys():
                # Build the transform message
                transform.parent_frame_id = "world"
                transform.child_frame_id = frame_name
                transform.timestamp.FromNanoseconds(now)

                pos_msg = getattr(msg, frame_name)
                transform.translation.x = pos_msg.x
                transform.translation.y = pos_msg.y
                transform.translation.z = pos_msg.z

                # Update the pose for this frame in the scene
                curr_traj_scene.update(frame_name, now)

                # Send messages
                tasks.append(server.send_message(tf_chan_id, now, transform.SerializeToString()))
                tasks.append(server.send_message(des_traj_chan_id, now, curr_traj_scene.serialized_msg(frame_name)))
            process_data_saver("foxglove")
            await asyncio.gather(*tasks)


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


##YAML parse
with open(pnc_path) as yaml_file:
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

data_saver = DataSaver()

#
# Visualizer Settings
#
if args.visualizer != "none":
    # both meshcat and foxglove make use of Pinocchio model and model data
    if args.robot == "draco":
        from messages.draco_pb2 import *

        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            "robot_model/draco/draco_modified.urdf",
            "robot_model/draco",
            pin.JointModelFreeFlyer(),
        )
    elif args.robot == "g1":
        from messages.g1_pb2 import *

        model, collision_model, visual_model = pin.buildModelsFromUrdf(
            "robot_model/g1/g1_29dof_lock_waist.urdf",
            "robot_model/g1",
            pin.JointModelFreeFlyer(),
        )
    else:
        raise NotImplementedError(f"Specify location of URDF of {args.robot}")

    msg = pnc_msg()

    data, collision_data, visual_data = pin.createDatas(
        model, collision_model, visual_model
    )
    vis_q = pin.neutral(model)

    # define and initialize elements to visualize
    if args.visualizer == "meshcat":
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        try:
            viz.initViewer(open=True)
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install python meshcat"
            )
            print(err)
            exit()
        viz.loadViewerModel(rootNodeName=args.robot)

        # add other visualizations to viewer
        left_hand_viz, left_hand_model = vis_tools.add_sphere(
            viz.viewer, "left_hand_des", color=[1.0, 0.0, 0.0, 0.4]
        )
        left_hand_q = pin.neutral(left_hand_model)

        right_hand_viz, right_hand_model = vis_tools.add_sphere(
            viz.viewer, "right_hand_des", color=[1.0, 0.0, 0.0, 0.4]
        )
        right_hand_q = pin.neutral(right_hand_model)

        torso_des_viz, torso_des_model = vis_tools.add_sphere(
            viz.viewer, "torso_des_pos", color=[1.0, 0.0, 0.0, 0.4]
        )
        torso_des_q = pin.neutral(torso_des_model)

        left_ankle_ref_viz, left_ankle_ref_model = vis_tools.add_sphere(
            viz.viewer, "left_ankle_roll_des_pos", color=[1.0, 0.0, 0.0, 0.4]
        )
        left_ankle_ref_q = pin.neutral(left_ankle_ref_model)

        right_ankle_ref_viz, right_ankle_ref_model = vis_tools.add_sphere(
            viz.viewer, "right_ankle_roll_des_pos", color=[1.0, 0.0, 0.0, 0.4]
        )
        right_ankle_ref_q = pin.neutral(right_ankle_ref_model)

        left_knee_ref_viz, left_knee_ref_model = vis_tools.add_sphere(
            viz.viewer, "left_knee_des_pos", color=[1.0, 0.0, 0.0, 0.4]
        )
        left_knee_ref_q = pin.neutral(left_knee_ref_model)

        right_knee_ref_viz, right_knee_ref_model = vis_tools.add_sphere(
            viz.viewer, "right_knee_des_pos", color=[1.0, 0.0, 0.0, 0.4]
        )
        right_knee_ref_q = pin.neutral(right_knee_ref_model)

        com_des_pos_viz, com_des_pos_model = vis_tools.add_sphere(
            viz.viewer, "com_des_pos", color=[1.0, 0.0, 0.0, 0.4]
        )
        com_des_pos_q = pin.neutral(com_des_pos_model)

        # Current values
        left_hand_curr_viz, left_hand_curr_model = vis_tools.add_sphere(
            viz.viewer, "left_rubber_hand_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        left_hand_curr_q = pin.neutral(left_hand_curr_model)

        right_hand_curr_viz, right_hand_curr_model = vis_tools.add_sphere(
            viz.viewer, "right_rubber_hand_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        right_hand_curr_q = pin.neutral(right_hand_curr_model)

        torso_curr_viz, torso_curr_model = vis_tools.add_sphere(
            viz.viewer, "torso_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        torso_curr_q = pin.neutral(torso_curr_model)

        left_ankle_curr_viz, left_ankle_curr_model = vis_tools.add_sphere(
            viz.viewer, "left_ankle_roll_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        left_ankle_curr_q = pin.neutral(left_ankle_curr_model)

        right_ankle_curr_viz, right_ankle_curr_model = vis_tools.add_sphere(
            viz.viewer, "right_ankle_roll_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        right_ankle_curr_q = pin.neutral(right_ankle_curr_model)

        left_knee_curr_viz, left_knee_curr_model = vis_tools.add_sphere(
            viz.viewer, "left_knee_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        left_knee_curr_q = pin.neutral(left_knee_curr_model)

        right_knee_curr_viz, right_knee_curr_model = vis_tools.add_sphere(
            viz.viewer, "right_knee_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        right_knee_curr_q = pin.neutral(right_knee_curr_model)

        com_curr_pos_viz, com_curr_pos_model = vis_tools.add_sphere(
            viz.viewer, "com_curr_pos", color=[0.0, 0.0, 1.0, 0.4]
        )
        com_curr_pos_q = pin.neutral(com_curr_pos_model)
else:
    from messages.g1_pb2 import *
    msg = pnc_msg()

def process_data_saver(visualize_type):
    if visualize_type == "meshcat":
        # save data in pkl file (saved to replay data)
        data_saver.add("time", msg.time)
        data_saver.add("phase", msg.phase)
        data_saver.add("est_base_joint_pos", list(msg.est_base_joint_pos))
        data_saver.add("est_base_joint_ori", list(msg.est_base_joint_ori))
        data_saver.add("kf_base_joint_pos", list(msg.kf_base_joint_pos))
        data_saver.add("kf_base_joint_ori", list(msg.kf_base_joint_ori))
        data_saver.add("joint_positions", list(msg.joint_positions))
        data_saver.add("des_com_pos", list(msg.des_com_pos))
        data_saver.add("act_com_pos", list(msg.act_com_pos))
        data_saver.add("lfoot_pos", list(msg.lfoot_pos))
        data_saver.add("rfoot_pos", list(msg.rfoot_pos))
        data_saver.add("lfoot_ori", list(msg.lfoot_ori))
        data_saver.add("rfoot_ori", list(msg.rfoot_ori))
        data_saver.add("lfoot_rf_cmd", list(msg.lfoot_rf_cmd))
        data_saver.add("rfoot_rf_cmd", list(msg.rfoot_rf_cmd))
        data_saver.add("b_lfoot", msg.b_lfoot)
        data_saver.add("b_rfoot", msg.b_rfoot)
        data_saver.add("lfoot_volt_normal_raw", msg.lfoot_volt_normal_raw)
        data_saver.add("rfoot_volt_normal_raw", msg.rfoot_volt_normal_raw)
        data_saver.add("lfoot_rf_normal", msg.lfoot_rf_normal)
        data_saver.add("rfoot_rf_normal", msg.rfoot_rf_normal)
        data_saver.add("lfoot_rf_normal_filt", msg.lfoot_rf_normal_filt)
        data_saver.add("rfoot_rf_normal_filt", msg.rfoot_rf_normal_filt)
        data_saver.add("est_icp", list(msg.est_icp))
        data_saver.add("des_icp", list(msg.des_icp))
        data_saver.add("des_cmp", list(msg.des_cmp))
        data_saver.add("quat_world_local", list(msg.quat_world_local))
        data_saver.add("joint_pos_des", list(msg.joint_pos_des))
        data_saver.add("joint_vel_des", list(msg.joint_vel_des))
        data_saver.add("joint_trq_des", list(msg.joint_trq_des))

    elif visualize_type == "foxglove":
        data_saver.add("time", msg.time)
        data_saver.add("est_base_joint_pos", list(msg.est_base_joint_pos))
        data_saver.add("est_base_joint_ori", list(msg.est_base_joint_ori))
        data_saver.add("joint_positions", list(msg.joint_positions))
        data_saver.add("lfoot_pos", list(msg.lfoot_pos))
        data_saver.add("rfoot_pos", list(msg.rfoot_pos))
        data_saver.add("lfoot_ori", list(msg.lfoot_ori))
        data_saver.add("rfoot_ori", list(msg.rfoot_ori))
        data_saver.add("lhand_pos", list(msg.lhand_pos))
        data_saver.add("rhand_pos", list(msg.rhand_pos))
        data_saver.add("xreg_costs", list(msg.xreg_costs))
        data_saver.add("ureg_costs", list(msg.ureg_costs))
        data_saver.add("xbound_costs", list(msg.xbound_costs))
        data_saver.add("com_costs", list(msg.com_costs))
        data_saver.add("left_hand_contact_costs", list(msg.left_hand_contact_costs))
        data_saver.add("left_foot_contact_costs", list(msg.left_foot_contact_costs))
        data_saver.add("right_foot_contact_costs", list(msg.right_foot_contact_costs))
        data_saver.add("right_hand_contact_costs", list(msg.right_hand_contact_costs))
        data_saver.add("right_hand_frame_costs", list(msg.right_hand_frame_costs))
        data_saver.add("left_hand_frame_costs", list(msg.left_hand_frame_costs))
        data_saver.add("left_ankle_frame_costs", list(msg.left_ankle_frame_costs))
        data_saver.add("right_ankle_frame_costs", list(msg.right_ankle_frame_costs))
        data_saver.add("left_knee_frame_costs", list(msg.left_knee_frame_costs))
        data_saver.add("right_knee_frame_costs", list(msg.right_knee_frame_costs))
        data_saver.add("torso_link_frame_costs", list(msg.torso_link_frame_costs))
        data_saver.add("l_foot_rf", [msg.l_foot_rf.x, msg.l_foot_rf.y, msg.l_foot_rf.z])
        data_saver.add("r_foot_rf", [msg.r_foot_rf.x, msg.r_foot_rf.y, msg.r_foot_rf.z])
        data_saver.add("l_hand_rf", [msg.l_hand_rf.x, msg.l_hand_rf.y, msg.l_hand_rf.z])
        data_saver.add("r_hand_rf", [msg.r_hand_rf.x, msg.r_hand_rf.y, msg.r_hand_rf.z])
        data_saver.add("b_fddp_feasible", msg.b_fddp_feasible)
        data_saver.add("total_iterations", msg.total_iterations)
        data_saver.add("solve_duration", msg.solve_duration)

        for frame_name in viz_des_trajectories.keys():
            pos_msg = getattr(msg, frame_name)
            data_saver.add(f"{frame_name}", [pos_msg.x, pos_msg.y, pos_msg.z])
        for frame_name in viz_curr_trajectories.keys():
            pos_msg = getattr(msg, frame_name)
            data_saver.add(f"{frame_name}", [pos_msg.x, pos_msg.y, pos_msg.z])

    elif visualize_type == "none":
        data_saver.add("time", msg.time)
        data_saver.add("est_base_joint_pos", list(msg.est_base_joint_pos))
        data_saver.add("est_base_joint_ori", list(msg.est_base_joint_ori))
        data_saver.add("joint_positions", list(msg.joint_positions))
        data_saver.add("lfoot_pos", list(msg.lfoot_pos))
        data_saver.add("rfoot_pos", list(msg.rfoot_pos))
        data_saver.add("lfoot_ori", list(msg.lfoot_ori))
        data_saver.add("rfoot_ori", list(msg.rfoot_ori))
        data_saver.add("lhand_pos", list(msg.lhand_pos))
        data_saver.add("rhand_pos", list(msg.rhand_pos))
        data_saver.add("xreg_costs", list(msg.xreg_costs))
        data_saver.add("ureg_costs", list(msg.ureg_costs))
        data_saver.add("xbound_costs", list(msg.xbound_costs))
        data_saver.add("com_costs", list(msg.com_costs))
        data_saver.add("left_hand_contact_costs", list(msg.left_hand_contact_costs))
        data_saver.add("left_foot_contact_costs", list(msg.left_foot_contact_costs))
        data_saver.add("right_foot_contact_costs", list(msg.right_foot_contact_costs))
        data_saver.add("right_hand_contact_costs", list(msg.right_hand_contact_costs))
        data_saver.add("right_hand_frame_costs", list(msg.right_hand_frame_costs))
        data_saver.add("left_hand_frame_costs", list(msg.left_hand_frame_costs))
        data_saver.add("left_ankle_frame_costs", list(msg.left_ankle_frame_costs))
        data_saver.add("right_ankle_frame_costs", list(msg.right_ankle_frame_costs))
        data_saver.add("left_knee_frame_costs", list(msg.left_knee_frame_costs))
        data_saver.add("right_knee_frame_costs", list(msg.right_knee_frame_costs))
        data_saver.add("torso_link_frame_costs", list(msg.torso_link_frame_costs))
        data_saver.add("l_foot_rf", [msg.l_foot_rf.x, msg.l_foot_rf.y, msg.l_foot_rf.z])
        data_saver.add("r_foot_rf", [msg.r_foot_rf.x, msg.r_foot_rf.y, msg.r_foot_rf.z])
        data_saver.add("l_hand_rf", [msg.l_hand_rf.x, msg.l_hand_rf.y, msg.l_hand_rf.z])
        data_saver.add("r_hand_rf", [msg.r_hand_rf.x, msg.r_hand_rf.y, msg.r_hand_rf.z])

        for frame_name in viz_des_trajectories.keys():
            pos_msg = getattr(msg, frame_name)
            data_saver.add(f"{frame_name}", [pos_msg.x, pos_msg.y, pos_msg.z])
        for frame_name in viz_curr_trajectories.keys():
            pos_msg = getattr(msg, frame_name)
            data_saver.add(f"{frame_name}", [pos_msg.x, pos_msg.y, pos_msg.z])


    data_saver.advance()

while True:

    encoded_msg = socket.recv()
    msg.ParseFromString(encoded_msg)

    if args.visualizer != "none":
        check_if_kf_estimator(msg.kf_base_joint_pos, msg.est_base_joint_pos)

        if b_using_kf_estimator:
            base_pos = msg.kf_base_joint_pos
            base_ori = msg.kf_base_joint_ori
        else:
            base_pos = msg.est_base_joint_pos
            base_ori = msg.est_base_joint_ori

        vis_q[0:3] = np.array(base_pos)
        vis_q[3:7] = np.array(base_ori)  # quaternion [x,y,z,w]
        vis_q[7:] = np.array(msg.joint_positions)

        if args.visualizer == "meshcat":
            process_data_saver("meshcat")

            # update visualizer viewers
            viz.display(vis_q)

            if hasattr(msg, "right_rubber_hand_curr_pos") and msg.right_rubber_hand_curr_pos:
                right_hand_curr_q[:3] = np.array([msg.right_rubber_hand_curr_pos.x, msg.right_rubber_hand_curr_pos.y, msg.right_rubber_hand_curr_pos.z])
                right_hand_curr_viz.display(right_hand_curr_q)

            if hasattr(msg, "left_rubber_hand_curr_pos") and msg.left_rubber_hand_curr_pos:
                left_hand_curr_q[:3] = np.array([msg.left_rubber_hand_curr_pos.x, msg.left_rubber_hand_curr_pos.y, msg.left_rubber_hand_curr_pos.z])
                left_hand_curr_viz.display(left_hand_curr_q)

            if hasattr(msg, "torso_curr_pos") and msg.torso_curr_pos:
                torso_curr_q[:3] = np.array([msg.torso_curr_pos.x, msg.torso_curr_pos.y, msg.torso_curr_pos.z])
                torso_curr_viz.display(torso_curr_q)

            if hasattr(msg, "left_ankle_roll_curr_pos") and msg.left_ankle_roll_curr_pos:
                left_ankle_curr_q[:3] = np.array([msg.left_ankle_roll_curr_pos.x, msg.left_ankle_roll_curr_pos.y, msg.left_ankle_roll_curr_pos.z])
                left_ankle_curr_viz.display(left_ankle_curr_q)

            if hasattr(msg, "right_ankle_roll_curr_pos") and msg.right_ankle_roll_curr_pos:
                right_ankle_curr_q[:3] = np.array([msg.right_ankle_roll_curr_pos.x, msg.right_ankle_roll_curr_pos.y, msg.right_ankle_roll_curr_pos.z])
                right_ankle_curr_viz.display(right_ankle_curr_q)

            if hasattr(msg, "left_knee_curr_pos") and msg.left_knee_curr_pos:
                left_knee_curr_q[:3] = np.array([msg.left_knee_curr_pos.x, msg.left_knee_curr_pos.y, msg.left_knee_curr_pos.z])
                left_knee_curr_viz.display(left_knee_curr_q)

            if hasattr(msg, "right_knee_curr_pos") and msg.right_knee_curr_pos:
                right_knee_curr_q[:3] = np.array([msg.right_knee_curr_pos.x, msg.right_knee_curr_pos.y, msg.right_knee_curr_pos.z])
                right_knee_curr_viz.display(right_knee_curr_q)

            if hasattr(msg, "right_rubber_hand_des_pos") and msg.right_rubber_hand_des_pos:
                right_hand_q[:3] = np.array([msg.right_rubber_hand_des_pos.x, msg.right_rubber_hand_des_pos.y, msg.right_rubber_hand_des_pos.z])
                right_hand_viz.display(right_hand_q)

            if hasattr(msg, "left_rubber_hand_des_pos") and msg.left_rubber_hand_des_pos:
                left_hand_q[:3] = np.array([msg.left_rubber_hand_des_pos.x, msg.left_rubber_hand_des_pos.y, msg.left_rubber_hand_des_pos.z])
                left_hand_viz.display(left_hand_q)
            
            if hasattr(msg, "torso_des_pos") and msg.torso_des_pos:
                torso_des_q[:3] = np.array([msg.torso_des_pos.x, msg.torso_des_pos.y, msg.torso_des_pos.z])
                torso_des_viz.display(torso_des_q)

            if hasattr(msg, "left_ankle_roll_des_pos") and msg.left_ankle_roll_des_pos:
                left_ankle_ref_q[:3] = np.array([msg.left_ankle_roll_des_pos.x, msg.left_ankle_roll_des_pos.y, msg.left_ankle_roll_des_pos.z])
                left_ankle_ref_viz.display(left_ankle_ref_q)
            
            if hasattr(msg, "right_ankle_roll_des_pos") and msg.right_ankle_roll_des_pos:
                right_ankle_ref_q[:3] = np.array([msg.right_ankle_roll_des_pos.x, msg.right_ankle_roll_des_pos.y, msg.right_ankle_roll_des_pos.z])
                right_ankle_ref_viz.display(right_ankle_ref_q)

            if hasattr(msg, "left_knee_des_pos") and msg.left_knee_des_pos:
                left_knee_ref_q[:3] = np.array(
                    [msg.left_knee_des_pos.x, msg.left_knee_des_pos.y, msg.left_knee_des_pos.z]
                )
                left_knee_ref_viz.display(left_knee_ref_q)
            
            if hasattr(msg, "right_knee_des_pos") and msg.right_knee_des_pos:
                right_knee_ref_q[:3] = np.array([msg.right_knee_des_pos.x, msg.right_knee_des_pos.y, msg.right_knee_des_pos.z])
                right_knee_ref_viz.display(right_knee_ref_q)

            if hasattr(msg, "com_des_pos") and msg.com_des_pos:
                com_des_pos_q[:3] = np.array([msg.com_des_pos.x, msg.com_des_pos.y, msg.com_des_pos.z])
                com_des_pos_viz.display(com_des_pos_q)
            
            if hasattr(msg, "com_curr_pos") and msg.com_curr_pos:
                com_curr_pos_q[:3] = np.array([msg.com_curr_pos.x, msg.com_curr_pos.y, msg.com_curr_pos.z])
                com_curr_pos_viz.display(com_curr_pos_q)
            
        elif args.visualizer == "foxglove":
            th_fast = threading.Thread(target=asyncio.run(main()), args=())
            th_fast.start()

    else:  # if 'none' specified
        process_data_saver("none")

    # publish back to plot juggler
    # note: currently, this is not reached when using foxglove but the corresponding
    # ROS messages can be visualized within foxglove
    if args.b_use_plotjuggler:
        process_data_saver("none")
        pj_socket.send_string(json.dumps(data_saver.history))
