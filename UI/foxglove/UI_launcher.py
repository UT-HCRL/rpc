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
    "--visualizer", choices=["none", "meshcat", "foxglove"], default="meshcat"
)
parser.add_argument(
    "--robot", choices=["draco", "g1", "fixed_draco", "manipulator"], default="g1"
)
parser.add_argument("--hw_or_sim", choices=["hw", "sim"], default="sim")
args = parser.parse_args()

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

            Ry = R.from_euler("y", -np.pi / 2).as_matrix()


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

        left_hand_ref_viz, left_hand_ref_model = vis_tools.add_sphere(
            viz.viewer, "left_hand_ref", color=[1.0, 0.0, 0.0, 0.4]
        )
        left_hand_ref_q = pin.neutral(left_hand_ref_model)

        right_hand_ref_viz, right_hand_ref_model = vis_tools.add_sphere(
            viz.viewer, "right_hand_ref", color=[1.0, 0.0, 0.0, 0.4]
        )
        right_hand_ref_q = pin.neutral(right_hand_ref_model)

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
        pass

    elif visualize_type == "none":
        # save data in pkl file (typically, for Plotjuggler)
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
        data_saver.add("com_xy_weight", list(msg.com_xy_weight))
        data_saver.add("com_xy_kp", list(msg.com_xy_kp))
        data_saver.add("com_xy_kd", list(msg.com_xy_kd))
        data_saver.add("com_xy_ki", list(msg.com_xy_ki))
        data_saver.add("com_z_weight", msg.com_z_weight)
        data_saver.add("com_z_kp", msg.com_z_kp)
        data_saver.add("com_z_kd", msg.com_z_kd)
        data_saver.add("torso_ori_weight", list(msg.torso_ori_weight))
        data_saver.add("torso_ori_kp", list(msg.torso_ori_kp))
        data_saver.add("torso_ori_kd", list(msg.torso_ori_kd))
        data_saver.add("lf_pos_weight", list(msg.lf_pos_weight))
        data_saver.add("lf_pos_kp", list(msg.lf_pos_kp))
        data_saver.add("lf_pos_kd", list(msg.lf_pos_kd))
        data_saver.add("rf_pos_weight", list(msg.rf_pos_weight))
        data_saver.add("rf_pos_kp", list(msg.rf_pos_kp))
        data_saver.add("rf_pos_kd", list(msg.rf_pos_kd))
        data_saver.add("lf_ori_weight", list(msg.lf_ori_weight))
        data_saver.add("lf_ori_kp", list(msg.lf_ori_kp))
        data_saver.add("lf_ori_kd", list(msg.lf_ori_kd))
        data_saver.add("rf_ori_weight", list(msg.rf_ori_weight))
        data_saver.add("rf_ori_kp", list(msg.rf_ori_kp))
        data_saver.add("rf_ori_kd", list(msg.rf_ori_kd))
        data_saver.add("quat_world_local", list(msg.quat_world_local))
        data_saver.add("joint_pos_des", list(msg.joint_pos_des))
        data_saver.add("joint_vel_des", list(msg.joint_vel_des))
        data_saver.add("joint_trq_des", list(msg.joint_trq_des))

    data_saver.advance()

while True:
    # print("\nFLAG_B1")
    # receive msg through socket
    encoded_msg = socket.recv()
    msg.ParseFromString(encoded_msg)
    # print("FLAG_MSG")
    # if publishing raw messages, floating base estimates names are not important
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
            
        elif args.visualizer == "foxglove":
            # webbrowser.open('https://app.foxglove.dev/view?ds=foxglove-websocket&ds.url=ws%3A%2F%2Flocalhost%3A8765')
            th_fast = threading.Thread(target=asyncio.run(main()), args=())
            th_fast.start()
            # asyncio.run(main())

    else:  # if 'none' specified
        process_data_saver("none")

    # publish back to plot juggler
    # note: currently, this is not reached when using foxglove but the corresponding
    # ROS messages can be visualized within foxglove
    if args.b_use_plotjuggler:
        process_data_saver("none")
        pj_socket.send_string(json.dumps(data_saver.history))
