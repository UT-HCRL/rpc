"""
Converts a pkl file to mcap file for visualization in Foxglove.
This assumes a specific set of parameters available in the pkl file,
such as time, base_pos, base_ori, joint_positions, icp_est, icp_des, etc.
"""

import os
import sys
import argparse
import pickle
import pinocchio as pin
import numpy as np

from scipy.spatial.transform import Rotation as R

cwd = os.getcwd()
sys.path.append(cwd)

from util.python_utils.util import so3_from_vec_to_vec


from mcap_protobuf.writer import Writer
from UI.visualization_toolbox import update_robot_transform, update_2d_transform, update_3d_transform, get_rgba, COLOR_RGBA_MAP, rot_to_quat
from google.protobuf.wrappers_pb2 import FloatValue, BoolValue, Int32Value
from foxglove_schemas_protobuf.Point3_pb2 import Point3
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate


def create_sphere_scene(scene_frame_id, rgba, sized=0.03):
    sphere_scene = SceneUpdate()
    sphere_entity = sphere_scene.entities.add()
    sphere_entity.frame_id = scene_frame_id
    sphere_model = sphere_entity.spheres.add()
    sphere_model.color.r = rgba[0]
    sphere_model.color.g = rgba[1]
    sphere_model.color.b = rgba[2]
    sphere_model.color.a = rgba[3]
    sphere_model.size.x = sized
    sphere_model.size.y = sized
    sphere_model.size.z = sized
    return sphere_scene

def create_arrow_scene(scene_frame_id, force_x, force_y, force_z, rgba, arrow_settings = [0.03, 0.1, 0.08], scaling = 1200.0):
    """
    Create an arrow scene for MCAP based on raw force components
    
    Args:
        scene_frame_id: Frame ID for the scene
        force_x, force_y, force_z: raw force components
        rgba: color array [r, g, b, a]
    """
    arrow_scene = SceneUpdate()
    arrow_entity = arrow_scene.entities.add()
    arrow_entity.frame_id = scene_frame_id
    
    arrow_model = arrow_entity.arrows.add()
    arrow_model.color.r = rgba[0]
    arrow_model.color.g = rgba[1] 
    arrow_model.color.b = rgba[2]
    arrow_model.color.a = rgba[3]
    
    force_dir = np.array([force_x, force_y, force_z])
    
    if np.all(force_dir != 0.0):
        force_norm = np.linalg.norm(force_dir)
        force_dir_norm = force_dir / force_norm
        force_magnitude = force_norm / scaling
        
        rot_ang = np.arccos(np.clip(force_dir_norm.dot(np.array([0, 0, 1])), -1.0, 1.0))
        
        if rot_ang > 1e-6:  # Avoid division by zero for parallel vectors
            rot_ax = np.cross(force_dir_norm, np.array([0, 0, 1]))
            rot_ax /= np.linalg.norm(rot_ax)
            
            ax_hat = np.array([
                [0, -rot_ax[2], rot_ax[1]],
                [rot_ax[2], 0, -rot_ax[0]],
                [-rot_ax[1], rot_ax[0], 0],
            ])
            R_rot_force = (
                np.eye(3)
                + np.sin(rot_ang) * ax_hat
                + (1 - np.cos(rot_ang)) * ax_hat @ ax_hat
            )
            quat_force = rot_to_quat(R_rot_force)
        else:
            quat_force = [0, 0, 0, 1]
    else:
        force_magnitude = 0.0
        quat_force = [0, 0, 0, 1]

    arrow_scale = force_magnitude
    arrow_model.shaft_length = arrow_scale
    arrow_model.shaft_diameter = arrow_settings[0]
    arrow_model.head_length = arrow_settings[1]
    arrow_model.head_diameter = arrow_settings[2]
    
    arrow_model.pose.position.x = 0.0
    arrow_model.pose.position.y = 0.0
    arrow_model.pose.position.z = 0.0
    arrow_model.pose.orientation.x = quat_force[0]
    arrow_model.pose.orientation.y = quat_force[1]
    arrow_model.pose.orientation.z = quat_force[2]
    arrow_model.pose.orientation.w = quat_force[3]
    
    return arrow_scene

def main():

    robot_name = "g1"
    urdf_path = "robot_model/g1/g1_29dof_lock_waist.urdf"
    package_path = "robot_model/g1"

    time = []
    base_pos, base_ori, joint_positions = [], [], []

    vis_3d_object_names = [
        "lfoot_pos", 
        "rfoot_pos", 
        "lfoot_ori",
        "rfoot_ori",
        "lhand_pos", 
        "rhand_pos", 
        "torso_des_pos", 
        "left_rubber_hand_des_pos", 
        "right_rubber_hand_des_pos", 
        "left_ankle_roll_des_pos", 
        "right_ankle_roll_des_pos", 
        "left_knee_des_pos", 
        "right_knee_des_pos",
        "left_ankle_roll_curr_pos",
        "right_ankle_roll_curr_pos",
        "left_knee_curr_pos",
        "right_knee_curr_pos",
        "left_rubber_hand_curr_pos",
        "right_rubber_hand_curr_pos",
        "torso_curr_pos",
        "lf_contact_force",
        "rf_contact_force",
        "lh_contact_force",
        "rh_contact_force"
    ]

    arrows_fddp_object_names = [
        "l_foot_rf",
        "r_foot_rf",
        "l_hand_rf",
        "r_hand_rf"
    ]

    arrows_sensor_object_names = [
        "lf_contact_force",
        "rf_contact_force",
        "lh_contact_force",
        "rh_contact_force"
    ]

    spheres_des_object_names = [
        "torso_des_pos", 
        "left_rubber_hand_des_pos", 
        "right_rubber_hand_des_pos", 
        "left_ankle_roll_des_pos", 
        "right_ankle_roll_des_pos", 
        "left_knee_des_pos", 
        "right_knee_des_pos"
    ]

    spheres_curr_object_names = [
        "left_ankle_roll_curr_pos",
        "right_ankle_roll_curr_pos",
        "left_knee_curr_pos",
        "right_knee_curr_pos",
        "left_rubber_hand_curr_pos",
        "right_rubber_hand_curr_pos",
        "torso_curr_pos"
    ]

    vis_horzon_object_names = [
        "xbound_costs",
        "com_costs",
        "xreg_costs",
        "ureg_costs",
        "left_hand_frame_costs",
        "right_hand_frame_costs",
        "left_ankle_frame_costs",
        "right_ankle_frame_costs",
        "left_knee_frame_costs",
        "right_knee_frame_costs",
        "torso_link_frame_costs",
        "left_hand_contact_costs",
        "right_hand_contact_costs",
        "left_foot_contact_costs",
        "right_foot_contact_costs",
    ]

    mpc_horizon = 4

    predicted_object_names = [
        "predicted_torso",
        "predicted_lhand",
        "predicted_rhand",
        "predicted_lankle",
        "predicted_rankle",
        "predicted_lknee",
        "predicted_rknee",
    ]

    single_value_names = [
        "b_fddp_feasible",
        "total_iterations",
        "solve_duration",
    ]

    vis_3d_dict = {}
    vis_horizon_dict = {}
    vis_des_spheres_dict = {}
    vis_curr_spheres_dict = {}
    vis_arrows_dict = {}
    vis_sensor_arrows_dict = {}
    predicted_object_dict = {}
    single_value_dict = {}

    for oname in vis_3d_object_names:
        vis_3d_dict[oname] = []

    for oname in vis_horzon_object_names:
        vis_horizon_dict[oname] = []
    
    for oname in spheres_des_object_names:
        vis_des_spheres_dict[oname] = []

    for oname in spheres_curr_object_names:
        vis_curr_spheres_dict[oname] = []
    
    for oname in arrows_fddp_object_names:
        for i in range(mpc_horizon):
            vis_arrows_dict[f"{oname}_{i}"] = []
            print(f"Debugging vis_arrows_dict[{oname}_{i}]: {vis_arrows_dict[f'{oname}_{i}']}")

    for oname in arrows_sensor_object_names:
        vis_sensor_arrows_dict[oname] = []

    # print(f"Debugging vis_sensor_arrows_dict: {vis_sensor_arrows_dict}")

    for oname in predicted_object_names:
        for i in range(mpc_horizon):
            predicted_object_dict[f"{oname}_{i}"] = []

    for oname in single_value_names:
        single_value_dict[oname] = []

    # Read and collect all data from pkl file
    with open(cwd + "/experiment_data/debug.pkl", "rb") as f:
        while True:
            try:
                d = pickle.load(f)
                time.append(d["time"])
                base_pos.append(d["est_base_joint_pos"])
                base_ori.append(d["est_base_joint_ori"])
                joint_positions.append(d["joint_positions"])

                for oname in vis_3d_object_names:
                    vis_3d_dict[oname].append(d[oname])

                for oname in vis_horzon_object_names: #costs
                    vis_horizon_dict[oname].append(d[oname])

                for oname in spheres_des_object_names: #des frame spheres
                    vis_des_spheres_dict[oname].append(d[oname])

                for oname in spheres_curr_object_names: #curr frame spheres
                    vis_curr_spheres_dict[oname].append(d[oname])
                
                for oname in arrows_fddp_object_names: #fddp arrows
                    for i in range(mpc_horizon):
                        vis_arrows_dict[f"{oname}_{i}"].append(d[f"{oname}_{i}"])

                for oname in arrows_sensor_object_names:
                    vis_sensor_arrows_dict[oname].append(d[oname])

                for oname in predicted_object_names: #predicted frame positions
                    for i in range(mpc_horizon):
                        predicted_object_dict[f"{oname}_{i}"].append(d[f"{oname}_{i}"])

                for oname in single_value_names: #fddp solve statistics
                    single_value_dict[oname].append(d[oname])


            except EOFError:
                break

    # Load pinocchio model and data of G1
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_path, package_path, pin.JointModelFreeFlyer())
    data, collision_data, visual_data = pin.createDatas(model, collision_model, visual_model)
    vis_q = pin.neutral(model)
    transform = FrameTransform()

    scenes_dict = {}
    for frame_id in spheres_des_object_names:
        scenes_dict[frame_id] = create_sphere_scene(frame_id, get_rgba("red"), sized=0.03)

    for frame_name in predicted_object_names:
        for i in range(mpc_horizon):
            alpha = np.maximum(1 - 0.2*i, 0.02)
            scenes_dict[f"{frame_name}_{i}"] = create_sphere_scene(f"{frame_name}_{i}", [0, 0, 1, alpha], sized=0.03)

    # send data to mcap file
    with open(cwd + "/experiment_data/" + robot_name + "_foxglove.mcap", "wb") as f, Writer(f) as mcap_writer:
        for i in range(len(time)):
            for oname, ovalue in single_value_dict.items():
                if isinstance(ovalue[i], bool):
                    val = BoolValue(value=ovalue[i])
                elif isinstance(ovalue[i], int):
                    val = Int32Value(value=ovalue[i])
                elif isinstance(ovalue[i], float):
                    val = FloatValue(value=ovalue[i])
                else:
                    raise ValueError(f"Unsupported type for {oname}: {type(ovalue[i])}")
                mcap_writer.write_message(
                    oname,
                    val,
                    int(time[i] * 1e9),
                    int(time[i] * 1e9),
                )
            # Update all transforms (to visualize URDF)
            vis_q[0:3] = np.array(base_pos[i])
            vis_q[3:7] = np.array(base_ori[i])  # quaternion [x,y,z,w]
            vis_q[7:] = np.array(joint_positions[i])
            # ===========================================
            # update transformations of all visual model objects
            pin.forwardKinematics(model, data, vis_q)
            pin.updateGeometryPlacements(model, data, visual_model, visual_data)
            pin.updateFramePlacements(model, data)
            for visual in visual_model.geometryObjects:
                update_robot_transform(visual, visual_data, visual_model, transform)
                mcap_writer.write_message(
                    "transforms", transform, int(time[i] * 1e9), int(time[i] * 1e9)
                )
                transform.rotation.Clear()
                transform.translation.Clear()
            

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
                mcap_writer.write_message(
                    "transforms", transform, int(time[i] * 1e9), int(time[i] * 1e9)
                )
                transform.rotation.Clear()
                transform.translation.Clear()
            
            for vname, vval in vis_3d_dict.items():
                mcap_writer.write_message(
                    vname,
                    Point3(x=vval[i][0], y=vval[i][1], z=vval[i][2]),
                    int(time[i] * 1e9),
                    int(time[i] * 1e9),
                )

            for vname, vval in vis_arrows_dict.items():
                mcap_writer.write_message(
                    vname,
                    Point3(x=vval[i][0], y=vval[i][1], z=vval[i][2]),
                    int(time[i] * 1e9),
                    int(time[i] * 1e9),
                )

            # for vname, vval in vis_sensor_arrows_dict.items():
            #     mcap_writer.write_message(
            #         vname,
            #         Point3(x=vval[i][0], y=vval[i][1], z=vval[i][2]),
            #         int(time[i] * 1e9),
            #         int(time[i] * 1e9),
            #     )

            for cname, cval in vis_horizon_dict.items():
                for knot_index, knot_value in enumerate(cval[i]):
                    name = f"{cname}_N_{knot_index}"
                    mcap_writer.write_message(
                        name,
                        FloatValue(value=knot_value),
                        int(time[i] * 1e9),
                        int(time[i] * 1e9),
                    )

            for vname, vval in vis_des_spheres_dict.items():
                update_3d_transform(vname, vval[i], transform)
                mcap_writer.write_message(
                    "transforms", transform, int(time[i] * 1e9), int(time[i] * 1e9)
                )
                transform.rotation.Clear()
                transform.translation.Clear()

            for vname, vval in predicted_object_dict.items():
                update_3d_transform(vname, vval[i], transform)
                mcap_writer.write_message(
                    "transforms", transform, int(time[i] * 1e9), int(time[i] * 1e9)
                )
                transform.rotation.Clear()
                transform.translation.Clear()

            for vname, vval in vis_curr_spheres_dict.items():
                update_3d_transform(vname, vval[i], transform)
                mcap_writer.write_message(
                    "transforms", transform, int(time[i] * 1e9), int(time[i] * 1e9)
                )
                transform.rotation.Clear()
                transform.translation.Clear()
            
            for scene_name, scene in scenes_dict.items():
                scene.entities[0].timestamp.FromNanoseconds(int(time[i] * 1e9))
                mcap_writer.write_message(
                    f"{scene_name}_marker", scene, int(time[i] * 1e9), int(time[i] * 1e9)
                )

            for arrow_name in arrows_fddp_object_names:

                if arrow_name == "l_foot_rf":
                    pos_data = vis_3d_dict["lfoot_pos"][i]
                    ori_data = vis_3d_dict["lfoot_ori"][i]
                elif arrow_name == "r_foot_rf":
                    pos_data = vis_3d_dict["rfoot_pos"][i]
                    ori_data = vis_3d_dict["rfoot_ori"][i]
                elif arrow_name == "l_hand_rf":
                    pos_data = vis_3d_dict["lhand_pos"][i]
                    ori_data = [0, 0, 0, 1]
                elif arrow_name == "r_hand_rf":
                    pos_data = vis_3d_dict["rhand_pos"][i]
                    ori_data = [0, 0, 0, 1]
                
                for knot_index in range(mpc_horizon):
                    knot_arrow_name = f"{arrow_name}_{knot_index}"
                    if knot_arrow_name in vis_arrows_dict and len(vis_arrows_dict[knot_arrow_name]) > i:
                        force_data = vis_arrows_dict[knot_arrow_name][i]
                        if np.all(np.array(force_data) != 0.0):
                            so3_up_to_ori = so3_from_vec_to_vec(np.array([1, 0, 0]), np.array(force_data))
                            q_cmd_arrow = rot_to_quat(so3_up_to_ori)
                            transform.rotation.x = q_cmd_arrow[0]
                            transform.rotation.y = q_cmd_arrow[1]
                            transform.rotation.z = q_cmd_arrow[2]
                            transform.rotation.w = q_cmd_arrow[3]
                        else:
                            transform.rotation.x = 0
                            transform.rotation.y = 0
                            transform.rotation.z = 0
                            transform.rotation.w = 1
                    else:
                        print(f"Warning: {knot_arrow_name} not found or index {i} out of range in vis_arrows_dict.")
                        continue

                    transform.parent_frame_id = "world"
                    transform.child_frame_id = knot_arrow_name

                    transform.translation.x = pos_data[0]
                    transform.translation.y = pos_data[1]
                    transform.translation.z = pos_data[2]

                    mcap_writer.write_message(
                        "transforms", transform, int(time[i] * 1e9), int(time[i] * 1e9)
                    )
                    transform.rotation.Clear()
                    transform.translation.Clear()

                    if np.all(np.array(force_data) < 1e-6):
                        arrow_scene = create_arrow_scene(
                            knot_arrow_name,
                            force_data[0],
                            force_data[1],
                            force_data[2],
                            [0.0, 0.0, 1.0, 0.0]  # Transparent arrow if no force
                        )
                    else:
                        arrow_scene = create_arrow_scene(
                            knot_arrow_name,
                            force_data[0],
                            force_data[1],
                            force_data[2],
                            get_rgba("s_blue" if knot_index == 0 else "blue")
                        )
                    arrow_scene.entities[0].timestamp.FromNanoseconds(int(time[i] * 1e9))
                    mcap_writer.write_message(
                        f"{knot_arrow_name}_arrow_marker", arrow_scene, int(time[i] * 1e9), int(time[i] * 1e9)
                    )

            for arrow_name in arrows_sensor_object_names:
                if arrow_name == "lf_contact_force":
                    pos_data = vis_3d_dict["lfoot_pos"][i]
                    ori_data = vis_3d_dict["lfoot_ori"][i]
                elif arrow_name == "rf_contact_force":
                    pos_data = vis_3d_dict["rfoot_pos"][i]
                    ori_data = vis_3d_dict["rfoot_ori"][i]
                elif arrow_name == "lh_contact_force":
                    pos_data = vis_3d_dict["lhand_pos"][i]
                    ori_data = [0, 0, 0, 1]
                elif arrow_name == "rh_contact_force":
                    pos_data = vis_3d_dict["rhand_pos"][i]
                    ori_data = [0, 0, 0, 1]

                force_data = vis_sensor_arrows_dict[arrow_name][i]
                transform.parent_frame_id = "world"
                transform.child_frame_id = arrow_name
                transform.translation.x = pos_data[0]
                transform.translation.y = pos_data[1]
                transform.translation.z = pos_data[2]
                transform.rotation.x = 0
                transform.rotation.y = 0
                transform.rotation.z = 0
                transform.rotation.w = 1
                mcap_writer.write_message(
                    "transforms", transform, int(time[i] * 1e9), int(time[i] * 1e9)
                )
                transform.rotation.Clear()
                transform.translation.Clear()
                if np.all(np.array(force_data) < 1e-6):
                    arrow_scene = create_arrow_scene(
                        arrow_name,
                        force_data[0],
                        force_data[1],
                        force_data[2],
                        [0.0, 0.0, 1.0, 0.0]  # Transparent arrow if no force
                    )
                else:
                    arrow_scene = create_arrow_scene(
                        arrow_name,
                        force_data[0],
                        force_data[1],
                        force_data[2],
                        get_rgba("s_yellow")
                    )
                arrow_scene.entities[0].timestamp.FromNanoseconds(int(time[i] * 1e9))
                mcap_writer.write_message(
                    f"{arrow_name}_arrow_marker", arrow_scene, int(time[i] * 1e9), int(time[i] * 1e9)
                )

        mcap_writer.finish()


if __name__ == "__main__":
    main()