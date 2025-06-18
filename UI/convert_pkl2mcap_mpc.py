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

cwd = os.getcwd()
sys.path.append(cwd)

from mcap_protobuf.writer import Writer
from UI.visualization_toolbox import update_robot_transform, update_2d_transform, update_3d_transform, get_rgba, COLOR_RGBA_MAP
from google.protobuf.wrappers_pb2 import FloatValue
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
        "l_foot_rf", 
        "r_foot_rf", 
        "l_hand_rf", 
        "r_hand_rf"
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
    ]

    vis_3d_dict = {}
    vis_horizon_dict = {}
    vis_des_spheres_dict = {}
    vis_curr_spheres_dict = {}

    for oname in vis_3d_object_names:
        vis_3d_dict[oname] = []

    for oname in vis_horzon_object_names:
        vis_horizon_dict[oname] = []
    
    for oname in spheres_des_object_names:
        vis_des_spheres_dict[oname] = []

    for oname in spheres_curr_object_names:
        vis_curr_spheres_dict[oname] = []

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

    for frame_id in spheres_curr_object_names:
        scenes_dict[frame_id] = create_sphere_scene(frame_id, get_rgba("blue"), sized=0.03)

    # send data to mcap file
    with open(cwd + "/experiment_data/" + robot_name + "_foxglove.mcap", "wb") as f, Writer(f) as mcap_writer:
        for i in range(len(time)):
            # Update all transforms (to visualize URDF)
            vis_q[0:3] = np.array(base_pos[i])
            vis_q[3:7] = np.array(base_ori[i])  # quaternion [x,y,z,w]
            vis_q[7:] = np.array(joint_positions[i])
            # ===========================================
            # update transformations of all visual model objects
            pin.forwardKinematics(model, data, vis_q)
            pin.updateGeometryPlacements(model, data, visual_model, visual_data)
            for visual in visual_model.geometryObjects:
                update_robot_transform(visual, visual_data, visual_model, transform)
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

        mcap_writer.finish()


if __name__ == "__main__":
    main()