import os
import sys
from typing import List

cwd = os.getcwd()
sys.path.append(cwd)

import numpy as np
import pinocchio as pin
from meshcat.geometry import TriangularMeshGeometry

# Pinocchio Meshcat
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf

# Python-Meshcat
from meshcat.animation import Animation
from pinocchio.visualize.meshcat_visualizer import isMesh


violet = [1.0, 0.0, 1.0, 0.3]


class Color(object):
    RED = 0xFF0000
    GREEN = 0x00FF00
    BLUE = 0x0000FF
    GREY = 0x888888
    BLACK = 0x000000
    CYAN = 0x00FFFF
    YELLOW = 0xFFFF00
    VIOLET = 0x8F00FF


def add_arrow(meshcat_visualizer, obj_name, color=[1, 0, 0], height=0.1):
    arrow_shaft = g.Cylinder(height, 0.01)
    arrow_head = g.Cylinder(0.04, 0.04, radiusTop=0.001, radiusBottom=0.04)
    material = g.MeshPhongMaterial()
    material.color = (
        int(color[0] * 255) * 256**2 + int(color[1] * 255) * 256 + int(color[2] * 255)
    )

    meshcat_visualizer[obj_name].set_object(arrow_shaft, material)
    meshcat_visualizer[obj_name]["head"].set_object(arrow_head, material)


def add_arrow_composite(meshcat_visualizer, obj_name, color=[1, 0, 0], height=0.1):
    arrow_shaft = g.Cylinder(height, 0.01)
    arrow_head = g.Cylinder(0.04, 0.04, radiusTop=0.001, radiusBottom=0.04)
    material = g.MeshPhongMaterial()
    material.color = (
        int(color[0] * 255) * 256**2 + int(color[1] * 255) * 256 + int(color[2] * 255)
    )

    shaft_offset = tf.translation_matrix([0.0, height / 2.0, 0.0])
    meshcat_visualizer[obj_name]["arrow/shaft"].set_object(arrow_shaft, material)
    meshcat_visualizer[obj_name]["arrow/head"].set_object(arrow_head, material)
    meshcat_visualizer[obj_name]["arrow/head"].set_transform(shaft_offset)


def add_footsteps(
    meshcat_visualizer,
    obj_name,
    footsteps_to_add,
    color=[1, 0, 0],
    foot_length=0.25,
    foot_width=0.15,
):
    # create footstep
    footstep = g.Box([foot_length, foot_width, 0.01])
    material = g.MeshPhongMaterial()
    material.color = (
        int(color[0] * 255) * 256**2 + int(color[1] * 255) * 256 + int(color[2] * 255)
    )
    material.opacity = 0.4

    # add all footsteps to visualizer
    for step in range(footsteps_to_add):
        meshcat_visualizer[obj_name]["step" + str(step)].set_object(footstep, material)


def add_sphere(
    parent_visualizer,
    node_name="sphere",
    urdf_path="robot_model/ground/sphere.urdf",
    visuals_path="robot_model/ground",
    color=[0.0, 0.0, 1.0, 0.5],
):
    sphere_model, sphere_collision_model, sphere_visual_model = pin.buildModelsFromUrdf(
        urdf_path, visuals_path, pin.JointModelFreeFlyer()
    )
    sphere_viz = MeshcatVisualizer(
        sphere_model, sphere_collision_model, sphere_visual_model
    )
    sphere_viz.initViewer(parent_visualizer)
    sphere_viz.loadViewerModel(rootNodeName=node_name, color=color)

    return sphere_viz, sphere_model


def add_coordiante_frame(frame_viz, name):
    arrow_height = 0.2
    add_arrow_composite(frame_viz, name + "/x", [1, 0, 0], arrow_height)
    add_arrow_composite(frame_viz, name + "/y", [0, 1, 0], arrow_height)
    add_arrow_composite(frame_viz, name + "/z", [0, 0, 1], arrow_height)

    arrow_offset_x = tf.translation_matrix([arrow_height / 2.0, 0.0, 0.0])
    arrow_offset_y = tf.translation_matrix([0.0, arrow_height / 2.0, 0.0])
    arrow_offset_z = tf.translation_matrix([0.0, 0.0, arrow_height / 2.0])
    tf_front = tf.rotation_matrix(-np.pi / 2.0, [0.0, 0.0, 1.0])
    tf_left = tf.identity_matrix()
    tf_up = tf.rotation_matrix(np.pi / 2.0, [1.0, 0.0, 0.0])

    # translate and rotate
    T_front = tf.concatenate_matrices(arrow_offset_x, tf_front)
    T_left = tf.concatenate_matrices(arrow_offset_y, tf_left)
    T_up = tf.concatenate_matrices(arrow_offset_z, tf_up)
    frame_viz[name + "/x"]["arrow"].set_transform(T_front)
    frame_viz[name + "/y"]["arrow"].set_transform(T_left)
    frame_viz[name + "/z"]["arrow"].set_transform(T_up)


def set_grf_default_position(meschat_visualizer, foot_position):
    arrow_height = np.array([0, 0, 0.05])
    arrow_head_offset = np.array([0, arrow_height[2] + 0.02 / 2, 0.0])
    T_rot = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    T_trans = tf.translation_matrix(foot_position + arrow_height)
    T_trans_arrow_head = tf.translation_matrix(arrow_head_offset)

    # first translate, then rotate
    T = tf.concatenate_matrices(T_trans, T_rot)
    meschat_visualizer.set_transform(T)
    meschat_visualizer["head"].set_transform(T_trans_arrow_head)


def get_rpy_from_world_to(foot_grf):
    foot_grf_normalized = foot_grf[3:] / np.linalg.norm(foot_grf[3:])
    roll = -np.arcsin(foot_grf_normalized[1])
    pitch = np.arctan2(foot_grf_normalized[0], foot_grf_normalized[2])

    return np.array([roll, pitch, 0.0])


def grf_display(meshcat_visualizer, foot_pos, foot_ori, foot_grf):
    # scale length
    scale = foot_grf[5] / 100.0  # 200 is about half weight
    S = tf.identity_matrix()
    S[1, 1] = scale  # y-axis corresponds to height (i.e., length) of cylinder

    # translate and rotate GRF vectors
    arrow_height = np.array([0, 0, (0.1 * scale) / 2.0])
    arrow_head_offset = np.array([0, 0.1 / 2, 0.0])
    # arrow_head_offset = np.array([0, arrow_height[2]+0.02/2, 0.0])
    T_arrow_vertical = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    grf_ori = get_rpy_from_world_to(foot_grf)
    T_grf_ori = tf.euler_matrix(grf_ori[0], grf_ori[1], grf_ori[2])
    T_trans = tf.translation_matrix(foot_pos + arrow_height)
    T_trans_arrow_head = tf.translation_matrix(arrow_head_offset)

    # first translate, then rotate, and scale
    T = tf.concatenate_matrices(T_trans, T_grf_ori, T_arrow_vertical, S)
    meshcat_visualizer.set_transform(T)
    meshcat_visualizer["head"].set_transform(T_trans_arrow_head)


def update_footstep(meshcat_visualizer, footstep_pos, footstep_ori):
    num_steps, _ = np.shape(footstep_pos)
    for step in range(num_steps):
        T_rot = tf.quaternion_matrix(footstep_ori[step])
        T_trans = tf.translation_matrix(footstep_pos[step])

        T = tf.concatenate_matrices(T_trans, T_rot)
        meshcat_visualizer["step" + str(step)].set_transform(T)


def display_visualizer_frames(meshcat_visualizer, frame):
    for visual in meshcat_visualizer.visual_model.geometryObjects:
        # Get mesh pose.
        M = meshcat_visualizer.visual_data.oMg[
            meshcat_visualizer.visual_model.getGeometryId(visual.name)
        ]
        # Manage scaling
        scale = np.asarray(visual.meshScale).flatten()
        S = np.diag(np.concatenate((scale, [1.0])))
        T = np.array(M.homogeneous).dot(S)
        # Update viewer configuration.
        frame[
            meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.VISUAL)
        ].set_transform(T)


def display_coordinate_frame(viz_name, frame_quat, viz_frame):
    # Note: frame_quat assumes convention [w,x,y,z], e.g., as
    # if coming from Eigen

    frame_quat = np.array(frame_quat)
    frame_quat = frame_quat[[3, 0, 1, 2]]
    tf_quat = tf.quaternion_matrix(frame_quat)
    viz_frame[viz_name].set_transform(tf_quat)

class MeshcatPinocchioAnimation:
    def __init__(self, pin_robot_model, collision_model, visual_model,
                 robot_data, visual_data, collision_data,
                 ctrl_freq=1000, save_freq=50):
        # self.robot = pin_robot_model
        self.robot_data = robot_data
        self.model = pin_robot_model
        self.robot_nq = pin_robot_model.nq
        self.viz = MeshcatVisualizer(self.model, collision_model, visual_model)
        try:
            self.viz.initViewer(open=True)
            self.viz.viewer.wait()
        except ImportError as err:
            print(
                "Error while initializing the viewer. It seems you should install Python meshcat"
            )
            print(err)
            sys.exit(0)
        self.viz.loadViewerModel(rootNodeName=self.model.name)

        # animation settings
        self.anim = Animation(default_framerate=ctrl_freq / save_freq)
        self.frame_idx = 0              # index of frame being saved in animation
        self.save_freq = save_freq      # display every save_freq simulation steps

        self.visual_model = visual_model
        self.visual_data = visual_data
        self.collision_model = collision_model
        self.collision_data = collision_data

    def add_robot(self, robot_name, pin_rob_model, collision_model, visual_model,
                  rob_position, rob_quaternion):
        viz = MeshcatVisualizer(pin_rob_model, collision_model, visual_model)
        viz.initViewer(self.viz.viewer)
        viz.loadViewerModel(rootNodeName=robot_name)

        rob_quaternion = rob_quaternion[[3, 0, 1, 2]]    # shift to wxyz used by tf
        tf_transl = tf.translation_matrix(rob_position)
        tf_rot = tf.quaternion_matrix(rob_quaternion)
        tf_pose = tf.concatenate_matrices(tf_transl, tf_rot)
        viz.viewer[robot_name].set_transform(tf_pose)

    def add_arrow(self, obj_name, color=[1, 0, 0], height=0.1):
        arrow_shaft = g.Cylinder(height, 0.01)
        arrow_head = g.Cylinder(0.04, 0.04, radiusTop=0.001, radiusBottom=0.04)
        material = g.MeshPhongMaterial()
        material.color = int(color[0] * 255) * 256 ** 2 + int(
            color[1] * 255) * 256 + int(color[2] * 255)

        arrow_offset = tf.translation_matrix([0., height/2., 0.])
        shaft_rotation = tf.rotation_matrix(np.pi/2., [1., 0., 0.])
        # arrow_vertical = tf.concatenate_matrices(arrow_offset, shaft_rotation)
        self.viz.viewer[obj_name]["arrow"].set_object(arrow_shaft, material)
        self.viz.viewer[obj_name]["arrow"].set_transform(shaft_rotation)
        self.viz.viewer[obj_name]["arrow/head"].set_object(arrow_head, material)
        self.viz.viewer[obj_name]["arrow/head"].set_transform(arrow_offset)

    def start_animation(self):
        self.frame_idx = 0

    def finish_animation(self):
        # save animation
        self.viz.viewer.set_animation(self.anim, play=False)

    def animate_single_collision(self, collision_name: str, collision_target: np.array):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_single_collision(frame, collision_target, collision_name)

    def animate_single_shape(self, viewer_name: str,
                             collision_target: np.ndarray):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_single_shape(frame, viewer_name, collision_target)

    def animate_target(self, end_effector_name, targets, color=None):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_targets(end_effector_name, targets, color, animation=True)

    def animate_frame(self, q):
        with self.anim.at_frame(self.viz.viewer, self.frame_idx) as frame:
            self.display_visualizer_frames(frame, q)

    def animation_step(self):
        self.frame_idx += 1

    def display_visualizer_frames(self, frame, q):
        meshcat_visualizer = self.viz

        geom_model = self.visual_model
        geom_data = self.visual_data

        pin.forwardKinematics(self.model, self.robot_data, q)
        pin.updateGeometryPlacements(self.model, self.robot_data,
                                     geom_model, geom_data)
        for visual in geom_model.geometryObjects:
            viewer_name = meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.VISUAL)
            # Get mesh pose.
            M = geom_data.oMg[geom_model.getGeometryId(visual.name)]
            # Manage scaling
            if isMesh(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                # S = visual.placement.homogeneous
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous
            # Update viewer configuration.
            frame[viewer_name].set_transform(T)

    def display_collisions(self, frame, q):
        meshcat_visualizer = self.viz

        geom_model = self.collision_model
        geom_data = self.collision_data

        pin.forwardKinematics(self.model, self.robot_data, q)
        pin.updateGeometryPlacements(self.model, self.robot_data,
                                     geom_model, geom_data)
        for visual in geom_model.geometryObjects:
            viewer_name = meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.COLLISION)
            # Get mesh pose.
            M = geom_data.oMg[geom_model.getGeometryId(visual.name)]
            # Manage scaling
            if isMesh(visual):
                scale = np.asarray(visual.meshScale).flatten()
                S = np.diag(np.concatenate((scale, [1.0])))
                # S = visual.placement.homogeneous
                T = np.array(M.homogeneous).dot(S)
            else:
                T = M.homogeneous
            # Update viewer configuration.
            frame[viewer_name].set_transform(T)

    def display_single_collision(self, frame, target, base_name):
        meshcat_visualizer = self.viz

        geom_model = self.collision_model
        for visual in geom_model.geometryObjects:
            if visual.name == base_name:
                viewer_name = meshcat_visualizer.getViewerNodeName(visual, pin.GeometryType.COLLISION)
                T = np.array([
                    [1.0, 0.0, 0.0, target[0]],
                    [0.0, 1.0, 0.0, target[1]],
                    [0.0, 0.0, 1.0, target[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ])
                frame[viewer_name].set_transform(T)
                return

    def display_single_shape(self,
                             frame,
                             viewer_name,
                             target_pos):
        frame[viewer_name].set_transform(target_pos)

    def display_targets(self, end_effector_name, targets, color=None, animation=False):
        if color is None:
            color = [1, 0, 0]
        material = g.MeshPhongMaterial()
        material.color = int(color[0] * 255) * 256 ** 2 + int(
            color[1] * 255) * 256 + int(color[2] * 255)
        material.opacity = 0.4
        for i, target in enumerate(targets):
            self.viz.viewer[end_effector_name + "/" + str(i)].set_object(g.Sphere(0.01),  material)
            Href = np.array(
                [
                    [1.0, 0.0, 0.0, target[0]],
                    [0.0, 1.0, 0.0, target[1]],
                    [0.0, 0.0, 1.0, target[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            if animation:
                self.anim.at_frame(self.viz.viewer, self.frame_idx)[end_effector_name + "/" + str(i)].set_transform(Href)
            else:
                self.viz.viewer[end_effector_name+"/" + str(i)].set_transform(Href)

    def hide_visuals(self, viz_list, b_visualize=False):
        for viz in viz_list:
            self.viz.viewer[viz].set_property("visible", b_visualize)

    def save_html(self, path, filename):
        viewer_html = self.viz.viewer.static_html()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + filename, "w") as f:
            f.write(viewer_html)