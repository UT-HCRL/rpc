import os
import numpy as np
import pinocchio as pin
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform

from scipy.spatial.transform import Rotation as R
from util.python_utils.util import rot_to_quat


def isMesh(geometry_object):
    """Check whether the geometry object contains a Mesh supported by MeshCat"""
    if geometry_object.meshPath == "":
        return False

    _, file_extension = os.path.splitext(geometry_object.meshPath)
    if file_extension.lower() in [".dae", ".obj", ".stl"]:
        return True

    return False

def update_robot_transform(
    visual: pin.GeometryModel,
    visual_data: pin.GeometryData,
    visual_model: pin.GeometryModel,
    transform: FrameTransform,
):
    parent_name = "world"
    # Get mesh pose.
    M = visual_data.oMg[visual_model.getGeometryId(visual.name)]
    # Manage scaling
    if isMesh(visual):
        scale = np.asarray(visual.meshScale).flatten()
        S = np.diag(np.concatenate((scale, [1.0])))
        T = np.array(M.homogeneous).dot(S)
    else:
        T = M.homogeneous
    visual_name = visual.name[:-2]
    transform.parent_frame_id = parent_name
    transform.child_frame_id = visual_name
    transform.translation.x = T[0][3]
    transform.translation.y = T[1][3]
    transform.translation.z = T[2][3]
    rot = T[:3, :3]
    q = rot_to_quat(rot)
    transform.rotation.x = q[0]
    transform.rotation.y = q[1]
    transform.rotation.z = q[2]
    transform.rotation.w = q[3]

def update_2d_transform(obj_name: str, pos_2d: np.ndarray, transform: FrameTransform):
    transform.parent_frame_id = "world"
    transform.child_frame_id = obj_name
    transform.translation.x = pos_2d[0]
    transform.translation.y = pos_2d[1]

def update_3d_transform(obj_name: str, pos_3d: np.ndarray, transform: FrameTransform):
    transform.parent_frame_id = "world"
    transform.child_frame_id = obj_name
    transform.translation.x = pos_3d[0]
    transform.translation.y = pos_3d[1]
    transform.translation.z = pos_3d[2]

COLOR_RGBA_MAP = {
    "red": [1.0, 0.0, 0.0, 0.5],
    "green": [0.0, 1.0, 0.0, 0.5],
    "blue": [0.0, 0.0, 1.0, 0.5],
    "yellow": [1.0, 1.0, 0.0, 0.5],
    "cyan": [0.0, 1.0, 1.0, 0.5],
    "magenta": [1.0, 0.0, 1.0, 0.5],
    "white": [1.0, 1.0, 1.0, 0.5],
    "black": [0.0, 0.0, 0.0, 0.5],
    "s_red": [1.0, 0.0, 0.0, 1.0],
    "s_green": [0.0, 1.0, 0.0, 1.0],
    "s_blue": [0.0, 0.0, 1.0, 1.0],
    "s_yellow": [1.0, 1.0, 0.0, 1.0],
    "s_cyan": [0.0, 1.0, 1.0, 1.0],
    "s_magenta": [1.0, 0.0, 1.0, 1.0],
    "s_white": [1.0, 1.0, 1.0, 1.0],
    "s_black": [0.0, 0.0, 0.0, 1.0],
}

def get_rgba(color_name):
    return COLOR_RGBA_MAP.get(color_name.lower(), [0.0, 0.0, 0.0, 0.5])

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