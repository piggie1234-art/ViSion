import numpy as np

def rpy_to_rot_matrix(roll, pitch, yaw):
    """Convert RPY angles (in radians) to rotation matrix. In Z-Y-X order"""

    # Precompute cosines and sines of RPY angles
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    # Create the rotation matrix
    rot_matrix = np.array([
        [cos_yaw*cos_pitch, cos_yaw*sin_pitch*sin_roll - sin_yaw*cos_roll, cos_yaw*sin_pitch*cos_roll + sin_yaw*sin_roll],
        [sin_yaw*cos_pitch, sin_yaw*sin_pitch*sin_roll + cos_yaw*cos_roll, sin_yaw*sin_pitch*cos_roll - cos_yaw*sin_roll],
        [-sin_pitch, cos_pitch*sin_roll, cos_pitch*cos_roll]
    ])

    return rot_matrix

def transform_xyzabc2mat(xyzabc):
    mat = np.eye(4)
    a = np.radians(xyzabc[3])
    b = np.radians(xyzabc[4])
    c = np.radians(xyzabc[5])
    R = rpy_to_rot_matrix(a, b, c)
    mat[:3, :3] = R
    mat[:3, 3] = xyzabc[:3]
    return mat
