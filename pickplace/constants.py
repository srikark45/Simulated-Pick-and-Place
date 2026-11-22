"""World layout and tuning constants.  Frame: z up, table top at z = 0 (metres)."""
from __future__ import annotations

import numpy as np

from .camera import CameraModel

TABLE_Z = 0.0

# Gripper (a 4-DOF gantry: x, y, z slides + wrist yaw, two-finger parallel jaw).
# The TCP is the point between the pad centres.
HOME_POS = np.array([0.0, 0.10, 0.30])
HOME_YAW = 0.0
HOME_WIDTH = 0.06      # jaw opening (distance between inner pad faces)
MAX_OPEN = 0.09
YAW_LIMIT = 2.0        # wrist range is +-2 rad
HOVER_Z = 0.15         # transit height of the TCP
MIN_TCP_Z = 0.015      # keeps the pads (half height 12 mm) off the table

# Fixed overhead-ish RGB-D camera.  Perception only gets an *estimate* of its pose.
CAM_POS = (0.0, -0.30, 0.62)
CAM_TARGET = (0.0, -0.07, 0.0)
CAM_FOVY_DEG = 40.0
IMG_W, IMG_H = 160, 120

# Randomisation regions (cube centre / place target centre, xy)
SPAWN_X = (-0.15, 0.15)
SPAWN_Y = (-0.18, 0.04)
PLACE_X = (-0.12, 0.12)
PLACE_Y = (0.11, 0.20)

# Success definition (evaluated on simulator ground truth, never on estimates)
PLACE_TOL = 0.02                 # final cube centre within 2 cm of target (xy)
TILT_TOL_DEG = 10.0              # cube must rest roughly flat
REST_Z_TOL = 0.005               # cube centre height within 5 mm of size/2


def make_camera() -> CameraModel:
    return CameraModel.look_at(CAM_POS, CAM_TARGET, CAM_FOVY_DEG, IMG_W, IMG_H)
