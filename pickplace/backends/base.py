"""Backend interface (duck-typed).  A backend must provide:

    name: str                      'mujoco' or 'mock'
    ctrl_dt: float                 seconds advanced per step()
    camera: CameraModel            the TRUE camera (perception gets a perturbed copy)
    reset(scenario)                build the world for this scenario
    render_depth() -> (H,W) z-depth, 0.0 = no hit.  Sees table + cube only, not the arm.
    state() -> ArmState            measured TCP position / wrist yaw / jaw width
    step(cmd: ArmCommand)          apply targets, advance ctrl_dt
    object_pose() -> dict          GROUND TRUTH {xy, z, yaw, tilt_deg}; evaluation only
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArmCommand:
    pos: object        # (3,) target TCP position
    yaw: float         # target wrist yaw
    width: float       # target jaw opening (0 = squeeze)


@dataclass
class ArmState:
    pos: np.ndarray
    yaw: float
    width: float
