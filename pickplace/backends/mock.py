"""IDEALISED kinematic stand-in for MuJoCo, used only to exercise the pipeline logic
(perception -> planner -> controller -> eval harness) on machines without MuJoCo.

What is real here: the camera model and the depth image (exact analytic ray/box
intersection through the same CameraModel), so perception is tested against a true
geometric ground truth.

What is NOT real: contact physics.  A grasp "succeeds" by a hand-written geometric rule
(cube between the pads, yaw within 12 deg, lateral offset small), the held cube then
rigidly follows the gripper, and a released cube drops straight down upright.  Success
rates from this backend say nothing about MuJoCo friction, slip or contact behaviour.
"""
from __future__ import annotations

import numpy as np

from .. import constants as C
from ..scenario import Scenario
from .base import ArmCommand, ArmState

TAU = 0.03            # first-order lag of the arm tracking its command (s)
FINGER_SPEED = 0.15   # m/s of jaw opening change
ALIGN_TOL = np.radians(12.0)
PAD_LEN_TOL = 0.015   # max cube-centre offset along the pad direction
Z_TOL_FRAC = 0.5      # TCP must be within size*0.5 of the cube centre height


def box_ray_range(origins, dirs, centre, yaw, half):
    """Distance along unit `dirs` to a yaw-rotated axis-aligned cube; inf on miss.
    origins/dirs broadcast to (...,3); `half` is the half edge length."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o = (origins - centre) @ Rz.T
    d = dirs @ Rz.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    tmin = np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2))
    tmax = np.where(np.isnan(t2), np.inf, np.maximum(t1, t2))
    # rays parallel to a slab and outside it never hit
    parallel_out = (np.abs(d) < 1e-12) & (np.abs(o) > half)
    tnear = tmin.max(axis=-1)
    tfar = tmax.min(axis=-1)
    hit = (tnear <= tfar) & (tfar > 0) & ~parallel_out.any(axis=-1)
    return np.where(hit, np.maximum(tnear, 0.0), np.inf)


class MockBackend:
    name = "mock"
    ctrl_dt = 0.02

    def __init__(self):
        self.camera = C.make_camera()

    def reset(self, scn: Scenario) -> None:
        self.size = scn.cube_size
        self.obj_xy = np.array(scn.cube_xy, float)
        self.obj_yaw = float(scn.cube_yaw)
        self.obj_z = self.size / 2.0
        self.held = False
        self._rel = None
        self.pos = C.HOME_POS.astype(float).copy()
        self.yaw = float(C.HOME_YAW)
        self.width = float(C.HOME_WIDTH)

    # ---- sensing ---------------------------------------------------------
    def render_depth(self) -> np.ndarray:
        cam = self.camera
        dirs = cam.unit_rays_world()
        o = cam.t_wc
        with np.errstate(divide="ignore", invalid="ignore"):
            t_table = np.where(dirs[..., 2] < 0, -o[2] / dirs[..., 2], np.inf)
        centre = np.array([self.obj_xy[0], self.obj_xy[1], self.obj_z])
        t_box = box_ray_range(o, dirs, centre, self.obj_yaw, self.size / 2.0)
        t = np.minimum(t_table, t_box)
        depth = np.where(np.isfinite(t), t * (dirs @ cam.forward), 0.0)
        return depth

    def state(self) -> ArmState:
        return ArmState(self.pos.copy(), self.yaw, self.width)

    def object_pose(self) -> dict:
        return {"xy": self.obj_xy.copy(), "z": float(self.obj_z),
                "yaw": float(self.obj_yaw), "tilt_deg": 0.0}

    # ---- dynamics --------------------------------------------------------
    def _alignment(self):
        d = self.obj_xy - self.pos[:2]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        gx, gy = c * d[0] + s * d[1], -s * d[0] + c * d[1]   # cube centre in gripper frame
        dyaw = (self.obj_yaw - self.yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        z_ok = abs(self.pos[2] - self.obj_z) <= Z_TOL_FRAC * self.size
        return gx, gy, dyaw, z_ok

    def step(self, cmd: ArmCommand) -> None:
        dt = self.ctrl_dt
        a = 1.0 - np.exp(-dt / TAU)
        self.pos += (np.asarray(cmd.pos, float) - self.pos) * a
        self.yaw += (cmd.yaw - self.yaw) * a
        target_w = float(np.clip(cmd.width, 0.0, C.MAX_OPEN))
        w_new = self.width + np.clip(target_w - self.width, -FINGER_SPEED * dt, FINGER_SPEED * dt)

        if not self.held:
            gx, gy, dyaw, z_ok = self._alignment()
            # cube must start INSIDE the open jaws (checked with the pre-step width) ...
            inside = abs(gy) + self.size / 2 <= self.width / 2 + 0.002
            engaged = z_ok and abs(dyaw) <= ALIGN_TOL and abs(gx) <= PAD_LEN_TOL and inside
            if engaged and w_new <= self.size + 2 * abs(gy):
                # ... then the nearer pad touches it and pushes it to the jaw centre
                c, s = np.cos(self.yaw), np.sin(self.yaw)
                self.obj_xy = self.obj_xy - gy * np.array([-s, c])
                gy = 0.0
                w_new = max(w_new, self.size)
                if w_new <= self.size and target_w < self.size:   # squeezing -> held, aligned
                    self.held = True
                    k = np.round((self.obj_yaw - self.yaw) / (np.pi / 2)) * (np.pi / 2)
                    self._rel = (gx, k, self.obj_z - self.pos[2])
        else:
            if target_w > self.size + 0.004 and w_new > self.size + 0.004:   # released
                self.held = False
                self.obj_z = self.size / 2.0                                 # drops upright
            else:
                w_new = max(w_new, self.size)
        self.width = float(w_new)

        if self.held:
            gx, k, dz = self._rel
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            self.obj_xy = self.pos[:2] + np.array([c * gx, s * gx])
            self.obj_yaw = self.yaw + k
            self.obj_z = self.pos[2] + dz
