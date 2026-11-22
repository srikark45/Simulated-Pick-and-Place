"""MuJoCo backend (CPU only; no OpenGL needed).

Depth is produced by casting one `mj_ray` per pixel through the pinhole CameraModel, so
no renderer / EGL / OSMesa is required and the depth image is geometrically exact for
the camera model perception uses.  Only geom group 0 (table, cube) is visible to rays.

STATUS: written without MuJoCo available in the authoring environment, so this file has
NOT been executed.  Run `python scripts/smoke_mujoco.py` first.
"""
from __future__ import annotations

import numpy as np

from .. import constants as C
from ..camera import CameraModel
from ..scenario import Scenario
from .base import ArmCommand, ArmState
from .scene import ACTUATORS, JOINTS, build_mjcf


class MuJoCoBackend:
    name = "mujoco"
    ctrl_dt = 0.02

    def __init__(self):
        import mujoco  # imported lazily so the mock backend works without it
        self.mj = mujoco
        self.camera: CameraModel = C.make_camera()
        self._geomgroup = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    # ---- setup -----------------------------------------------------------
    def reset(self, scn: Scenario) -> None:
        mj = self.mj
        self.m = mj.MjModel.from_xml_string(build_mjcf(scn, self.camera))
        self.d = mj.MjData(self.m)
        self.substeps = int(round(self.ctrl_dt / self.m.opt.timestep))
        self.qadr = {j: int(self.m.joint(j).qposadr[0]) for j in JOINTS}
        self.aid = [int(self.m.actuator(a).id) for a in ACTUATORS]
        self.cube_bid = int(self.m.body("cube").id)
        home = np.append(C.HOME_POS, C.HOME_YAW)
        for j, v in zip(JOINTS[:4], home):
            self.d.qpos[self.qadr[j]] = v
        self.d.qpos[self.qadr["jfl"]] = self.d.qpos[self.qadr["jfr"]] = C.HOME_WIDTH / 2
        mj.mj_forward(self.m, self.d)
        self._check_camera()
        hold = ArmCommand(C.HOME_POS, C.HOME_YAW, C.HOME_WIDTH)
        for _ in range(15):                      # 0.3 s: let the cube settle on the table
            self.step(hold)

    def _check_camera(self) -> None:
        """Self-check: my pinhole extrinsics must match the camera MuJoCo built."""
        cid = int(self.m.camera("cam").id)
        pos = self.d.cam_xpos[cid]
        R = self.d.cam_xmat[cid].reshape(3, 3)   # columns: right, up, back
        want = np.stack([self.camera.R_wc[:, 0], -self.camera.R_wc[:, 1],
                         -self.camera.R_wc[:, 2]], axis=1)
        if not (np.allclose(pos, self.camera.t_wc, atol=1e-6)
                and np.allclose(R, want, atol=1e-6)):
            raise RuntimeError("MuJoCo camera pose disagrees with CameraModel")

    # ---- sensing ---------------------------------------------------------
    def render_depth(self) -> np.ndarray:
        cam, mj = self.camera, self.mj
        dirs = np.ascontiguousarray(cam.unit_rays_world())
        origin = np.asarray(cam.t_wc, dtype=np.float64)
        H, W = dirs.shape[:2]
        dist = np.zeros((H, W))
        gid = np.zeros(1, dtype=np.int32)
        for i in range(H):
            for j in range(W):
                t = mj.mj_ray(self.m, self.d, origin, dirs[i, j], self._geomgroup, 1, -1, gid)
                if t >= 0:
                    dist[i, j] = t
        return cam.range_to_zdepth(dist)

    def state(self) -> ArmState:
        q = self.d.qpos
        a = self.qadr
        return ArmState(np.array([q[a["jx"]], q[a["jy"]], q[a["jz"]]]), float(q[a["jyaw"]]),
                        float(q[a["jfl"]] + q[a["jfr"]]))

    def object_pose(self) -> dict:
        R = self.d.xmat[self.cube_bid].reshape(3, 3)
        return {"xy": self.d.xpos[self.cube_bid][:2].copy(),
                "z": float(self.d.xpos[self.cube_bid][2]),
                "yaw": float(np.arctan2(R[1, 0], R[0, 0])),
                "tilt_deg": float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))}

    def grip_forces(self) -> dict:
        """DIAGNOSTICS ONLY (never used by the controller or the scoring).
        Normal contact force (N) on the cube from each pad and from the table, plus the
        force each jaw actuator is applying.  Added after the first MuJoCo runs; written
        without being able to execute it here, so check the numbers for sanity."""
        mj = self.mj
        cube = int(self.m.geom("cube_geom").id)
        names = {int(self.m.geom(n).id): n for n in ("pad_l", "pad_r", "table")}
        out = {"pad_l": 0.0, "pad_r": 0.0, "table": 0.0}
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            other = g2 if g1 == cube else g1 if g2 == cube else None
            if other in names:
                mj.mj_contactForce(self.m, self.d, i, f6)
                out[names[other]] += abs(float(f6[0]))      # component 0 = normal
        out["act_l"] = abs(float(self.d.actuator_force[self.aid[4]]))
        out["act_r"] = abs(float(self.d.actuator_force[self.aid[5]]))
        return out

    # ---- dynamics --------------------------------------------------------
    def step(self, cmd: ArmCommand) -> None:
        w = float(np.clip(cmd.width, 0.0, C.MAX_OPEN)) / 2.0
        ctrl = [cmd.pos[0], cmd.pos[1], cmd.pos[2], cmd.yaw, w, w]
        self.d.ctrl[self.aid] = ctrl
        for _ in range(self.substeps):
            self.mj.mj_step(self.m, self.d)
