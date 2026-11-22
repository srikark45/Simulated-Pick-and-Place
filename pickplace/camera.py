"""Pinhole camera model shared by every backend and by perception.

Convention (OpenCV style): camera x = right, y = down, z = forward.
`R_wc` has those three axes as its columns, expressed in the world frame;
`t_wc` is the camera centre in the world frame.  Depth images hold z-depth
(distance along the optical axis, metres); 0.0 marks an invalid pixel.

MuJoCo's own camera looks along -z with y up, so the MJCF builder converts:
mujoco_x = right, mujoco_y = up = -down, mujoco_z = back = -forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rodrigues(w) -> np.ndarray:
    """Rotation matrix for rotation vector `w` (axis * angle, radians)."""
    w = np.asarray(w, dtype=float)
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_wc: np.ndarray  # (3,3) columns: right, down, forward in world
    t_wc: np.ndarray  # (3,) camera centre in world

    @classmethod
    def look_at(cls, pos, target, fovy_deg: float, width: int, height: int,
                up=(0.0, 0.0, 1.0)) -> "CameraModel":
        pos = np.asarray(pos, dtype=float)
        forward = _unit(np.asarray(target, dtype=float) - pos)
        right = _unit(np.cross(forward, np.asarray(up, dtype=float)))
        down = np.cross(forward, right)
        fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
        return cls(width, height, fy, fy, width / 2.0, height / 2.0,
                   np.stack([right, down, forward], axis=1), pos)

    # ---- geometry -------------------------------------------------------
    @property
    def forward(self) -> np.ndarray:
        return self.R_wc[:, 2]

    @property
    def fovy_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan2(self.height / 2.0, self.fy)))

    def _cam_rays(self) -> np.ndarray:
        """(H,W,3) rays in the camera frame with z == 1 (pixel centres)."""
        u = np.arange(self.width) + 0.5
        v = np.arange(self.height) + 0.5
        uu, vv = np.meshgrid(u, v)
        return np.stack([(uu - self.cx) / self.fx, (vv - self.cy) / self.fy,
                         np.ones_like(uu)], axis=-1)

    def unit_rays_world(self) -> np.ndarray:
        """(H,W,3) unit ray directions in the world frame."""
        r = self._cam_rays() @ self.R_wc.T
        return r / np.linalg.norm(r, axis=-1, keepdims=True)

    def range_to_zdepth(self, dist: np.ndarray) -> np.ndarray:
        """Convert distance along each pixel ray into z-depth (0 stays 0)."""
        return dist * (self.unit_rays_world() @ self.forward)

    def backproject(self, depth: np.ndarray) -> np.ndarray:
        """(H,W) z-depth -> (H,W,3) world points."""
        pts_c = self._cam_rays() * depth[..., None]
        return pts_c @ self.R_wc.T + self.t_wc

    def project(self, pts_w) -> tuple[np.ndarray, np.ndarray]:
        """(N,3) world points -> ((N,2) pixel coords, (N,) z-depth)."""
        pc = (np.atleast_2d(np.asarray(pts_w, dtype=float)) - self.t_wc) @ self.R_wc
        z = pc[:, 2]
        uv = np.stack([self.fx * pc[:, 0] / z + self.cx,
                       self.fy * pc[:, 1] / z + self.cy], axis=-1)
        return uv, z

    def perturbed(self, rng: np.random.Generator, pos_sigma: float,
                  rot_sigma_rad: float) -> "CameraModel":
        """A copy with a small random extrinsic error (calibration error)."""
        dR = rodrigues(rng.normal(0.0, rot_sigma_rad, 3))
        dt = rng.normal(0.0, pos_sigma, 3)
        return CameraModel(self.width, self.height, self.fx, self.fy, self.cx,
                           self.cy, dR @ self.R_wc, self.t_wc + dt)
