"""Seeded scenario sampling.  Everything random about an episode derives from `seed`.

Sub-streams are keyed by (seed, k) so adding a draw to one stream never shifts another:
  k=0  scenario (object pose, size, mass, friction, place target)
  k=1  perception corruption (depth noise, dropout, camera-pose error)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import constants as C

# Nominal noise levels; `noise_scale` multiplies all four.
DEPTH_SIGMA = 0.0005        # 0.5 mm gaussian depth noise
DEPTH_DROPOUT = 0.005       # 0.5 % of pixels invalid
CAM_POS_SIGMA = 0.003       # 3 mm camera-position calibration error (per axis)
CAM_ROT_SIGMA_DEG = 0.3     # 0.3 deg camera-orientation calibration error (per axis)


@dataclass(frozen=True)
class Scenario:
    seed: int
    cube_xy: tuple
    cube_yaw: float
    cube_size: float          # full edge length, metres
    cube_mass: float
    cube_friction: float
    place_xy: tuple
    depth_sigma: float
    depth_dropout: float
    cam_pos_sigma: float
    cam_rot_sigma: float      # radians

    def to_dict(self) -> dict:
        return asdict(self)


def sample_scenario(seed: int, noise_scale: float = 1.0) -> Scenario:
    rng = np.random.default_rng([int(seed), 0])
    cube_xy = (float(rng.uniform(*C.SPAWN_X)), float(rng.uniform(*C.SPAWN_Y)))
    cube_yaw = float(rng.uniform(-np.pi, np.pi))
    cube_size = float(rng.uniform(0.035, 0.045))
    cube_mass = float(rng.uniform(0.03, 0.08))
    cube_friction = float(rng.uniform(0.6, 1.2))
    place_xy = (float(rng.uniform(*C.PLACE_X)), float(rng.uniform(*C.PLACE_Y)))
    s = float(noise_scale)
    return Scenario(int(seed), cube_xy, cube_yaw, cube_size, cube_mass, cube_friction,
                    place_xy, DEPTH_SIGMA * s, DEPTH_DROPOUT * s, CAM_POS_SIGMA * s,
                    float(np.radians(CAM_ROT_SIGMA_DEG)) * s)


def perception_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), 1])
