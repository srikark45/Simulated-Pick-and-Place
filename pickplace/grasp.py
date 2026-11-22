"""Top-down parallel-jaw grasp planner for a cube.

Candidates: the jaw closing axis can align with either pair of cube faces and the
wrist can approach from either side, i.e. yaw = est.yaw + k*pi/2, k = 0..3.  A
candidate is feasible if it is inside the wrist range; the cheapest rotation from the
current wrist yaw wins (ties -> smaller |yaw|).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C
from .perception import ObjectEstimate


class GraspError(Exception):
    """Raised with a short reason: 'too_wide' or 'no_reachable_yaw'."""


@dataclass(frozen=True)
class GraspPlan:
    xyz: tuple           # TCP position at grasp
    yaw: float           # wrist yaw at grasp
    open_width: float    # jaw opening used for the approach
    hover_z: float       # TCP height for approach / transit


def wrap_pi(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class GraspPlanner:
    def __init__(self, max_open: float = C.MAX_OPEN, yaw_limit: float = C.YAW_LIMIT,
                 side_margin: float = 0.015, min_margin: float = 0.006,
                 min_tcp_z: float = C.MIN_TCP_Z, hover_z: float = C.HOVER_Z):
        self.max_open, self.yaw_limit = max_open, yaw_limit
        self.side_margin, self.min_margin = side_margin, min_margin
        self.min_tcp_z, self.hover_z = min_tcp_z, hover_z

    def plan(self, est: ObjectEstimate, current_yaw: float = 0.0) -> GraspPlan:
        if est.size + 2 * self.min_margin > self.max_open:
            raise GraspError("too_wide")
        open_width = min(self.max_open, est.size + 2 * self.side_margin)
        best = None
        for k in range(4):
            yaw = wrap_pi(est.yaw + k * np.pi / 2)
            if abs(yaw) > self.yaw_limit - 0.05:
                continue
            cost = (abs(yaw - current_yaw), abs(yaw))
            if best is None or cost < best[0]:
                best = (cost, yaw)
        if best is None:
            raise GraspError("no_reachable_yaw")
        z = max(self.min_tcp_z, est.top_z / 2.0)     # centre of the cube's side faces
        return GraspPlan((float(est.xy[0]), float(est.xy[1]), float(z)), best[1],
                         float(open_width), self.hover_z)
