"""Closed-loop pick-and-place state machine.

Talks to a backend only through `state()` and `step(ArmCommand)`.  Each move is a
min-jerk trajectory followed by a settle loop that waits for the *measured* TCP to
converge (so a sluggish or blocked arm is detected rather than assumed).  The grasp is
verified from the measured jaw width, and checked again after lift and after transit.
The controller never reads ground-truth object state.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as C
from .backends.base import ArmCommand
from .grasp import GraspPlan


def min_jerk(tau: float) -> float:
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


@dataclass
class ControllerConfig:
    v_max: float = 0.35
    v_slow: float = 0.12          # descents / final approach
    yaw_rate: float = 2.0
    min_seg: float = 0.4
    pos_tol: float = 0.003
    yaw_tol: float = 0.03
    settle_timeout: float = 1.5
    grip_hold: float = 0.4        # extra hold after closing before measuring
    width_tol: float = 0.012      # |measured width - estimated size| to accept a grasp
    slip_tol: float = 0.004
    place_clearance: float = 0.004
    # Guarded placement (opt-in; default False keeps the original behaviour exactly).
    # A held cube can sit lower in the jaws than assumed (observed in MuJoCo for ~77-79 g
    # cubes), so it meets the table before the TCP reaches the place height.  Guarded mode
    # accepts a TCP that has stopped moving up to `stall_max` above the target, then re-checks
    # the jaw width so a cube that slid out is still reported as a slip.
    guarded_place: bool = False
    stall_max: float = 0.012      # max height of a stalled TCP above the target (m)
    stall_speed: float = 0.003    # TCP vertical speed below which it counts as stopped (m/s)
    stall_ticks: int = 10         # consecutive control ticks that must be stopped


class ControllerFailure(Exception):
    pass


@dataclass
class ControllerResult:
    ok: bool
    failure: str | None
    phases: list = field(default_factory=list)
    sim_time: float = 0.0


class PickPlaceController:
    def __init__(self, backend, cfg: ControllerConfig | None = None):
        self.b = backend
        self.cfg = cfg or ControllerConfig()
        self.dt = backend.ctrl_dt
        self.t = 0.0
        self.phases: list[dict] = []

    # ---- primitives -----------------------------------------------------
    def _step(self, cmd: ArmCommand) -> None:
        self.b.step(cmd)
        self.t += self.dt

    def _move(self, name: str, pos, yaw: float, width: float, speed: float,
              check_pos: bool = True, stall_z: float | None = None) -> None:
        cfg = self.cfg
        s0 = self.b.state()
        p0, y0, w0 = np.array(s0.pos, float), s0.yaw, s0.width
        pos = np.asarray(pos, float)
        T = max(cfg.min_seg, np.linalg.norm(pos - p0) / speed,
                abs(yaw - y0) / cfg.yaw_rate)
        n = int(np.ceil(T / self.dt))
        for k in range(1, n + 1):
            s = min_jerk(k / n)
            self._step(ArmCommand(p0 + (pos - p0) * s, y0 + (yaw - y0) * s,
                                  w0 + (width - w0) * s))
        final = ArmCommand(pos, yaw, width)
        waited = 0.0
        stalled, prev_z = 0, None
        while True:
            st = self.b.state()
            p = np.array(st.pos)
            yaw_ok = abs(st.yaw - yaw) < cfg.yaw_tol
            if np.linalg.norm(p - pos) < cfg.pos_tol and yaw_ok:
                break
            if stall_z is not None:
                above = p[2] - pos[2]
                still = prev_z is not None and abs(p[2] - prev_z) / self.dt < cfg.stall_speed
                prev_z = p[2]
                lateral_ok = np.linalg.norm(p[:2] - pos[:2]) < cfg.pos_tol
                stalled = stalled + 1 if (lateral_ok and yaw_ok and 0.0 <= above <= stall_z
                                          and still) else 0
                if stalled >= cfg.stall_ticks:
                    self.phases.append({"phase": name, "ok": True, "stalled": True, "t": self.t})
                    return
            if waited >= cfg.settle_timeout:
                self.phases.append({"phase": name, "ok": False, "t": self.t})
                raise ControllerFailure(f"tracking_timeout:{name}")
            self._step(final)
            waited += self.dt
        self.phases.append({"phase": name, "ok": True, "t": self.t})

    def _hold(self, cmd: ArmCommand, seconds: float) -> None:
        for _ in range(int(np.ceil(seconds / self.dt))):
            self._step(cmd)

    def _check_holding(self, name: str, ref_width: float) -> None:
        w = self.b.state().width
        if abs(w - ref_width) > self.cfg.slip_tol:
            raise ControllerFailure(f"slipped:{name}")

    # ---- full sequence ---------------------------------------------------
    def run(self, plan: GraspPlan, place_xy, size: float) -> ControllerResult:
        cfg = self.cfg
        gx, gy, gz = plan.xyz
        hover = plan.hover_z
        try:
            self._move("approach", (gx, gy, hover), plan.yaw, plan.open_width, cfg.v_max)
            self._move("descend", (gx, gy, gz), plan.yaw, plan.open_width, cfg.v_slow)
            self._move("close", (gx, gy, gz), plan.yaw, 0.0, cfg.v_slow)
            self._hold(ArmCommand((gx, gy, gz), plan.yaw, 0.0), cfg.grip_hold)
            w = self.b.state().width
            if abs(w - size) > cfg.width_tol or w < 0.5 * size:
                self.phases.append({"phase": "grasp_check", "ok": False, "width": w})
                raise ControllerFailure("grasp_miss")
            self.phases.append({"phase": "grasp_check", "ok": True, "width": w})
            self._move("lift", (gx, gy, hover), plan.yaw, 0.0, cfg.v_slow)
            self._check_holding("lift", w)
            px, py = place_xy
            self._move("transit", (px, py, hover), plan.yaw, 0.0, cfg.v_max)
            self._check_holding("transit", w)
            place_z = size / 2.0 + cfg.place_clearance
            self._move("place_descend", (px, py, place_z), plan.yaw, 0.0, cfg.v_slow,
                       stall_z=cfg.stall_max if cfg.guarded_place else None)
            if cfg.guarded_place:
                self._check_holding("place_descend", w)
            self._move("release", (px, py, place_z), plan.yaw, min(C.MAX_OPEN, size + 0.03),
                       cfg.v_slow)
            self._move("retreat", (px, py, hover), plan.yaw, min(C.MAX_OPEN, size + 0.03),
                       cfg.v_max)
        except ControllerFailure as e:
            return ControllerResult(False, str(e), self.phases, self.t)
        return ControllerResult(True, None, self.phases, self.t)
