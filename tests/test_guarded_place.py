"""Tests for the opt-in guarded-place controller mode (no MuJoCo needed).

A stub backend reproduces the situation seen in MuJoCo: a held cube meets the table while
the TCP is still `stall_offset` above its place target, so the TCP stops short.  The stub
is NOT a model of contact physics; it only exercises the controller's decision logic.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

from pickplace.backends.base import ArmState
from pickplace.controller import ControllerConfig, PickPlaceController
from pickplace.eval import main as eval_main
from pickplace.grasp import GraspPlan

SIZE = 0.04
PLACE_XY = (0.1, 0.1)


class StubBackend:
    ctrl_dt = 0.02

    def __init__(self, stall_offset=0.0, drop_on_contact=False):
        self.size, self.held_w = SIZE, SIZE - 0.0015
        self.offset, self.drop = stall_offset, drop_on_contact
        self.pos = np.array([0.0, 0.0, 0.15])
        self.yaw, self.width = 0.0, 0.07
        self.holding, self.dropped = False, False

    def state(self):
        return ArmState(self.pos.copy(), self.yaw, self.width)

    def step(self, cmd):
        a = 0.25
        self.pos = self.pos + (np.asarray(cmd.pos, float) - self.pos) * a
        self.yaw += (cmd.yaw - self.yaw) * a
        w = self.width + float(np.clip(cmd.width - self.width, -0.002, 0.002))
        if not self.dropped:
            if self.holding:
                if cmd.width > self.held_w + 0.004:
                    self.holding = False            # released
                else:
                    w = self.held_w
            elif w <= self.held_w and cmd.width < self.held_w:
                self.holding, w = True, self.held_w  # closed on the cube
        self.width = max(w, 0.0)
        floor = SIZE / 2 + 0.004 + self.offset
        if self.holding and abs(self.pos[0]) > 0.05 and self.pos[2] < floor:
            self.pos[2] = floor                      # cube is on the table: TCP can't go lower
            if self.drop:
                self.dropped, self.holding, self.width = True, False, 0.0


def run(cfg, **stub_kw):
    plan = GraspPlan(xyz=(0.0, 0.0, 0.02), yaw=0.0, open_width=0.07, hover_z=0.15)
    return PickPlaceController(StubBackend(**stub_kw), cfg).run(plan, PLACE_XY, SIZE)


class GuardedPlaceTests(unittest.TestCase):
    def test_default_times_out_when_tcp_stops_3p5mm_short(self):
        r = run(ControllerConfig(), stall_offset=0.0035)
        self.assertFalse(r.ok)
        self.assertEqual(r.failure, "tracking_timeout:place_descend")

    def test_guarded_accepts_a_stalled_tcp_and_records_it(self):
        r = run(ControllerConfig(guarded_place=True), stall_offset=0.0035)
        self.assertTrue(r.ok, r.failure)
        self.assertTrue(any(p.get("stalled") for p in r.phases if p["phase"] == "place_descend"))

    def test_guarded_does_not_flag_a_normal_descent(self):
        r = run(ControllerConfig(guarded_place=True), stall_offset=0.0)
        self.assertTrue(r.ok, r.failure)
        self.assertFalse(any(p.get("stalled") for p in r.phases))

    def test_guarded_still_times_out_when_stall_is_too_high(self):
        r = run(ControllerConfig(guarded_place=True), stall_offset=0.020)   # > stall_max 12 mm
        self.assertFalse(r.ok)
        self.assertEqual(r.failure, "tracking_timeout:place_descend")

    def test_guarded_reports_a_cube_that_left_the_jaws(self):
        r = run(ControllerConfig(guarded_place=True), stall_offset=0.0035, drop_on_contact=True)
        self.assertFalse(r.ok)
        self.assertEqual(r.failure, "slipped:place_descend")

    def test_stall_must_last_stall_ticks(self):
        cfg = ControllerConfig(guarded_place=True, stall_ticks=10_000)   # never satisfied
        r = run(cfg, stall_offset=0.0035)
        self.assertEqual(r.failure, "tracking_timeout:place_descend")


class EvalFlagTests(unittest.TestCase):
    def _summary(self, *extra):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            rc = eval_main(["--backend", "mock", "--seeds", "0:3", "--noise-scale", "0",
                            "--out", d, *extra])
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "summary.json")) as f:
                return json.load(f)

    def test_default_is_unguarded_and_recorded(self):
        self.assertEqual(self._summary()["controller"], {"guarded_place": False})

    def test_flag_enables_guarded_and_is_recorded(self):
        self.assertEqual(self._summary("--guarded-place")["controller"], {"guarded_place": True})


if __name__ == "__main__":
    unittest.main()
