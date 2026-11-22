"""Trace what happens to the held cube in chosen seeds.

    python scripts/diagnose_seed.py --backend mujoco --seeds 46,38,106,166,395 --noise-scale 1
    python scripts/diagnose_seed.py --backend mujoco --seeds 5,6,7 --noise-scale 1     # light cubes, for comparison

For each seed it prints, after every controller phase:
  dz     cube-centre height minus TCP height (mm).  A held cube that slides in the jaws
         shows up as dz drifting between 'grasp' / 'lift' / 'transit' / 'place_descend'.
  width  measured jaw opening (mm)
  tcp_err  measured TCP minus the phase target, x / y / z (mm), and yaw error (deg)
and, if a phase times out, the last samples of the settle loop so you can see WHICH
tolerance (z, xy or yaw) kept it from converging.

Ground truth (`object_pose`) is used for printing only; the controller never sees it.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pickplace import episode as E  # noqa: E402
from pickplace.backends import make_backend  # noqa: E402
from pickplace.controller import (ControllerConfig, ControllerFailure,  # noqa: E402
                                  PickPlaceController)
from pickplace.scenario import sample_scenario  # noqa: E402


class Traced(PickPlaceController):
    rows: list = []
    zlog: list = []

    def _step(self, cmd):
        super()._step(cmd)
        if getattr(self, "_phase", None) == "place_descend":
            st, gt = self.b.state(), self.b.object_pose()
            Traced.zlog.append((self.t, cmd.pos[2], st.pos[2], gt["z"], st.yaw - cmd.yaw))

    def _snap(self, label, pos, yaw):
        st, gt = self.b.state(), self.b.object_pose()
        err = (np.array(st.pos) - np.asarray(pos, float)) * 1e3
        try:                                  # MuJoCo backend only; never breaks the trace
            g = self.b.grip_forces()
            force = (f"pad {g['pad_l']:5.2f}/{g['pad_r']:5.2f} N  act {g['act_l']:5.2f}/"
                     f"{g['act_r']:5.2f} N  table {g['table']:5.2f} N  tilt {gt['tilt_deg']:5.1f}")
        except Exception as e:                # mock backend, or an API mismatch
            force = f"(grip forces unavailable: {type(e).__name__})"
        Traced.rows.append((label, (gt["z"] - st.pos[2]) * 1e3, st.width * 1e3, err,
                            np.degrees(st.yaw - yaw), force))

    def _move(self, name, pos, yaw, width, speed, *args, **kwargs):
        # *args/**kwargs are forwarded untouched so this tracer keeps working if the
        # controller's _move gains extra parameters (e.g. a locally modified controller).
        self._phase = name
        try:
            super()._move(name, pos, yaw, width, speed, *args, **kwargs)
        except ControllerFailure:
            self._snap(name + "  <-- TIMEOUT", pos, yaw)
            raise
        stalled = bool(self.phases and self.phases[-1].get("stalled"))
        self._snap(name + ("  (accepted: TCP stalled)" if stalled else ""), pos, yaw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mujoco", "mock"], required=True)
    ap.add_argument("--seeds", required=True, help="comma-separated, e.g. 46,38,106")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--guarded-place", action="store_true",
                    help="trace the experimental guarded-place controller variant")
    args = ap.parse_args(argv)
    ctrl_cfg = ControllerConfig(guarded_place=args.guarded_place)

    E.PickPlaceController = Traced        # run_episode resolves this name at call time
    backend = make_backend(args.backend)
    for s in [int(x) for x in args.seeds.split(",")]:
        Traced.rows, Traced.zlog = [], []
        r = E.run_episode(backend, sample_scenario(s, args.noise_scale), ctrl_cfg=ctrl_cfg)
        sc = r.scenario
        print(f"\n=== seed {s}  {'OK' if r.success else 'FAIL'}  {r.failure or ''}")
        print(f"    mass {sc['cube_mass']*1e3:.1f} g  mu {sc['cube_friction']:.2f}  size "
              f"{sc['cube_size']*1e3:.1f} mm  perc_xy_err {r.perc_xy_err_mm:.1f} mm"
              if r.perc_xy_err_mm is not None else "    (perception failed)")
        print(f"    {'phase':26s} {'dz(mm)':>8s} {'width':>7s}   tcp_err x / y / z (mm)    yaw_err(deg)"
              "   forces on cube (normal, N) / cube tilt (deg)")
        for label, dz, w, e, yaw, force in Traced.rows:
            print(f"    {label:26s} {dz:8.2f} {w:7.2f}   {e[0]:6.2f} {e[1]:6.2f} {e[2]:6.2f}   {yaw:8.2f}   {force}")
        if not r.success and Traced.zlog:
            z = Traced.zlog
            print("    place_descend settle samples (t, cmd_z, tcp_z, cube_z [mm], yaw_err deg):")
            for t, cz, tz, bz, ye in z[:: max(1, len(z) // 8)][-9:]:
                print(f"      t={t:5.2f}  cmd {cz*1e3:6.2f}  tcp {tz*1e3:6.2f}  cube {bz*1e3:6.2f}  yaw_err {np.degrees(ye):6.2f}")
        print(f"    final: z_err {r.final_z_err_mm:.2f} mm  tilt {r.final_tilt_deg:.2f} deg  xy_err {r.final_xy_err_mm:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
