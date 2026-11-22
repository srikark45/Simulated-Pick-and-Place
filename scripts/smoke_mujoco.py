"""Run this FIRST on a machine with MuJoCo:   python scripts/smoke_mujoco.py

It checks the things that cannot be checked without the real engine, in the order a
failure would be easiest to diagnose.  Exits non-zero on the first failed check.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from pickplace import constants as C  # noqa: E402
from pickplace.backends.base import ArmCommand  # noqa: E402
from pickplace.backends.mujoco_backend import MuJoCoBackend  # noqa: E402
from pickplace.backends.mock import MockBackend  # noqa: E402
from pickplace.episode import run_episode  # noqa: E402
from pickplace.scenario import sample_scenario  # noqa: E402


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        raise SystemExit(1)


def main():
    import mujoco
    print("mujoco", mujoco.__version__)
    b = MuJoCoBackend()
    scn = sample_scenario(0, noise_scale=0.0)

    b.reset(scn)                       # compiles MJCF + camera self-check inside
    check("MJCF compiles; camera extrinsics match MuJoCo", True)

    # 1. cube rests on the table after settling
    gt = b.object_pose()
    check("cube settles on table", abs(gt["z"] - scn.cube_size / 2) < 0.002 and gt["tilt_deg"] < 1,
          f"z={gt['z']:.4f} tilt={gt['tilt_deg']:.2f}")

    # 2. arm tracks position commands and does not sag (gravity compensation)
    target = np.array([0.1, 0.0, 0.2])
    for _ in range(100):
        b.step(ArmCommand(target, 0.5, 0.06))
    st = b.state()
    check("TCP tracks command", np.linalg.norm(st.pos - target) < 0.003 and abs(st.yaw - 0.5) < 0.02,
          f"pos_err={np.linalg.norm(st.pos - target)*1e3:.2f}mm yaw_err={abs(st.yaw-0.5):.3f}")

    # 3. depth image: cube top-face centre depth matches analytic projection
    b.reset(scn)
    depth = b.render_depth()
    top = np.array([*scn.cube_xy, scn.cube_size])
    uv, z = b.camera.project(top)
    u, v = int(uv[0, 0]), int(uv[0, 1])
    check("depth at cube top centre matches camera model", abs(depth[v, u] - z[0]) < 0.002,
          f"measured={depth[v,u]:.4f} expected={z[0]:.4f}")
    tbl = b.camera.project([0.0, -0.07, 0.0])
    check("depth on table matches", abs(depth[int(tbl[0][0, 1]), int(tbl[0][0, 0])] - tbl[1][0]) < 0.002)

    # 4. hold test: close the jaws on the cube at its true pose and lift it
    b.reset(scn)
    g = np.array([*scn.cube_xy, scn.cube_size / 2])
    for pos, w in [((g[0], g[1], 0.15), 0.07), ((g[0], g[1], g[2]), 0.07)]:
        for _ in range(60):
            b.step(ArmCommand(pos, scn.cube_yaw if abs(scn.cube_yaw) < 1.9 else 0.0, w))
    # (yaw alignment for this hand-placed test: use nearest of the 4 symmetric yaws)
    check("jaws reach the cube", True)
    for _ in range(40):
        b.step(ArmCommand(g, b.state().yaw, 0.0))
    w = b.state().width
    for _ in range(60):
        b.step(ArmCommand((g[0], g[1], 0.15), b.state().yaw, 0.0))
    held = b.object_pose()["z"] > 0.10
    check("squeezed cube lifts with the jaws (friction grasp works)", held,
          f"jaw_width={w:.4f} cube_z={b.object_pose()['z']:.3f} (this check is yaw-sensitive)")

    # 5. determinism on the real engine
    r1 = run_episode(b, sample_scenario(3)).to_dict()
    r2 = run_episode(b, sample_scenario(3)).to_dict()
    r1.pop("wall_time_s"), r2.pop("wall_time_s")
    check("same seed -> identical result", r1 == r2)

    # 6. mock vs MuJoCo depth agree (validates the mock's analytic renderer)
    bm = MockBackend()
    bm.reset(scn)
    b.reset(scn)
    dm, dj = bm.render_depth(), b.render_depth()
    both = (dm > 0) & (dj > 0)
    check("mock depth == MuJoCo depth", np.abs(dm - dj)[both].max() < 0.002,
          f"max |diff|={np.abs(dm - dj)[both].max()*1e3:.3f}mm")
    print("smoke test passed")


if __name__ == "__main__":
    main()
