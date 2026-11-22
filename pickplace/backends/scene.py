"""MJCF generation for the MuJoCo backend (pure string building, no mujoco import).

Robot: a 4-DOF gantry (x, y, z slides, wrist yaw hinge) carrying a two-finger parallel
jaw.  Joint coordinates ARE the TCP coordinates because every body frame sits at the
world origin when its joint is at zero.  Gravity is compensated on the arm bodies so
position actuators do not sag.  Arm geoms are in geom group 1; the depth ray-caster
only looks at group 0 (table + cube), so the arm never occludes the camera.

Jaw: finger i slides along +/-y of the wrist frame; qpos_l + qpos_r == jaw opening
(distance between inner pad faces).  Cube friction is baked per scenario; MuJoCo uses
the max of the two geoms' sliding friction, and the pads use 1.0.
"""
from __future__ import annotations

import numpy as np

from ..camera import CameraModel
from ..scenario import Scenario
from .. import constants as C

JOINTS = ("jx", "jy", "jz", "jyaw", "jfl", "jfr")
ACTUATORS = ("ax", "ay", "az", "ayaw", "afl", "afr")


def build_mjcf(scn: Scenario, cam: CameraModel) -> str:
    h = scn.cube_size / 2.0
    cx, cy = scn.cube_xy
    quat = f"{np.cos(scn.cube_yaw / 2):.9f} 0 0 {np.sin(scn.cube_yaw / 2):.9f}"
    right, up = cam.R_wc[:, 0], -cam.R_wc[:, 1]
    xyaxes = " ".join(f"{v:.9f}" for v in (*right, *up))
    cpos = " ".join(f"{v:.9f}" for v in cam.t_wc)
    return f"""
<mujoco model="pickplace">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <default>
    <geom condim="4" friction="1 0.005 0.0001"/>
  </default>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="table" type="box" size="0.6 0.6 0.02" pos="0 0 -0.02" group="0"
          rgba="0.6 0.6 0.6 1"/>
    <camera name="cam" pos="{cpos}" xyaxes="{xyaxes}" fovy="{cam.fovy_deg:.9f}"/>

    <body name="cube" pos="{cx:.9f} {cy:.9f} {h + 0.0002:.9f}" quat="{quat}">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="{h:.9f} {h:.9f} {h:.9f}" mass="{scn.cube_mass:.9f}"
            friction="{scn.cube_friction:.9f} 0.005 0.0001" rgba="0.85 0.15 0.15 1" group="0"/>
    </body>

    <body name="gx" pos="0 0 0" gravcomp="1">
      <inertial pos="0 0 0" mass="1.5" diaginertia="0.01 0.01 0.01"/>
      <joint name="jx" type="slide" axis="1 0 0" range="-0.6 0.6" damping="190"/>
      <body name="gy" gravcomp="1">
        <inertial pos="0 0 0" mass="1.5" diaginertia="0.01 0.01 0.01"/>
        <joint name="jy" type="slide" axis="0 1 0" range="-0.6 0.6" damping="160"/>
        <body name="gz" gravcomp="1">
          <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
          <joint name="jz" type="slide" axis="0 0 1" range="-0.05 0.5" damping="115"/>
          <body name="wrist" gravcomp="1">
            <joint name="jyaw" type="hinge" axis="0 0 1" range="-2.0 2.0" damping="0.1"/>
            <geom name="palm" type="box" size="0.03 0.05 0.005" pos="0 0 0.035" mass="0.5"
                  contype="0" conaffinity="0" group="1" rgba="0.2 0.2 0.8 1"/>
            <body name="fl" gravcomp="1">
              <joint name="jfl" type="slide" axis="0 1 0" range="0 0.045" damping="8"/>
              <geom name="pad_l" type="box" size="0.012 0.004 0.012" pos="0 0.004 0" mass="0.05"
                    friction="1 0.005 0.0001" group="1" rgba="0.2 0.8 0.2 1"/>
            </body>
            <body name="fr" gravcomp="1">
              <joint name="jfr" type="slide" axis="0 -1 0" range="0 0.045" damping="8"/>
              <geom name="pad_r" type="box" size="0.012 0.004 0.012" pos="0 -0.004 0" mass="0.05"
                    friction="1 0.005 0.0001" group="1" rgba="0.2 0.8 0.2 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <contact>
    <exclude body1="fl" body2="fr"/>
  </contact>
  <actuator>
    <position name="ax" joint="jx" kp="3000" forcerange="-300 300"/>
    <position name="ay" joint="jy" kp="3000" forcerange="-300 300"/>
    <position name="az" joint="jz" kp="3000" forcerange="-300 300"/>
    <position name="ayaw" joint="jyaw" kp="3" forcerange="-2 2"/>
    <position name="afl" joint="jfl" kp="400" ctrlrange="0 0.045" forcerange="-20 20"/>
    <position name="afr" joint="jfr" kp="400" ctrlrange="0 0.045" forcerange="-20 20"/>
  </actuator>
</mujoco>
"""
