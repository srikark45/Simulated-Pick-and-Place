"""One seeded episode: reset -> sense -> estimate -> plan -> execute -> score.

Success needs the controller to report completion AND the final cube pose, taken from
simulator ground truth (`backend.object_pose()`, never the pipeline's estimates), to be
within tolerance.  Ground truth is also used to log perception errors, but the pipeline
itself never sees it.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from . import constants as C
from .controller import ControllerConfig, PickPlaceController
from .grasp import GraspError, GraspPlanner
from .perception import PerceptionError, corrupt_depth, estimate_object, wrap_quarter
from .scenario import Scenario, perception_rng


@dataclass
class EpisodeResult:
    seed: int
    backend: str
    success: bool
    failure: str | None          # None on success; else 'stage:reason'
    perc_xy_err_mm: float | None
    perc_yaw_err_deg: float | None
    perc_size_err_mm: float | None
    final_xy_err_mm: float | None
    final_tilt_deg: float | None
    final_z_err_mm: float | None
    sim_time_s: float
    wall_time_s: float
    scenario: dict

    def to_dict(self) -> dict:
        return asdict(self)


def run_episode(backend, scn: Scenario, planner: GraspPlanner | None = None,
                ctrl_cfg: ControllerConfig | None = None) -> EpisodeResult:
    t0 = time.perf_counter()
    planner = planner or GraspPlanner()
    backend.reset(scn)
    gt0 = backend.object_pose()          # logging only
    sim_t = 0.0

    def result(failure, est=None, ctrl=None):
        gt = backend.object_pose()
        xy_err = float(np.linalg.norm(gt["xy"] - np.array(scn.place_xy)))
        z_err = abs(gt["z"] - scn.cube_size / 2.0)
        ok = (failure is None and xy_err <= C.PLACE_TOL
              and gt["tilt_deg"] <= C.TILT_TOL_DEG and z_err <= C.REST_Z_TOL)
        if failure is None and not ok:
            failure = "placement:unstable" if (gt["tilt_deg"] > C.TILT_TOL_DEG
                                               or z_err > C.REST_Z_TOL) else "placement:off_target"
        p = {}
        if est is not None:
            p = {"perc_xy_err_mm": 1e3 * float(np.linalg.norm(est.xy - gt0["xy"])),
                 "perc_yaw_err_deg": float(np.degrees(abs(wrap_quarter(est.yaw - gt0["yaw"])))),
                 "perc_size_err_mm": 1e3 * abs(est.size - scn.cube_size)}
        return EpisodeResult(
            seed=scn.seed, backend=backend.name, success=ok, failure=failure,
            perc_xy_err_mm=p.get("perc_xy_err_mm"), perc_yaw_err_deg=p.get("perc_yaw_err_deg"),
            perc_size_err_mm=p.get("perc_size_err_mm"), final_xy_err_mm=1e3 * xy_err,
            final_tilt_deg=gt["tilt_deg"], final_z_err_mm=1e3 * z_err,
            sim_time_s=ctrl.sim_time if ctrl else sim_t,
            wall_time_s=time.perf_counter() - t0, scenario=scn.to_dict())

    # --- sense: depth image + (imperfect) knowledge of the camera pose
    rng = perception_rng(scn.seed)
    depth = corrupt_depth(backend.render_depth(), rng, scn.depth_sigma, scn.depth_dropout)
    cam_est = backend.camera.perturbed(rng, scn.cam_pos_sigma, scn.cam_rot_sigma)

    try:
        est = estimate_object(depth, cam_est, table_z=C.TABLE_Z)
    except PerceptionError as e:
        return result(f"perception:{e}")
    try:
        plan = planner.plan(est, backend.state().yaw)
    except GraspError as e:
        return result(f"planning:{e}", est)

    ctrl = PickPlaceController(backend, ctrl_cfg).run(plan, scn.place_xy, est.size)
    if not ctrl.ok:
        return result(f"control:{ctrl.failure}", est, ctrl)
    return result(None, est, ctrl)
