"""Unit tests that run WITHOUT MuJoCo.  `python -m unittest discover -s tests -v`"""
import json
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from pickplace import constants as C
from pickplace.backends.mock import MockBackend, box_ray_range
from pickplace.backends.scene import ACTUATORS, JOINTS, build_mjcf
from pickplace.camera import CameraModel
from pickplace.controller import min_jerk
from pickplace.episode import run_episode
from pickplace.grasp import GraspError, GraspPlanner
from pickplace.perception import ObjectEstimate, PerceptionError, estimate_object, wrap_quarter
from pickplace.scenario import sample_scenario
from pickplace.stats import wilson_interval


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.cam = C.make_camera()

    def test_axes_orthonormal_right_handed(self):
        R = self.cam.R_wc
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_looks_at_target(self):
        uv, z = self.cam.project(np.array(C.CAM_TARGET))
        np.testing.assert_allclose(uv[0], [self.cam.cx, self.cam.cy], atol=1e-9)
        self.assertGreater(z[0], 0)

    def test_project_backproject_roundtrip(self):
        rng = np.random.default_rng(0)
        pts = np.c_[rng.uniform(-.15, .15, 50), rng.uniform(-.18, .04, 50), rng.uniform(0, .05, 50)]
        uv, z = self.cam.project(pts)
        # rebuild a sparse depth image and back-project pixel centres
        for p, (u, v), zz in zip(pts, uv, z):
            ray = np.array([(u - self.cam.cx) / self.cam.fx, (v - self.cam.cy) / self.cam.fy, 1.0])
            back = self.cam.R_wc @ (ray * zz) + self.cam.t_wc
            np.testing.assert_allclose(back, p, atol=1e-9)

    def test_up_in_image_is_up_in_world(self):
        # a point higher in z must appear higher in the image (smaller v)
        a, _ = self.cam.project([0, -0.07, 0.0])
        b, _ = self.cam.project([0, -0.07, 0.05])
        self.assertLess(b[0, 1], a[0, 1])

    def test_spawn_region_inside_image(self):
        for x in C.SPAWN_X:
            for y in C.SPAWN_Y:
                for z in (0.0, 0.045):
                    uv, _ = self.cam.project([x, y, z])
                    self.assertTrue(12 < uv[0, 0] < C.IMG_W - 12 and 12 < uv[0, 1] < C.IMG_H - 12,
                                    f"corner {(x, y, z)} -> {uv[0]}")

    def test_pixel_size_gives_enough_pixels_on_cube(self):
        uv0, _ = self.cam.project([0, -0.07, 0.04])
        uv1, _ = self.cam.project([0.04, -0.07, 0.04])
        self.assertGreater(abs(uv1[0, 0] - uv0[0, 0]), 8)    # a 4 cm edge spans > 8 px

    def test_ray_box_range_known_case(self):
        o = np.array([0.0, 0.0, 1.0])
        d = np.array([0.0, 0.0, -1.0])
        t = box_ray_range(o, d, np.array([0, 0, 0.02]), 0.7, 0.02)
        self.assertAlmostEqual(float(t), 0.96, places=9)      # top face at z = 0.04
        miss = box_ray_range(o + [0.1, 0, 0], d, np.array([0, 0, 0.02]), 0.0, 0.02)
        self.assertTrue(np.isinf(miss))


class PerceptionTests(unittest.TestCase):
    def _est(self, scn, noise=False):
        b = MockBackend()
        b.reset(scn)
        depth = b.render_depth()
        if noise:
            from pickplace.perception import corrupt_depth
            from pickplace.scenario import perception_rng
            rng = perception_rng(scn.seed)
            depth = corrupt_depth(depth, rng, scn.depth_sigma, scn.depth_dropout)
        return estimate_object(depth, b.camera)

    def test_noiseless_estimate_is_accurate_over_many_poses(self):
        xy_err, yaw_err, size_err = [], [], []
        for seed in range(60):
            scn = sample_scenario(seed)
            e = self._est(scn)
            xy_err.append(np.linalg.norm(e.xy - scn.cube_xy))
            yaw_err.append(abs(wrap_quarter(e.yaw - scn.cube_yaw)))
            size_err.append(abs(e.size - scn.cube_size))
        self.assertLess(max(xy_err), 0.004)                    # < 4 mm
        # Yaw is limited by ~4 mm pixels: a nearly axis-aligned cube can snap to 0 deg, so
        # the worst case is ~5 deg (seed 45, diagnosed).  Assert the distribution instead.
        yd = np.degrees(yaw_err)
        self.assertLess(np.mean(yd), 1.0)
        self.assertLess(np.percentile(yd, 90), 2.0)
        self.assertLess(yd.max(), 6.0)
        self.assertLess(max(size_err), 0.002)

    def test_wrap_quarter_range_and_symmetry(self):
        for a in np.linspace(-7, 7, 57):
            w = wrap_quarter(a)
            self.assertTrue(-np.pi / 4 <= w < np.pi / 4)
            self.assertAlmostEqual(np.cos(4 * w), np.cos(4 * a), places=9)

    def test_empty_scene_raises(self):
        with self.assertRaises(PerceptionError):
            estimate_object(np.zeros((C.IMG_H, C.IMG_W)), C.make_camera())

    def test_flat_table_only_raises(self):
        b = MockBackend()
        scn = sample_scenario(0)
        b.reset(scn)
        b.obj_xy = np.array([5.0, 5.0])                       # cube far outside the view
        with self.assertRaises(PerceptionError) as cm:
            estimate_object(b.render_depth(), b.camera)
        self.assertEqual(str(cm.exception), "no_object")      # valid table, nothing on it

    def test_regression_camera_height_tilt_error_does_not_segment_the_table(self):
        # Before the table-plane fit, these seeds' calibration errors made the bare table
        # read 9-15 mm above z=0 and the table became the "object" (clipped by border).
        from pickplace.perception import corrupt_depth
        from pickplace.scenario import perception_rng
        for seed in (19, 31, 61, 72):
            scn = sample_scenario(seed)
            b = MockBackend()
            b.reset(scn)
            rng = perception_rng(seed)
            depth = corrupt_depth(b.render_depth(), rng, scn.depth_sigma, scn.depth_dropout)
            cam = b.camera.perturbed(rng, scn.cam_pos_sigma, scn.cam_rot_sigma)
            e = estimate_object(depth, cam)          # must not raise
            self.assertLess(np.linalg.norm(e.xy - scn.cube_xy), 0.015)
            self.assertLess(abs(e.size - scn.cube_size), 0.003)   # height error cancelled

    def test_pure_height_offset_is_cancelled_exactly(self):
        scn = sample_scenario(4)
        b = MockBackend()
        b.reset(scn)
        lifted = CameraModel(b.camera.width, b.camera.height, b.camera.fx, b.camera.fy,
                             b.camera.cx, b.camera.cy, b.camera.R_wc, b.camera.t_wc + [0, 0, 0.02])
        e = estimate_object(b.render_depth(), lifted)
        self.assertAlmostEqual(e.size, scn.cube_size, delta=0.002)

    def test_absurd_camera_pose_is_rejected(self):
        scn = sample_scenario(4)
        b = MockBackend()
        b.reset(scn)
        bad = CameraModel(b.camera.width, b.camera.height, b.camera.fx, b.camera.fy,
                          b.camera.cx, b.camera.cy, b.camera.R_wc, b.camera.t_wc + [0, 0, 0.0])
        # table plane 10 cm off: simulate by claiming the table is elsewhere
        with self.assertRaises(PerceptionError):
            estimate_object(b.render_depth(), bad, table_z=0.10)

    def test_wrong_camera_pose_shifts_estimate_as_expected(self):
        scn = sample_scenario(3)
        b = MockBackend()
        b.reset(scn)
        shifted = CameraModel(b.camera.width, b.camera.height, b.camera.fx, b.camera.fy,
                              b.camera.cx, b.camera.cy, b.camera.R_wc, b.camera.t_wc + [0.01, 0, 0])
        e = estimate_object(b.render_depth(), shifted)
        self.assertAlmostEqual(e.xy[0] - scn.cube_xy[0], 0.01, delta=0.004)


class PlannerTests(unittest.TestCase):
    def est(self, yaw, size=0.04):
        return ObjectEstimate(np.array([0.05, -0.05]), size, size, yaw, size, 100)

    def test_grasp_height_is_cube_mid_height(self):
        g = GraspPlanner().plan(self.est(0.1))
        self.assertAlmostEqual(g.xyz[2], 0.02)

    def test_picks_smallest_rotation(self):
        g = GraspPlanner().plan(self.est(0.5), current_yaw=0.0)
        self.assertAlmostEqual(g.yaw, 0.5)
        # est yaw 0.1 -> candidates 0.1, 1.67, -3.04 (out of range), -1.47; wrist at 1.6
        g = GraspPlanner().plan(self.est(0.1), current_yaw=1.6)
        self.assertAlmostEqual(g.yaw, 0.1 + np.pi / 2, places=9)
        # candidate beyond the wrist limit must never be chosen even if it is closest
        g = GraspPlanner().plan(self.est(0.5), current_yaw=1.95)  # 0.5+pi/2 = 2.07 > limit
        self.assertLessEqual(abs(g.yaw), C.YAW_LIMIT)

    def test_yaw_inside_wrist_limit(self):
        for y in np.linspace(-np.pi / 4, np.pi / 4, 21):
            g = GraspPlanner().plan(self.est(y))
            self.assertLessEqual(abs(g.yaw), C.YAW_LIMIT)

    def test_too_wide_rejected(self):
        with self.assertRaises(GraspError):
            GraspPlanner().plan(self.est(0.0, size=0.085))

    def test_opening_leaves_margin(self):
        g = GraspPlanner().plan(self.est(0.0, size=0.04))
        self.assertGreaterEqual(g.open_width - 0.04, 0.012)
        self.assertLessEqual(g.open_width, C.MAX_OPEN)


class MiscTests(unittest.TestCase):
    def test_min_jerk_endpoints_and_monotone(self):
        self.assertEqual(min_jerk(0.0), 0.0)
        self.assertAlmostEqual(min_jerk(1.0), 1.0)
        v = [min_jerk(t) for t in np.linspace(0, 1, 101)]
        self.assertTrue(all(b >= a for a, b in zip(v, v[1:])))
        h = 1e-6
        self.assertAlmostEqual((min_jerk(h) - min_jerk(0)) / h, 0.0, places=3)      # v(0) = 0
        self.assertAlmostEqual((min_jerk(1) - min_jerk(1 - h)) / h, 0.0, places=3)  # v(1) = 0

    def test_wilson_known_values(self):
        lo, hi = wilson_interval(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        lo, hi = wilson_interval(100, 100)
        self.assertAlmostEqual(hi, 1.0, places=9)
        self.assertLess(lo, 1.0)
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_scenario_seeded_and_in_range(self):
        self.assertEqual(sample_scenario(7), sample_scenario(7))
        self.assertNotEqual(sample_scenario(7), sample_scenario(8))
        for s in range(200):
            sc = sample_scenario(s)
            self.assertTrue(C.SPAWN_X[0] <= sc.cube_xy[0] <= C.SPAWN_X[1])
            self.assertTrue(C.SPAWN_Y[0] <= sc.cube_xy[1] <= C.SPAWN_Y[1])
            self.assertTrue(C.PLACE_Y[0] <= sc.place_xy[1] <= C.PLACE_Y[1])
            self.assertGreater(np.hypot(sc.cube_xy[0] - sc.place_xy[0],
                                        sc.cube_xy[1] - sc.place_xy[1]), 0.06)

    def test_noise_scale_zero_is_noiseless(self):
        sc = sample_scenario(1, noise_scale=0.0)
        self.assertEqual((sc.depth_sigma, sc.depth_dropout, sc.cam_pos_sigma, sc.cam_rot_sigma),
                         (0, 0, 0, 0))

    def test_mjcf_is_well_formed_and_consistent(self):
        scn = sample_scenario(5)
        root = ET.fromstring(build_mjcf(scn, C.make_camera()))
        joints = {j.get("name") for j in root.iter("joint")}
        self.assertTrue(set(JOINTS) <= joints)
        acts = [a.get("name") for a in root.find("actuator")]
        self.assertEqual(tuple(acts), ACTUATORS)
        for a in root.find("actuator"):
            self.assertIn(a.get("joint"), joints)
        cube = [g for g in root.iter("geom") if g.get("name") == "cube_geom"][0]
        self.assertAlmostEqual(float(cube.get("size").split()[0]), scn.cube_size / 2, places=8)
        cam = root.find("worldbody").find("camera")
        xy = np.array(cam.get("xyaxes").split(), float)
        right, up = xy[:3], xy[3:]
        self.assertAlmostEqual(float(right @ up), 0.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(right)), 1.0, places=6)


class EpisodeTests(unittest.TestCase):
    def test_same_seed_same_result(self):
        b = MockBackend()
        a = run_episode(b, sample_scenario(11)).to_dict()
        c = run_episode(b, sample_scenario(11)).to_dict()
        a.pop("wall_time_s"), c.pop("wall_time_s")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(c, sort_keys=True))

    def test_noiseless_mock_episode_succeeds_and_is_scored_on_ground_truth(self):
        b = MockBackend()
        r = run_episode(b, sample_scenario(2, noise_scale=0.0))
        self.assertTrue(r.success, r.failure)
        self.assertLess(r.final_xy_err_mm, 20.0)

    def test_scoring_is_not_fooled_by_pipeline_claiming_done(self):
        # sabotage: object teleports after release -> controller says ok, scorer must say fail
        class Sabotaged(MockBackend):
            def object_pose(self):
                p = super().object_pose()
                p["xy"] = p["xy"] + [0.1, 0.0]
                return p
        r = run_episode(Sabotaged(), sample_scenario(2, noise_scale=0.0))
        self.assertFalse(r.success)
        self.assertEqual(r.failure, "placement:off_target")

    def test_missing_object_is_reported_as_perception_failure(self):
        class Empty(MockBackend):
            def render_depth(self):
                return np.zeros((C.IMG_H, C.IMG_W))
        r = run_episode(Empty(), sample_scenario(0))
        self.assertFalse(r.success)
        self.assertEqual(r.failure, "perception:too_few_valid_pixels")   # all-invalid depth


if __name__ == "__main__":
    unittest.main()
