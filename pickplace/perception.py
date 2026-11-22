"""Depth-only object pose estimation.

Pipeline: back-project depth with the (estimated) camera pose -> keep points more than
`min_height` above the table -> largest connected component -> take the top-face slice
(side faces are visible because the camera is tilted, so they are excluded) ->
minimum-area rectangle in the table plane gives centre and yaw.

Assumption (documented limitation): the object is a cube resting flat on the table, so
its edge length equals its top height.  Yaw is only defined modulo 90 degrees and is
returned in [-pi/4, pi/4).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraModel


class PerceptionError(Exception):
    """Raised with a short machine-readable reason (e.g. 'no_object')."""


@dataclass(frozen=True)
class ObjectEstimate:
    xy: np.ndarray       # (2,) top-face centre == cube centre xy
    top_z: float         # height of the top face above the table
    size: float          # cube edge length (== top_z for a cube on the table)
    yaw: float           # radians, in [-pi/4, pi/4)
    width_meas: float    # measured top-face edge length (sanity check only)
    n_points: int


def corrupt_depth(depth: np.ndarray, rng: np.random.Generator, sigma: float,
                  dropout: float) -> np.ndarray:
    """Add gaussian depth noise and random invalid pixels (0.0). Deterministic in rng."""
    out = depth.copy()
    valid = out > 0
    out[valid] += rng.normal(0.0, sigma, size=int(valid.sum()))
    out[rng.random(out.shape) < dropout] = 0.0
    return out


def wrap_quarter(a: float) -> float:
    """Wrap an angle to [-pi/4, pi/4) (symmetry of a square)."""
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def fit_table_plane(pts: np.ndarray, iters: int = 4, inlier: float = 0.006):
    """Robust least-squares plane z = a*x + b*y + c through (N,3) points, where most
    points are bare table.  Trimmed iteratively so the object does not bias the fit.
    Returns (a, b, c, n_inliers)."""
    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    keep = np.ones(len(pts), dtype=bool)
    coef = np.zeros(3)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[keep], pts[keep, 2], rcond=None)
        keep = np.abs(A @ coef - pts[:, 2]) < inlier
        if keep.sum() < 3:
            break
    return float(coef[0]), float(coef[1]), float(coef[2]), int(keep.sum())


def estimate_object(depth: np.ndarray, cam: CameraModel, table_z: float = 0.0,
                    min_height: float = 0.008, min_pixels: int = 30,
                    top_band: float = 0.005, size_consistency: float = 0.015,
                    plane_tol: float = 0.03) -> ObjectEstimate:
    """Heights are measured above a table plane fitted to the depth image itself, so a
    camera height/tilt calibration error cancels out (only the in-plane xy/yaw error
    remains).  `table_z` is used only as a plausibility check on the fitted plane."""
    pts = cam.backproject(depth)
    valid = depth > 0
    if valid.sum() < 1000:
        raise PerceptionError("too_few_valid_pixels")
    a, b, c, n_in = fit_table_plane(pts[valid])
    mx, my = pts[valid][:, 0].mean(), pts[valid][:, 1].mean()
    if n_in < 0.5 * valid.sum() or abs(a * mx + b * my + c - table_z) > plane_tol:
        raise PerceptionError("table_plane_implausible")
    pts = pts.copy()
    pts[..., 2] = pts[..., 2] - (a * pts[..., 0] + b * pts[..., 1] + c) + table_z
    mask = valid & (pts[..., 2] > table_z + min_height)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise PerceptionError("no_object")
    areas = stats[1:, cv2.CC_STAT_AREA]
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] < min_pixels:
        raise PerceptionError("too_few_pixels")
    x0, y0 = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
    w0, h0 = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
    if x0 == 0 or y0 == 0 or x0 + w0 >= cam.width or y0 + h0 >= cam.height:
        raise PerceptionError("object_clipped_by_image_border")

    P = pts[labels == k]
    z_hi = np.percentile(P[:, 2], 98)
    top = P[P[:, 2] >= z_hi - top_band]
    if len(top) < 10:
        raise PerceptionError("top_face_too_small")
    top_z = float(np.median(top[:, 2])) - table_z

    rect = cv2.minAreaRect(top[:, :2].astype(np.float32))
    box = cv2.boxPoints(rect)                       # 4 corners, consecutive
    e0, e1 = box[1] - box[0], box[2] - box[1]
    yaw = wrap_quarter(float(np.arctan2(e0[1], e0[0])))
    width_meas = float(0.5 * (np.linalg.norm(e0) + np.linalg.norm(e1)))
    if abs(width_meas - top_z) > size_consistency:
        raise PerceptionError("not_cube_shaped")    # top edge disagrees with height
    return ObjectEstimate(xy=np.asarray(rect[0], dtype=float), top_z=top_z, size=top_z,
                          yaw=yaw, width_meas=width_meas, n_points=len(top))
