"""Seeded evaluation.

    python -m pickplace.eval --backend mujoco --seeds 0:100 --out runs/mujoco_nominal
    python -m pickplace.eval --backend mock   --seeds 0:100 --out runs/mock_nominal

Writes results.jsonl (one episode per line, full scenario included so any seed can be
replayed) and summary.json.  Re-running with the same seeds must reproduce results
exactly (wall time aside); `--check-determinism N` re-runs the first N seeds to verify.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

from .backends import make_backend
from .controller import ControllerConfig
from .episode import run_episode
from .scenario import sample_scenario
from .stats import wilson_interval


def parse_seeds(spec: str) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in spec.split(",")]


def _comparable(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "wall_time_s"}


def summarize(results: list[dict], backend: str, noise_scale: float) -> dict:
    n = len(results)
    k = sum(r["success"] for r in results)
    lo, hi = wilson_interval(k, n)
    fails = Counter(r["failure"] for r in results if not r["success"])

    def stat(key):
        v = np.array([r[key] for r in results if r[key] is not None], float)
        return None if v.size == 0 else {"mean": float(v.mean()), "p95": float(np.percentile(v, 95)),
                                         "max": float(v.max()), "n": int(v.size)}
    return {
        "backend": backend, "noise_scale": noise_scale, "episodes": n, "successes": k,
        "success_rate": k / n if n else None, "wilson95": [lo, hi],
        "failure_modes": dict(fails.most_common()),
        "perception_xy_err_mm": stat("perc_xy_err_mm"),
        "perception_yaw_err_deg": stat("perc_yaw_err_deg"),
        "perception_size_err_mm": stat("perc_size_err_mm"),
        "final_xy_err_mm": stat("final_xy_err_mm"),
        "sim_time_s": stat("sim_time_s"), "wall_time_s": stat("wall_time_s"),
        "seeds": [results[0]["seed"], results[-1]["seed"]] if results else None,
    }


def env_info(backend: str) -> dict:
    """Library versions, so a result can be tied to the stack that produced it."""
    import platform

    import cv2
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "numpy": np.__version__, "opencv": cv2.__version__}
    if backend == "mujoco":
        import mujoco
        info["mujoco"] = mujoco.__version__
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mujoco", "mock"], required=True)
    ap.add_argument("--seeds", default="0:100")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-determinism", type=int, default=0, metavar="N")
    ap.add_argument("--guarded-place", action="store_true",
                    help="experimental controller variant: accept a stalled TCP just above the "
                         "place height (a different configuration; not comparable to default runs)")
    args = ap.parse_args(argv)
    ctrl_cfg = ControllerConfig(guarded_place=args.guarded_place)

    try:
        backend = make_backend(args.backend)
    except ModuleNotFoundError as e:
        print(f"cannot start backend '{args.backend}': {e}", file=sys.stderr)
        return 2
    if args.backend == "mock":
        print("*** MOCK BACKEND: idealised kinematics, NOT a MuJoCo measurement ***")

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.out, exist_ok=True)
    results = []
    with open(os.path.join(args.out, "results.jsonl"), "w") as f:
        for s in seeds:
            r = run_episode(backend, sample_scenario(s, args.noise_scale),
                            ctrl_cfg=ctrl_cfg).to_dict()
            results.append(r)
            f.write(json.dumps(r) + "\n")
            print(f"seed {s:5d}  {'OK ' if r['success'] else 'FAIL'}  {r['failure'] or ''}")

    summary = summarize(results, args.backend, args.noise_scale)
    summary["env"] = env_info(args.backend)
    summary["controller"] = {"guarded_place": args.guarded_place}
    if args.check_determinism:
        bad = []
        for r0 in results[:args.check_determinism]:
            r1 = run_episode(backend, sample_scenario(r0["seed"], args.noise_scale),
                             ctrl_cfg=ctrl_cfg).to_dict()
            if _comparable(r0) != _comparable(r1):
                bad.append(r0["seed"])
        summary["determinism_checked"] = min(args.check_determinism, len(results))
        summary["determinism_mismatches"] = bad
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    lo, hi = summary["wilson95"]
    print(f"\n[{args.backend}] success {summary['successes']}/{summary['episodes']} = "
          f"{summary['success_rate']:.3f}  (95% Wilson CI {lo:.3f}-{hi:.3f})")
    print("failure modes:", summary["failure_modes"] or "none")
    if args.check_determinism:
        print("determinism mismatches:", summary["determinism_mismatches"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
