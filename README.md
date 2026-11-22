# Simulated pick-and-place (MuJoCo, CPU only)

A depth camera with a (slightly wrong) known pose sees a cube on a table; the pipeline
estimates the cube's pose, plans a top-down parallel-jaw grasp, executes pick-and-place
with a closed-loop controller, and is scored over seeded trials on simulator ground truth.

```
depth image + estimated camera pose
   -> perception.estimate_object   (table-plane fit, segmentation, top-face rectangle -> x, y, yaw, size)
   -> grasp.GraspPlanner           (4 yaw candidates, wrist-limit + rotation cost, jaw opening)
   -> controller.PickPlaceController (min-jerk moves, measured-state settling, grasp/slip checks)
   -> episode.run_episode          (scores final cube pose from ground truth, not from the pipeline)
   -> eval                         (seeds -> results.jsonl + summary.json, Wilson 95% CI, failure modes)
```

**Status: read `AGENT_LOG.md` first.** The MuJoCo backend was written without MuJoCo
available and has since been run by the user on MuJoCo 4.0.1: 98.5 % success over 1,100
nominal-noise seeds, 98.5 % with no noise, 73.5 % at 3x noise. The nominal failures are all
heavy cubes (above about 72 g): they sag 4-10 mm in the grasp, the controller assumes no sag,
and the cube meets the table about 3 mm early, right at the controller's 3 mm settle
tolerance. Why they sag is not yet known: the grip force is measured at 7-8 N per pad, so it is not a weak grip. The mock backend's success rates are **not**
MuJoCo measurements.

## Run

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v          # 37 tests, no MuJoCo needed
python scripts/smoke_mujoco.py                   # FIRST thing to run with MuJoCo installed
python -m pickplace.eval --backend mujoco --seeds 0:100 --check-determinism 10 --out runs/mujoco_nominal
python -m pickplace.eval --backend mujoco --seeds 0:100 --noise-scale 0 --out runs/mujoco_noiseless
python -m pickplace.eval --backend mujoco --seeds 0:100 --noise-scale 3 --out runs/mujoco_3x
python -m pickplace.eval --backend mock   --seeds 0:200 --out runs/mock_nominal   # pipeline-logic check only
python -m pickplace.eval --backend mujoco --seeds 100:1100 --guarded-place --out runs/mujoco_guarded   # experimental variant, see AGENT_LOG.md
python scripts/diagnose_seed.py --backend mujoco --seeds 46,38,106,166,5,6 --noise-scale 1  # held-cube sag + grip forces per phase
```

`--noise-scale` multiplies depth noise (0.5 mm), pixel dropout (0.5 %), camera position
error (3 mm/axis) and camera orientation error (0.3 deg/axis).

## Layout

| path | role |
|---|---|
| `pickplace/camera.py` | pinhole model, look-at, project/back-project, extrinsic perturbation |
| `pickplace/perception.py` | depth-only cube pose estimate |
| `pickplace/grasp.py` | grasp planner |
| `pickplace/controller.py` | state machine + trajectories (opt-in `guarded_place` variant) |
| `pickplace/scenario.py` | seeded randomisation (separate RNG streams for scene / perception noise) |
| `pickplace/episode.py` | one episode, scoring, failure classification |
| `pickplace/eval.py` | seeded evaluation CLI |
| `pickplace/backends/scene.py` | MJCF: gantry + wrist yaw + two-finger jaw, cube, table, camera |
| `pickplace/backends/mujoco_backend.py` | MuJoCo backend (depth via `mj_ray`, no OpenGL) |
| `pickplace/backends/mock.py` | idealised kinematic backend with exact analytic depth |
| `scripts/smoke_mujoco.py` | engine-level checks to run before trusting any MuJoCo number |
| `scripts/diagnose_seed.py` | per-phase trace of the held cube for chosen seeds |
| `tests/test_pipeline.py` | unit + regression tests |
| `runs/` | mock outputs (authoring session) and the user's MuJoCo 4.0.1 runs |

## Success definition

Final cube centre within 2 cm of the place target (xy), tilt <= 10 deg, and resting height
within 5 mm of half the cube edge, measured from `backend.object_pose()` (never from the
pipeline's estimates). The controller must also report that it completed the sequence: an
abort counts as a failure even if the cube happens to be at the target.
