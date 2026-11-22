# Agent log

Written by the coding agent (Claude, Sonnet 5.5) that built this project.
Purpose: say what was built, how each part was checked, what went wrong along the way,
and — most importantly — what has **not** been verified.

## 0. Headline

| question | answer |
|---|---|
| Was the pipeline built end to end? | Yes: camera pose, perception, grasp planner, controller, seeded eval, tests. |
| Was a **MuJoCo** success rate measured? | **Yes, by the user, on MuJoCo 4.0.1** (section 4b): 98.5 % (1084/1100, 95 % CI 97.7-99.1) at nominal noise, 98.5 % with no noise, 73.5 % at 3x noise. I could not run MuJoCo in my workspace; I recomputed the figures from the per-seed files the user sent. |
| What did I verify myself? | Pipeline logic, harness, determinism and scoring on an idealised **mock backend** (section 4), and the analysis of the user's per-seed data. Mock rates say nothing about contact physics. |
| Main finding | At nominal noise the failures follow cube mass (all 19 are cubes above ~72 g), not perception error. The held-cube trace shows why they fail: heavy cubes sag 4-10 mm in the grasp, the controller assumes no sag, and the cube meets the table about 3 mm early, right at the 3 mm settle tolerance. |
| Still open | **Why heavy cubes slip** despite a measured 7-8 N per pad (about 20x the friction needed): contact-solver creep is the leading candidate and is untested. Also MuJoCo 3.x reproduction and low-friction grasps (friction is effectively inert in this eval). See section 5. |

## 1. Environment findings (checked, not assumed)

- Python 3.13, numpy 2.5, scipy, opencv 5.0, imageio, matplotlib present; **`mujoco` missing**.
- Searched the filesystem for any mujoco install or wheel: none. Network access for `pip` is off.
- libEGL/libGL exist but no PyOpenGL. Design consequence: depth comes from per-pixel `mj_ray`
  through my own pinhole model, so the MuJoCo path needs **no renderer** (EGL/OSMesa) at all.
- 2 CPU cores.

Decision: build the physics behind a small backend interface (`state`, `step`, `render_depth`,
`object_pose`) so everything except the physics can be run and tested here, and label the
mock loudly (banner printed by the eval, `backend` field in every result row and summary).

## 2. What was built

- **Camera pose** (`camera.py`): OpenCV-convention pinhole model; `look_at`; project /
  back-project; `perturbed()` applies a random extrinsic error. Perception is given a
  *perturbed* camera (3 mm / 0.3 deg per axis at nominal) while the simulator uses the true
  one, so calibration error is part of what the eval measures. The MuJoCo backend asserts at
  every reset that MuJoCo's camera pose equals the `CameraModel` (axis conversion: MuJoCo
  x=right, y=up, z=back).
- **Perception** (`perception.py`): depth-only. Fit the table plane to the image, take
  heights above it, largest connected component above 8 mm, top-face slice, minimum-area
  rectangle -> centre, yaw mod 90 deg; size = top height (cube assumption, cross-checked
  against measured top edge length).
- **Grasp planner** (`grasp.py`): four yaw candidates (`yaw + k*90deg`), wrist limit +-2 rad,
  min rotation cost, jaw opening = size + 30 mm capped at 90 mm, TCP at mid-height of the cube.
- **Controller** (`controller.py`): approach / descend / close / lift / transit / place-descend /
  release / retreat. Min-jerk segments, then a settle loop on the *measured* TCP with a
  timeout. Grasp verified from measured jaw width after closing, re-checked after lift and
  transit (slip detection). It never reads ground truth.
- **Seeded eval** (`scenario.py`, `episode.py`, `eval.py`): everything random derives from
  the seed with separate RNG streams (scene vs perception noise). Success is judged on
  simulator ground truth with fixed tolerances. Each failure is classified by stage
  (`perception:*`, `planning:*`, `control:*`, `placement:*`). Output: `results.jsonl`
  (full scenario per row, so any seed is replayable) and `summary.json` (success rate,
  Wilson 95 % CI, failure modes, perception/placement error stats).
- **MuJoCo scene** (`backends/scene.py`): 4-DOF gantry (x, y, z slides + wrist yaw) with a
  two-finger jaw, gravity-compensated, position actuators; cube mass/size/friction baked
  into the MJCF per scenario (a fresh model per episode, avoiding stale `geom_rbound`).
  A gantry was chosen deliberately: it removes IK as a variable so failures point at
  perception, grasping or contact rather than kinematics.
- **Mock backend** (`backends/mock.py`): exact analytic ray/box depth through the same camera
  model (so perception is tested against true geometry) plus a hand-written grasp rule.

## 3. How it was checked (what was actually run)

All results below are from this session.

1. **Unit tests: 29, all passing** (`python -m unittest discover -s tests`). They cover:
   camera axes orthonormal/right-handed, camera looks at its target, project/back-project
   round trip, image "up" = world up, spawn region inside the image with margin, >8 px on a
   4 cm cube, ray-box intersection on a known case; perception accuracy over 60 random
   poses, empty / out-of-view / wrong-pose cases; planner choices and limits; min-jerk
   endpoints; Wilson CI against known values; scenario determinism and ranges; MJCF is
   well-formed XML with consistent joint/actuator names and orthonormal camera axes;
   same seed twice gives an identical episode; a **sabotaged backend** that moves the cube
   after the controller finishes is scored as a failure (proves scoring uses ground truth,
   not the pipeline's own claim of success).
2. **Seeded eval on the mock backend**, 200 seeds, three noise levels, plus a determinism
   re-run of the first 20 seeds at each level (0 mismatches each time).
3. **Debugging by tracing rather than tuning** — three things went wrong, in order:

   - *Unit test threshold.* The first run failed on a 4.9 deg yaw error. I looked at the
     distribution before changing the test: mean 0.46 deg, p90 1.3 deg, and the worst case
     (seed 45) has a true yaw of 4.9 deg and was estimated as ~0. That is pixel
     quantisation (~4 mm/px) on a nearly axis-aligned cube, a real resolution limit and
     harmless for grasping (5 deg is ~1.7 mm of corner offset against 15 mm jaw margin).
     The test now asserts mean < 1 deg, p90 < 2 deg, max < 6 deg, with the reason written
     beside it. (The other failure in that run was my test: it expected a yaw candidate
     outside the wrist limit.)
   - *A mock bug that I first misread as a result.* The first nominal-noise run gave
     **44/100**, mostly `grasp_miss`. Perception error on the failing seeds was only 2-4 mm,
     so I traced seed 3 step by step: the cube was well aligned but the jaws closed to
     zero without ever blocking. My mock required the cube to stay inside `width/2 + 2 mm`
     while closing, leaving a 1.4 mm "blocked" window that the 3 mm-per-tick finger motion
     skipped. A real pad would push an off-centre cube to the centre. Fixed in
     `mock.py`. The 44 % number was an artefact and should not be quoted.
   - *A real perception weakness.* After that fix, 4/100 still failed with
     `object_clipped_by_image_border`. Hypothesis tested before fixing: through the
     perturbed camera, bare table reads up to 9-15 mm above z = 0 on those seeds (controls:
     within +-5 mm), exceeding the fixed 8 mm threshold, so the table became the "object"
     (components of 2,435-14,462 px vs ~100 px for the cube). Fix: fit the table plane from
     the depth image and measure heights relative to it; reject implausible planes. Added
     a regression test on those four seeds, a pure height-offset test (camera 2 cm too
     high is cancelled), and an implausible-plane test.
4. **Cross-check of the 3x-noise failures.** After the fixes, all 73 failures at 3x noise
   had perception xy error >= 16 mm (the mock's jaw margin ~15-17 mm), none below; so those
   failures are a genuine tolerance effect of the injected calibration error, not another
   bug. (Six successes had error > 17 mm because error along the pad axis is tolerated more
   than across the jaws.)
5. **Graceful degradation:** `--backend mujoco` without MuJoCo exits with code 2 and a clear
   message; `py_compile` passes for the MuJoCo backend, scene and smoke script.

## 4. Results — MOCK BACKEND ONLY (not MuJoCo)

200 seeds (0-199), `python -m pickplace.eval --backend mock --noise-scale X`:

| noise scale | success | 95 % Wilson CI | failure modes |
|---|---|---|---|
| 0 (no noise, no calibration error) | 200/200 | 0.981-1.000 | none |
| 1 (nominal: 0.5 mm depth, 3 mm / 0.3 deg camera) | 200/200 | 0.981-1.000 | none |
| 3 | 127/200 = 0.635 | 0.566-0.699 | 73 x `control:grasp_miss` |

At nominal noise (200 seeds) the perception xy error was 5.3 mm on average, max 15.5 mm,
dominated by the injected camera-pose error, which is irreducible for a depth-only
estimator that trusts its calibration (0.33 mm mean with no noise). Mean wall time per mock
episode ~12 ms.

**How to read this:** 100 % at nominal is a statement that the mock's tolerances (12 deg
alignment, 15 mm along pads, ~17 mm across jaws) are looser than the nominal error. Those
tolerances are my guesses, not measurements of any real gripper. The informative parts are
the harness behaviour (determinism, failure classification, CIs) and the perception
error statistics, which come from exact geometry rather than from a model of contact.

## 4b. Results — MuJoCo 4.0.1, run by the user

Provenance. The user ran these on their own machine and sent me the four `summary.json` and
four `results.jsonl` files (copied to `runs/mujoco_*`). I recomputed every summary figure
from the raw per-seed rows and they match. I did not observe the runs. Also as reported by
the user in chat: the smoke test passed (I have not seen its output); no files were edited
before the first run; Python 3.12 virtualenv, `pip install -r requirements.txt`; MuJoCo
**4.0.1**. The first zip's `requirements.txt` said `mujoco>=3.1` with no upper bound (the
`<4` pin came later), so pip took the newest release. The code was written against the 3.x
API and ran unmodified on 4.0.1. These summaries predate the `env` field, so the version is
not recorded in them, and I am assuming all four runs used the same environment. Whether 3.x
gives the same numbers is untested.

Controller provenance (2026-11-22). When the user ran the updated `diagnose_seed.py`, the
traceback showed their local `pickplace/controller.py` differing from the one I shipped. The
user then sent the file, and I diffed it against mine: the changes are purely additive and
opt-in. They add a `guarded_place` option (default `False`) with `stall_max`, `stall_speed`
and `stall_ticks` settings. In that mode `place_descend` accepts a TCP that has stopped
moving up to 12 mm above its target, then re-checks the jaw width so a cube that slid out is
still reported as `slipped:place_descend`. With `guarded_place` left at its default the code
path is the same as before, and 600 mock episodes (seeds 0-199 at noise 0, 1 and 3) came out
row-for-row identical to the earlier mock runs with this controller installed. I do not know
who made the edit or when relative to the four MuJoCo eval runs, because the user has not
said; the comment in the file cites the ~77-79 g observation, so it was written after the
mass analysis. The eval runs in this section are valid for the default controller if they
were run with the default config, which is the only mode `eval.py` could select until the
`--guarded-place` flag was added (2026-11-22).

| run | seeds | noise | success | 95 % Wilson CI | failure modes |
|---|---|---|---|---|---|
| nominal | 0-99 | 1x | 99/100 | 94.6-99.8 % | 1 x timeout:place_descend |
| nominal | 100-1099 | 1x | 985/1000 = 98.5 % | 97.5-99.1 % | 15 x timeout:place_descend |
| **nominal, pooled** | 0-1099 | 1x | **1084/1100 = 98.5 %** | 97.7-99.1 % | 16 x timeout:place_descend |
| noiseless | 0-199 | 0 | 197/200 = 98.5 % | 95.7-99.5 % | 3 x timeout:place_descend |
| 3x noise | 0-199 | 3x | 147/200 = 73.5 % | 67.0-79.1 % | 13 place_descend, 13 placement:unstable, 12 grasp_miss, 4 timeout:descend, 4 placement:off_target, 4 slipped:transit, 3 slipped:lift |

Other figures. Nominal, seeds 100-1099: perception xy error mean 5.4 mm (p95 10.2, max
16.3); yaw error mean 0.71 deg (max 5.1); size error mean 0.13 mm; final cube-to-target xy
error mean 3.3 mm; 63 ms wall time and 6.4 s sim time per episode. Noiseless: perception xy
error 0.35 mm mean (max 2.5). 3x: perception xy error 15.8 mm mean (p95 30.4, max 46.6).
Determinism: seeds 0-9 re-run in the same process gave 0 mismatches. Six seeds also
reproduced exactly in a separate process on the same machine (section 4b, point 5b); not
checked across machines or library versions.

Overlap: the noiseless, 3x and first nominal runs share scenes for seeds 0-99 (same scene,
different noise). The 1000-seed nominal run is disjoint from 0-99 but overlaps the
noiseless and 3x runs on seeds 100-199.

### What the per-seed data show (computed from the results.jsonl files)

1. **Nominal noise costs nothing measurable.** 98.5 % nominal vs 98.5 % noiseless. On the
   100 shared scenes, 98 passed both ways, seed 46 failed both ways and seed 38 failed only
   without noise. The failures are not caused by perception error.

2. **The nominal failures are driven by cube mass.** Every failure at nominal noise (16) and
   without noise (3) is `timeout:place_descend`. Nominal runs: 0/864 failures below 70 g;
   1/149 at 70-76 g; 4/40 (10 %) at 76-78 g; 11/47 (23 %) at 78-80 g (mass is sampled from
   30-80 g). Pooling nominal and noiseless (1300 episodes), failing cubes average 78.0 g
   against 55.6 g for passing ones (Mann-Whitney p = 3e-12). No significant association with
   friction (p = 0.11), cube size, perception error or place position. Ten features were
   compared; one (cube y position) had p = 0.0075 before any multiple-comparison correction
   and I do not interpret it.

3. **My first reading was wrong.** The three noiseless failures all had low friction
   (0.62-0.68) as well as heavy cubes (77-78 g). On three samples I suspected friction; the
   full data say mass, and friction is not significant.

4. **Friction randomisation is largely inert in this eval.** The pads use friction 1.0 and,
   as I understand MuJoCo's default, the larger of the two geoms' coefficients applies, so a
   cube mu below 1.0 is overridden. The data are consistent with that: at 70-80 g the failure
   rate is 9.3 / 5.7 / 6.1 % across the three friction bins. So this eval does not test
   low-friction grasps even though mu is sampled from 0.6-1.2. Not fixed, because changing it
   changes every result.

5. **Two failure shapes among the 19.** 14 ended with the cube upright (tilt under 5 deg)
   within 5 mm of the target, 0.73-0.76 mm from nominal rest height (successful episodes:
   0.11 mm): the controller aborted its 3 mm / 1.5 s settle check while lowering the held
   cube, before opening the jaws. The other 5 are disturbed (3.3-3.8 mm off rest height, one
   tilted 79 deg, up to 19 mm off target).

5b. **What the held-cube trace shows.** The user ran `scripts/diagnose_seed.py` on MuJoCo
   4.0.1 for six seeds (output saved as `runs/mujoco_diagnose_trace_v1.txt`; it predates the
   force readout described below). `dz` is cube-centre height minus TCP height in mm, taken
   from ground truth for printing only (the controller cannot see it).

   | seed | mass | outcome | dz after lift | dz after transit | dz at end of place_descend | TCP z error at end of place_descend |
   |---|---|---|---|---|---|---|
   | 46 | 76.7 g | FAIL | -0.29 | -6.01 | -7.81 | 3.16 mm |
   | 38 | 76.6 g | ok | -0.86 | -4.28 | -7.60 | 2.98 mm |
   | 106 | 78.2 g | FAIL | -0.91 | -10.49 | -7.70 | 3.03 mm |
   | 166 | 77.1 g | FAIL | -0.84 | -6.92 | -23.5 | 16.3 mm (cube left the jaws) |
   | 6 | 79.4 g | ok | -0.79 | -6.03 | -7.62 | 3.00 mm |
   | 5 | 32.7 g | ok | -0.11 | -0.16 | -0.26 | -0.07 mm |

   - **Reproduction.** All six outcomes and final cube states (height error, tilt, xy error,
     perception error) equal the corresponding `results.jsonl` rows to the printed precision.
     That is a separate process on the same machine, so it is a cross-process determinism
     check for these six seeds (same machine and install only).
   - **Heavy cubes hang low in the grasp.** The 32.7 g cube does not move relative to the
     jaws. All five 77-79 g cubes end up 4-10 mm below the TCP after the transit, including
     seed 6, which passes. The sag mostly appears during the transit, not the lift.
   - **This is what trips the timeout.** The controller sets the place height assuming the
     cube centre is at TCP height (TCP target = half the cube size + 4 mm). With the cube
     about 7.7 mm low, it reaches the table while the TCP is still roughly
     `|dz| - 4 mm - pad penetration` above its target. That evaluates to 3.02 and 2.95 mm for
     seeds 46 and 106 against 3.16 and 3.03 observed. The TCP then only descends as the
     cube creeps back up through the pads (roughly 1 mm/s in the settle samples), against a
     3 mm tolerance and a 1.5 s timeout. Seeds 38 and 6 finish at 2.98 and 3.00 mm, just
     inside; 46 and 106 at 3.16 and 3.03, just outside. So among slipping cubes the
     pass/fail split is a coin flip at the tolerance rather than a different regime. This
     also resolves the sign of the 0.75 mm height error: the cube is pressed into the table
     (cube centre 19.81 mm vs rest height 20.60 mm for seed 46; 19.05 vs 19.80 for seed 106).
   - **Seed 166 is the same slip running away.** After the transit the cube hangs 6.9 mm
     low; during the descent it leaves the pads (jaw width reads 0.64 mm, i.e. closed on
     nothing; the cube is 46 mm below the TCP by t = 5.34 s) and lands tilted 79 deg. It is
     the smallest cube of the group (37.3 mm). Why it escaped when the others did not is
     untested.
   - **Contacts are soft.** The jaw width at the grasp reads 1.5-2.7 mm narrower than the
     cube for every seed, including the light one, so the pads sink into the cube by about
     a millimetre each regardless of mass.
   - **Grip force measured: the grip is not weak.** The user ran the force readout on the
     same six seeds (`runs/mujoco_diagnose_trace_v2_forces.txt`); outcomes and final cube
     states are identical to the first trace. Readout sanity: before the grasp the table
     force equals the cube weight to 0.01 N for all six (0.75 N for 76.7 g, 0.32 N for
     32.7 g). While the cube is carried each pad presses with 6.7-8.3 N and agrees with its
     jaw actuator to within 0.14 N, for the light cube (6.9-7.2 N) as much as the heavy
     ones. So my estimate of about 8 N per pad was right. Two pads at about 7.5 N with pad
     friction 1.0 could carry roughly 15 N; a 77 g cube asks for 0.76 N, about 5 % of that.
     The sag is therefore not explained by a weak grip or a saturated actuator. What is
     left: creep in MuJoCo's soft-contact friction (the `noslip_iterations` candidate), or
     an effective friction well below 1.0 that I do not understand. **I have not separated
     these and have no direct evidence for either.**
   - **Other readings from the same trace.** (a) At the end of `place_descend` the table
     pushes back with 9.6-10.1 N, about 13 times the cube's weight. It equals the z
     actuator gain (3000 N/m) times the TCP error plus the cube's weight to within 0.15 N
     on every seed (e.g. seed 46: 3.16 mm gives 9.5 + 0.75 = 10.25 N against 10.10 N read;
     seed 166: 16.3 mm gives 48.8 + 0.76 = 49.6 N against 49.5 N). So the "stall" is the
     arm pressing the held cube into the table, and the TCP creeps down as the cube slips up
     through the pads under that load, roughly two-thirds of the friction capacity.
     (b) Seed 166 at timeout: the right pad reads 48.9 N and the table 49.5 N, the cube is
     tilted 79.5 deg, and the jaw actuators read 0.8 / 0.6 N, i.e. the gripper is pressing a
     tilted cube into the table with one pad. (c) Across the five heavy seeds the sag after
     the transit rises with the perception xy error (1.5, 2.7, 3.1, 4.2, 5.4 mm gives 4.3,
     6.0, 6.0, 6.9, 10.5 mm; rank correlation 1.0 with n = 5, which is weak evidence on its
     own), and the cube tilts slightly more (0.3-2.4 deg, against 0.0 for the light cube).
     Seeds 38, 46 and 106 also fail in the noiseless run, so a grasp offset is not required
     for the sag; a noiseless trace of these seeds would show whether it changes the size.
   - **No fix has been made.** The offset is invisible to the controller, and any change
     (a larger place clearance, releasing when the TCP stalls, a firmer grip, solver
     settings) alters every number above. A larger clearance or looser tolerance would
     probably hide the slip rather than remove it.

6. **Scoring nuance.** A controller abort counts as a failure even if the cube is physically
   at the target, which was true of 14 of the 19. That is deliberate (the pipeline must
   complete the task), but it makes 98.5 % stricter than "the cube ends up in the right
   place".

7. **At 3x noise the cliff is perception error, not mass.** Success against perception xy
   error in the 3x run: under 20 mm, 137/143 = 95.8 %; 20-30 mm, 10/46 = 21.7 %; over 30 mm,
   0/11. Large errors cause missed grasps, pads landing on the cube, corner grasps that slip,
   and knocked-over cubes (11 of the 13 `place_descend` failures at 3x ended tilted over 45
   deg, and their masses span 55-79 g, so that is a different mechanism from point 2). The
   mock had put the cliff near 16 mm and scored 63.5 % at 3x; MuJoCo scored 73.5 %, so the
   mock was pessimistic there. The mock's 200/200 at nominal missed the mass effect
   entirely, as expected of a model with no contact physics.

8. **Speed.** About 60 ms per episode, so 100 seeds take about 6 s.

Limits of all of the above: one cube, success tolerances I chose (2 cm / 10 deg / 5 mm),
gantry arm with no IK, camera never occluded by the arm, noise models not calibrated to a
sensor, friction effectively fixed (point 4).

### Experimental variant: guarded place (no MuJoCo result yet)

`--guarded-place` (in `eval.py` and `diagnose_seed.py`) selects the controller variant above.
It targets the failure in point 5b: a heavy cube that sags in the grasp stops the TCP about
3 mm short, and the default 3 mm / 1.5 s settle check aborts. It does not remove the sag,
and it changes what counts as success: the 14 of 19 aborts where the cube was physically
at the target would not abort. Any number from it is a different configuration, to be
reported next to the default one on the same seeds, never in place of it. `summary.json`
records `controller.guarded_place`.

Tested here without MuJoCo, using a stub backend whose TCP stops short (8 tests): the default
times out; guarded accepts a stall 3.5 mm short and records it; guarded does not flag a
normal descent; guarded still times out when the stall is above `stall_max`; guarded
reports `slipped:place_descend` if the cube left the jaws; the stall must last
`stall_ticks`. Two deliberate breakages of the controller (the stall window, the width
re-check) each made a test fail. The stub is not a contact model, so none of this says how
the mode behaves on the engine. One side effect: in default mode a cube that dropped out of
the jaws during `place_descend` without stalling the TCP is not detected before release.

## 5. Status of the earlier unverified items, and what is still open

Resolved by the user's runs (smoke test reported as passing, plus about 1,700 MuJoCo
episodes): the MJCF compiles; the `mj_ray` depth path and the runtime camera-consistency
assert work; mock depth agrees with MuJoCo depth (smoke check 6, as reported by the user);
the arm tracks well enough for 98.5 % success; wall-clock cost is about 60 ms per episode;
determinism holds in-process and, for six seeds, across processes on the same machine;
the grip-force readout (`MuJoCoBackend.grip_forces`) runs on the engine and passes sanity checks
(table force before the grasp equals the cube weight; pad force agrees with the actuator).
The mechanism that trips the `place_descend` timeout is explained (section 4b, point 5b).

Still open:

- **Why heavy cubes slip in the grasp.** Grip force is measured and adequate (section 4b,
  point 5b), so the cause is not a weak grip. Candidate next steps, none run yet: (1) trace
  the same seeds with `--noise-scale 0` to see whether the sag depends on the grasp offset;
  (2) an isolated hang test outside the controller (grasp a cube, hold still, sweep the mass
  and record the drift), which would separate a friction effect from a motion effect;
  (3) re-run the hang test with MuJoCo's `noslip_iterations` option set, the standard
  setting for reducing contact creep.
- **MuJoCo 3.x reproduction.** Every run used 4.0.1.
- **The smoke-test output**, which I have not seen.
- **Determinism across machines and library versions.**
- **Low-friction grasps**, effectively untested (point 4).
- **Controller tuning.** Gains were never tuned on the engine. The place height ignores
  in-hand slip, and the 3 mm / 1.5 s settle criterion is exactly what the failing episodes
  tripped. The guarded-place variant has not been run on MuJoCo; suggested comparison on
  the same seeds: `python -m pickplace.eval --backend mujoco --seeds 100:1100 --guarded-place
  --out runs/mujoco_guarded_seeds100-1099` (default run: 985/1000) and the same with
  `--seeds 0:200 --noise-scale 3` (default: 147/200).

## 6. Known limitations of the design (independent of MuJoCo)

- One cube, flat on the table; no distractors or clutter. Perception assumes a cube
  (size = top height) and rejects shapes whose top edge disagrees.
- Depth only (no RGB), single shot (no re-observation or visual servoing).
- Gantry + wrist yaw instead of an articulated arm: no IK, singularities or joint limits
  beyond the wrist; the arm is invisible to the depth camera.
- Yaw resolution floor of a few degrees at ~4 mm/pixel (section 3).
- Noise models (Gaussian depth noise, random dropout, Gaussian extrinsic error) are
  illustrative, not calibrated to a real sensor.
- Success tolerances (2 cm, 10 deg, 5 mm) are chosen, not derived from a task. Success also
  requires the controller to report completion, so a controller abort counts as a failure
  even when the cube is at the target.
- Cube friction is sampled from 0.6-1.2 but pads use 1.0 and MuJoCo (as I understand its
  default) takes the larger coefficient, so friction below 1.0 has no effect (section 4b, point 4).

## 7. Files

- `runs/mock_noise{0.0,1.0,3.0}/`: mock backend, 200 seeds each (from my session).
- `runs/mujoco_nominal_seeds0-99/`, `runs/mujoco_nominal_seeds100-1099/`,
  `runs/mujoco_noiseless_seeds0-199/`, `runs/mujoco_3x_seeds0-199/`: MuJoCo 4.0.1 runs by the
  user (`summary.json` + `results.jsonl`), copied unmodified.
- `scripts/smoke_mujoco.py`: engine checks (reported as passing by the user).
- `scripts/diagnose_seed.py`: per-phase trace of the held cube for chosen seeds. The user ran the
  first version on MuJoCo (`runs/mujoco_diagnose_trace_v1.txt`) and the version with the force
  readout (`runs/mujoco_diagnose_trace_v2_forces.txt`).

Re-running the same command reproduces a run exactly apart from `wall_time_s`.
