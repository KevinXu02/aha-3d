# Experimental source-guided trajectory proposals

For the current SAM3/Pi3X workflow with trajectory scale, vertical placement,
smooth root correction, track termination and requested contact constraints,
use [Human tracking and trajectory](HUMAN_TRACKING_AND_TRAJECTORY.md).
The SAM image-landmark optimizer documented below is a separate historical
experiment; its no-scale input policy does not prohibit trajectory-only scale
in the newer depth-based route. Body scale remains fixed in both.

Status: experimental, proposal-only. This module compares small world-coordinate
translation changes while preserving native human scale, shape, pose and global
orientation. It is not a default reconstruction stage and does not replace the
[placement/contact acceptance gates](CONTACT_REFINEMENT.md) or
[motion policy](MOTION_CONTROLS.md). Optimizer convergence is not motion acceptance.

## Methods and bounds

`tools/gvhmr/trajectory_methods.py` produces an unchanged baseline and four matched
SAM image-landmark fits: shared constant XYZ, affine-in-time drift, cubic spline,
and the same spline with an additional immutable-contact speed/plant penalty.
An optional sixth arm adds weak SAM camera-hip temporal displacement information.
The latter is an uncalibrated sensitivity test; SAM absolute depth is not truth.
There is no camera-space root substitution or per-frame viewing-ray floor snap.

All fitted arms use the same robust UV objective, regularization and penetration
penalty. Default shared XY norm is bounded by 0.5 m and shared absolute Z by 0.1 m;
additional deformation norm is bounded by 0.3 m. These are separate budgets, so
total displacement can exceed the shared budget. Cubic B-spline knots are 1.5 s
apart. A convex basis and bounded controls enforce the deformation limit between
knots, not only at sampled frames. Shared and spline components are regularized;
their decomposition is not independently identifiable.

The common floor penalty uses sigma 0.01 m. The contact arm adds foot speed sigma
0.03 m/s and plant sigma 0.025 m on a fixed baseline cohort. A shared placement
change relocates a plant anchor uniformly; deformation should preserve it.
These engineering settings do not prove physical accuracy. Reviewed horizontal
floor/grounded-motion scope is required; unknown support is unsupported.

## Inputs

Use the original unmodified world-aligned baseline cache, its exact camera/time
basis, and reviewed SAM landmarks. Do not start from an earlier scaled or
frame-dependent-root cache. Native scale remains 1; no body dimensions are fitted.

- Baseline NPZ: `vertices[T,V,3]`, `joints[T,J,3]` with the first 22 SMPL-X joints,
  and uniformly increasing `time_seconds[T]`.
- Camera NPZ: `c2w[T,4,4]`, `K[T,3,3]`, and matching `time_seconds[T]`. The CLI also
  recognizes `camera_to_world` or `camera_to_world_cv`, and `intrinsics`. A single
  3x3 K is broadcast. Camera poses are OpenCV camera-to-world rigid transforms.
- Observation NPZ: increasing `time_seconds[S]`, `uv[S,12,2]`, `valid[S,12]`, and
  `joint_names[12]`. UV must use the same full-image pixel grid as K. Rows must map
  uniquely to the nearest output frame within half a frame interval. Preserve
  invalid rows and their timestamps; do not delete them to change the split.
- Optional contact NPZ: `contact_centers[T,2,3]`, `observed_contact[T,2]`, `scope[T]`,
  and matching `time_seconds[T]`. Prefer immutable baseline foot mesh centers and
  reviewed observed contact masks. Without these, the fallback uses baseline
  ankle/toe joint centers and a low/slow model-derived cohort, labeled accordingly.

Anatomical order and correspondence:

| Landmark pair | SMPL-X joint IDs | SAM MHR70 IDs |
| --- | --- | --- |
| Left/right hip | 1, 2 | 9, 10 |
| Left/right knee | 4, 5 | 11, 12 |
| Left/right ankle | 7, 8 | 13, 14 |
| Left/right shoulder | 16, 17 | 5, 6 |
| Left/right elbow | 18, 19 | 7, 8 |
| Left/right wrist | 20, 21 | 62, 41 |

`joint_names` must be exactly `left_hip, right_hip, left_knee, right_knee,
left_ankle, right_ankle, left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist`. Verify the SAM convention against the installed model;
this mapping is not permission to treat occluded inferred joints as observations.

The optional weak-3D arm additionally requires `sam_camera_joints[S,70,3]`, a
reviewed `weak3d_valid[S]` mask, and explicit `--sam-opencv-confirmed`. Coordinates
must be OpenCV right/down/forward, with camera translation already added to local
SAM joints. It uses only reviewed training hip midpoints, transformed to world,
then subtracts training-only temporal means. No scale fit or absolute-depth target
is used. Fewer than two supported training hip rows disables that arm.

## Fixed split and invocation

Sorted observation row parity is fixed: even rows train, odd rows held out,
including invalid rows. Default minimum support is four reviewed landmarks per
sample. A separately declared partial-body protocol may use two, preserving the
same bounds, losses and split. A two-landmark proposal remains underconstrained;
it does not relax the independent detector acceptance threshold. Do not select
protocols or weights by heldout results.

With a claimed output path, use a configured Python environment with
NumPy and SciPy (see [runtime rules](../MACHINE.md)):

```bash
python tools/gvhmr/trajectory_methods.py \
  --cache /absolute/baseline.npz --camera /absolute/camera.npz \
  --observations /absolute/reviewed_sam.npz \
  --contacts /absolute/immutable_contacts.npz \
  --output /absolute/new-proposals
```

Use `--config preset.json` for explicit overrides from module `DEFAULTS`; unknown
keys are rejected. For a separate partial-body protocol, a minimal preset is
`{"min_joints_per_sample": 2}`. Use a distinct output path for each protocol.

The Python interface is `fit_variants(joints, vertices, times, camera_to_world,
K, observations, config=None, *, contact_centers=None, observed_contact=None,
scope=None)`. It returns a mapping from method name to offsets and report.

## Outputs, evaluation and adoption boundary

Each variant writes `offsets.npz` containing `offsets_world[T,3]`,
`shared_offset[3]`, `deformation[T,3]`, `time_seconds[T]`, plus `report.json`.
The top-level `summary.json` hashes the inputs and implementation. Outputs are
translations only; the CLI does not mutate/reskin baseline arrays or accept a
candidate. Apply the same offset to every vertex/joint in a distinct candidate
copy before downstream evaluation.

Reports separate training and heldout SAM UV, behind-camera samples, correction
size/speed/acceleration, immutable-contact foot speed, and penetration. Empty
contact cohorts remain unverified. Check optimizer success and realized bounds.

`tools/gvhmr/trajectory_evaluation.py` provides the independent Python function
`evaluate_arrays`. Supply original/candidate geometry, exact timestamps, two
foot-vertex ID sets, both immutable `observed` and `model` contact masks, reviewed
scope, offsets, total XYZ bounds, and sparse output-frame indices. Its optional
projection inputs must use actual anatomical detector correspondences, not the
SAM fitting targets. It checks fixed-cohort contact speed, foot acceleration,
penetration, root-only geometry invariance, bounds and independent detector
projection across whole-clip, time-window, train and heldout cohorts. Total XYZ
bounds are an enclosing box; also inspect separate radial/deformation bounds in
the optimizer report. This evaluator has no CLI and never marks a result accepted.

Run the existing placement/contact gates and inspect full source comparisons
before adoption. A good SAM training fit cannot establish 3D root accuracy,
identity, contact correctness or camera/room fidelity. Do not wire an experimental
proposal into default reconstruction merely because the optimizer returned.

## Evidence limits

The initial lounge comparison is a research example, not an accepted universal
repair. Controlled unit tests exercise bound enforcement, input preservation,
fixed train/heldout separation including weak 3D, fixed contact cohorts, penetration
penalty, anatomical/timing rejection and independent evaluation. These tests do
not establish model accuracy, whole-clip visual quality or cross-scene success.
No accepted second real-scene trajectory repair is established by this module.

In the initial single-actor lounge experiment, two fixed protocols compared six
arms each, including the baseline. All twelve source-gate results failed. Under
the reviewed-partial protocol, 6–8 s detector P90 improved from 684.69 px to
378.52 px for the unconstrained spline, but 4–6 s worsened from 88.13 to 98.32 px.
Observed-contact speed P90 rose from 7.40 to 13.28 cm/s. Adding contact penalties
reduced that observed-cohort value to 7.13 cm/s, while the separate model-contact
cohort increased from 7.51 to 13.22 cm/s. Thus one cohort's improvement did not
establish overall contact quality. Weak SAM-3D displacement gave nearly identical
results to the contact spline in this experiment. Review of 36 rendered mesh
comparisons still found incorrect late foreground extent. No failed candidate
was promoted into a replacement animation. These are observed results from one
example, not estimates of general method performance.
