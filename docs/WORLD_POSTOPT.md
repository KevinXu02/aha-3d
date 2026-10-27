# Full source-video world-postopt pipeline

Updated for corrected v2 on 2026-09-17. Earlier reports using acceleration weight
0.001 and camera-derived orientation are historical. Their camera-path-based
ranking is superseded; use the GVHMR global comparator and motion metrics below.

The full integration follows the external `world-postopt` skill from the source
video, not only its optimizer. Use `tools/gvhmr/world_pipeline.py prepare` to
create this resumable graph:

```text
source video → fresh SAM3 masks → bbox gate and reviewed lifecycle
             → fresh PMPose/features/GVHMR with scene floor/upright
             → full-rate dense Pi3X (Kimodo completion skipped by default)
             → camera-space/world adaptation → v2 → metrics and video review
```

The default 2D detector is mask-conditioned **PMPose-h**; ViTPose requires an
explicit `--pose-detector vitpose`. PMPose runs in its separate environment before
the first GVHMR inference, with exact input-keypoint verification and no cache
cloning. See the [whole pipeline](REAL2SIM_PIPELINE.md) for empty-mask handling,
lifecycle limits, runtime overrides and the optional endpoint-constrained Kimodo
completion experiment. The owner rejected its current visual quality; new runs
skip it unless `--kimodo-completion` is explicitly supplied.

Only **v2** runs. Reviewed Pi3X observations establish the same-shot reference
basis; dense Pi3X inference is still rerun over the complete source clip. GVHMR
uses the reviewed bundle's rotations, as in the external skill, and the dense
cameras subsequently lift its camera-frame bodies into world coordinates.
There are no YOLO tracking or DPVO inference stages.

```bash
python tools/gvhmr/world_pipeline.py prepare \
  --video /path/to/normalized_source.mp4 \
  --actor-id REVIEWED_ACTOR --box 100 100 400 650 --prompt-frame 0 \
  --lifecycle-review /path/to/source_reviewed_lifecycle.json \
  --pi3x-bundle /path/to/reviewed_same_shot_bundle \
  --sam3 /path/to/sam3 \
  --sam3-python /path/to/sam3/venv/bin/python --sam3-checkpoint /path/to/sam3.pt \
  --gvhmr-repo /path/to/GVHMR_BEDLAM2 \
  --pmpose-python /path/to/bmp/venv/bin/python --pmpose-root /path/to/BBoxMaskPose \
  --pmpose-checkpoint /path/to/PMPose-h-1.0.0.pth \
  --pi3 /path/to/Pi3 --pi3-checkpoint /path/to/model.safetensors \
  --smplx-model /path/to/SMPLX_NEUTRAL.npz \
  --scene-ground /path/to/scene_ground.json \
  --python /path/to/gvhmr/venv/bin/python --gpu 0 \
  --out runs/_tools/world-postopt/EXPERIMENT
bash tools/human_experiments/run.sh execute \
  runs/_tools/world-postopt/EXPERIMENT/experiment.json \
  --run-dir runs/_tools/world-postopt/EXPERIMENT/run
```

Tracking runs on the local GPU; `--gpu` is optional (default: an inherited
single-device `CUDA_VISIBLE_DEVICES`, else device 0). GVHMR's default intrinsics
(`estimate_K`) use the image diagonal as focal length, a fixed 53° diagonal field of
view; add `--intrinsics pi3x` to use the reviewed same-shot Pi3X intrinsics for
wider or narrower lenses. The default remains `upstream`.
SAM3 tracking currently requires 1280×720, normalized 30 Hz sources;
the wrapper rejects other raster sizes. No runtime is installed or modified by this tool.

Tracking creates a fresh predictor per propagation direction. A later prompt
uses anchor-to-end and reversed anchor-to-start streams. Bboxes use the skill's
trailing accepted-frame maximum, both scale and actual mask-fill gates,
centre/extent interpolation, and 1.2× crop enlargement. Raw masks stay available
for reviewing wrong identities, occlusion and unsupported intervals.

The lifecycle stage records the mask-derived support-end proposal separately
from the source-reviewed exit. It requires the existing
[lifecycle JSON contract](HUMAN_TRACKING_AND_TRAJECTORY.md) and refuses a review
whose reviewer is labeled `derived:`. A tiny/empty mask does not prove physical
exit; partially cropped actors remain active until the reviewed exit. This is a
deliberate correction to the external harness's automatic lifecycle labeling.
The current GVHMR lifecycle interface handles a terminal inactive suffix; early
identity failures must be flagged during visual review, not silently accepted.

The graph snapshots local adapter/bbox code and the external harness, hashes
installed model code and supplied checkpoints, and writes new per-stage caches.
GVHMR verifies fresh pose/feature extraction and rejects default tracking.
The review stage shows source masks, actual boxes, GVHMR with its recovered
camera, v2 with its per-person placement camera, and an original-camera diagnostic.
A separate fixed-world video compares actual motion. Both retain every source
frame and record complete decode/timing checks.

## Post-optimization-only experiments

Run **v2** by default for full source-person reconstruction after reviewed tracking,
GVHMR_BEDLAM2 and same-shot dense Pi3X (owner decision, 2026-09-22). Native global
motion is the comparison baseline. The shorter cached-input command below is for
a post-optimization-only request or valid stage reuse; it does not complete scene
delivery. V2 estimates per-person
world placement; its associated camera transforms are not replacements for the
shared authored-room camera.

The owner-authored v2 implementation and supporting helpers are tracked under
`tools/gvhmr/world_backend/`. Preparation snapshots that packaged backend by
default; no neighboring PromptHMR checkout is required. `--prompt-hmr` remains
an optional legacy override for explicitly supplied experimental backends.
See [human-motion installation](install/human-motion.md) for both environments,
external model libraries and weights, and [backend provenance](../tools/gvhmr/world_backend/README.md)
for the source migration. Licensed body models and run outputs remain external.

Claim the preparation and output paths first. Use the compatible GVHMR Python,
the local GPU, and a new directory. The human runner does not reserve the GPU;
run one GPU job at a time.

```bash
python tools/gvhmr/world_postopt.py prepare \
  --gvhmr-run /path/to/reviewed_gvhmr_run \
  --camera-tracks /path/to/camera_tracks_dense.npz \
  --smplx-model /path/to/SMPLX_NEUTRAL.npz \
  --python /path/to/gvhmr/venv/bin/python \
  --actor-id REVIEWED_ACTOR --gpu 0 \
  --out runs/_tools/world-postopt/EXPERIMENT
bash tools/human_experiments/run.sh plan \
  runs/_tools/world-postopt/EXPERIMENT/experiment.json
bash tools/human_experiments/run.sh execute \
  runs/_tools/world-postopt/EXPERIMENT/experiment.json \
  --run-dir runs/_tools/world-postopt/EXPERIMENT/run
```

For a scene-specific experiment allocate a scene run with `results new-run` and
put preparation beneath it. `_tools` is for cross-scene validation. The generated
graph plugs into the existing [human runner](../tools/human_experiments/RUNNER.md):
`adapt → v2 → measure`. Repeating execute validates hashes before reusing stages.
Changing upstream code requires a new preparation, not edits to its frozen copy.

For this shorter command, preparation snapshots the required upstream files and the integration entry point;
the runner hashes source, body model, predictions, boxes, keypoints, native motion,
provenance, dense camera and camera report. Existing inference is reused, not
reported as newly run SAM3/GVHMR/Pi3X. The adapter requires matching actor IDs,
SAM3 tracking and Pi3X camera provenance, full dense timestamps and
unchanged active-frame indices.
Use trusted model/result files: upstream Torch/joblib caches are executable pickle.

The dense camera NPZ must include `c2w`, `time_seconds`, `camera_source`,
`image_size`, and `observation_c2w`, with its adjacent `.npz.json` report.
Use the upstream `run_pi3x_dense.py` to produce one whole-clip Pi3X forward aligned
to reviewed same-shot cameras. Sparse-camera interpolation is not equivalent.
Review actor identity and exits in the source; metadata alone does not establish
visibility. This adapter currently supports one actor per GVHMR run.

The three primary measured results have different meanings:

| Result | Meaning |
| --- | --- |
| `gvhmr_global` | GVHMR's own global branch including its native postprocessing, in its own gravity-aligned frame. Per-frame Kabsch recovers its camera; the upstream builder asserts rigid-fit residual below 0.1 mm. No trajectory scale is fitted. |
| `lifted_init` | Camera-space GVHMR lifted through dense Pi3X; optimizer initialization, not the postprocessing baseline. |
| `v2` | Corrected per-person placement optimization using global-branch orientation. |

`adapt` also retains `results_native.pkl`, a scale-1 rigid scene-alignment diagnostic,
for compatibility. It is no longer the primary postprocessing baseline. The graph
snapshots `build_gvhmr_global.py` and preserves global orientation from the updated
`build_results.py`. Missing global orientation is an error, not silent fallback to
camera-derived orientation.

The integration removes a single constant floor offset using active frames only
for both lifted and global arms, shifting body and associated camera together.
Inactive padding cannot vote on the floor. Each arm retains its gravity/world
basis; the estimated support plane is not a measured room floor. All variants
preserve full source duration and reviewed actor exits.

Corrected defaults are explicit: 1,000 iterations; `postopt_lr_v2=0.01` for every
parameter group with the upstream cosine schedule; `postopt_orient_source=global`;
contact velocity 1000; Huber contact height 10 with a 2 cm target; acceleration
0.1; velocity prior 0.25; rotation prior 1; translation prior 0.5. Trajectory-scale
optimization is off. V2 does not read the old `postopt_lr` key. No final root filter
or frame-zero realignment is applied.

`measure` writes metrics, series, a comparison table/plot and a decision record.
Compare foot slip, contact height/penetration and root acceleration with
`gvhmr_global`. Verify matching contacts, lifecycle, fixed shape and local pose.
The camera-frame joints are also checked within a 2 mm numerical bound, with the
actual residual saved; float32/axis-angle round trips are not bit-exact.

V2's `people[p]['camera_world']` is a per-person placement transform, **not a shared
metric room camera**. Metrics and source overlays must use it. Camera path length
and original-camera reprojection are recorded as coordinate diagnostics and do
not rank motion quality. The original-camera diagnostic overrides every person's
camera as well as the top-level camera, avoiding a misleading no-op override.

The full graph's review includes a fixed-world skeleton video and source overlays.
The world view uses the same projection and bounds for both arms and the entire
clip; only constant yaw/xz translation is aligned for display. Scale, vertical
placement and temporal motion are unchanged. Every source frame is rendered,
including inactive tails. Skeleton review is not authored-room mesh validation.

Contact labels are model estimates. PMPose is a reconstruction input; upstream
`heldout` fields describe a confidence band, not independent validation. The
`allow_scene_import` flag remains false because this experiment does not perform
common-scene acceptance or select deliveries; it is not a rejection of v2's motion.

The [whole pipeline](REAL2SIM_PIPELINE.md) now implements scene-ground-guided GVHMR
and optional three-frame endpoint Kimodo completion. The historical comparisons below did
not enable either feature; their evidence and conclusions remain scoped to those inputs.

## Corrected implementation validation (2026-09-17)

Reused the validated PMPose-h/GVHMR bodies and dense Pi3X cameras from
`runs/_tools/pmpose-default/20260917/clip{32,42}-r2`. Reran adaptation, construction
of the GVHMR global comparator, corrected v2 and all metrics. No new tracking,
PMPose, feature, GVHMR or Pi3X inference is claimed in this comparison.

| Clip | Arm | Slip mean m/s | Root acceleration RMS m/s² | Mean penetration cm | Max penetration cm |
| --- | --- | ---: | ---: | ---: | ---: |
| 32 | GVHMR global | 0.0380 | 1.734 | 0.296 | 1.67 |
| 32 | corrected v2 | 0.0319 | 2.396 | 0.241 | 1.43 |
| 42 | GVHMR global | 0.0539 | 1.909 | 0.647 | 3.51 |
| 42 | corrected v2 | 0.0460 | 1.919 | 2.134 | 8.14 |

V2 has lower measured sliding in both clips. Clip32 has higher acceleration;
clip42 acceleration is almost equal but penetration is worse. These are tradeoffs,
not evidence that either method universally dominates. Clip42 still has a wrong
identity at the opening and uses the reviewed physical exit245, whereas the
upstream skill table uses234. Its numbers are diagnostic and are not a direct
replication of that five-clip table. The current body-derived floor limitation is
retained, motivating the scene-floor/upright TODO.

Both source and world videos retain260/302 frames at30 Hz and fully decode.
Representative world/source sheets were visually reviewed, including the exit.
Thirteen focused tests pass; package and catalog checks pass. Local evidence:
`runs/_tools/corrected-v2/20260917/{REPORT.md,summary.json,visual_review.json,comparison.html}`.
The first overly strict camera-space tolerance failed on float32/rotation
round-trip error, including the GVHMR comparator; investigated residuals and the
explicit2 mm numerical check are recorded. Source inference evidence stays intact.

## Historical validation with the old implementation (September 17, 2026)

The following results are retained as history. Their v2 configuration and camera
ranking do not describe the corrected implementation.

Fresh complete chains ran on clips 32 and 42, with one fresh ViTPose and feature
extraction per clip and full dense Pi3X inference (47.3 s and 61.3 s respectively).
The local evidence root is `runs/_tools/world-postopt/20260917/REPORT.md`; this
ignored data is not distributed and does not establish validation in a new clone.

| Clip | Native rigid foot slip | Lifted input foot slip | V2 foot slip | Input → v2 camera path |
| --- | ---: | ---: | ---: | ---: |
| 32 | 0.046 m/s | 0.655 m/s | 0.188 m/s | 1.437 → 6.483 m |
| 42 | 0.056 m/s | 1.048 m/s | 0.222 m/s | 0.868 → 9.310 m |

V2 improved the camera-lifted input's foot metric, with substantial camera-path
inflation; the native global comparator retained less sliding but worse source
projection. Clip42 failed initial identity review: backward tracking followed
the woman before the intended older man. Source inspection corrected its old
mask-derived exit234 to physical exit245, retaining the hand still visible at244.
The full-chain clip42 numbers include the uncertain opening and are diagnostic.
Neither candidate replaced an authored-room delivery. Videos retained and fully
decoded 260/302 frames at30 fps. Representative masks, actual bboxes, skeletons and
world-path plots were reviewed; authored-room geometry was not evaluated.

The optional bbox implementation is isolated in `world_tracking.py`, preserving
other work on the default adapter. Against clean-main adapter code it reproduced
clip32's actual boxes exactly. Its accepted-history fallback also prevents long
unsupported runs from restarting warmup when the trailing window loses support.


## Scene-ground and completion inputs

New full-chain preparation requires `--scene-ground prior.json`. The default graph
is `track -> lifecycle -> bodies -> camera -> adapt -> v2 -> measure -> review`.
It retains GVHMR motion, including uncertain occluded frames, and requires no
Kimodo runtime, checkpoint or action prompt.

For an explicit completion experiment, add
`--kimodo-completion --completion-prompt "Observed action description."` to the
preparation command. This inserts `complete` between `bodies` and `camera` and
routes its output into adaptation and review. Supplying a prompt alone does not
enable it. Optional `--completion-context 2|3` defaults to three reliable frames
on each side. Kimodo uses `--kimodo-python`, `--kimodo-upstream` and
`--kimodo-checkpoint` in its shared environment; GVHMR keeps its separate runtime.
No model environment is modified. Only this opt-in graph checks Kimodo assets and
runs completion; unresolved intervals fail its stage. Existing frozen runs retain
their recorded graph, so prepare a new run to use the new default.

Export an existing reviewed floor fit before preparing the graph:

```bash
python -m tools.gvhmr.scene_ground --cameras BUNDLE/cameras.json \
  --fit REVIEWED_ROOM_BASIS.json --out scene_ground.json
```

The exporter accepts the existing `T_bundle_world_to_room` room-basis format or
`floor_alignment.raw_to_room` camera bundle metadata. It does not estimate a new
plane. The exact source video and camera basis must match the inference inputs.
V2 keeps the corrected upstream losses and learning rate, freezes scene upright
with yaw-only placement, and uses the scene plane as its fixed contact support.
The baseline is scene-guided GVHMR; if completion is explicitly enabled, both
arms use the same accepted completion. Raw inference remains separately archived.


Whole-body endpoint conditioning alone did not pass the initial natural clip32
seam check (three-frame context, 2.684 m/s maximum boundary joint-velocity jump).
The completion stage therefore performs a bounded C2 endpoint splice on generated
motion, using only reliable source keys and an SO(3) Kimodo residual. This acts only
inside the invalid interval, before v2. It is not a whole-clip root filter. Raw
Kimodo samples and pre-/post-splice metrics are retained for comparison. Finite-frame
continuity checks and mesh review remain required even with a continuous bridge.


## Export v2 into an existing authored room

Reuse the saved room and its reviewed cameras. Export the actual SMPL-X surface
with the licensed body model in the GVHMR environment:

```bash
python -m tools.gvhmr.world_scene export --results V2/results.pkl \
  --meta ADAPT/adapter_meta.json --model /path/to/SMPLX_NEUTRAL.npz \
  --video SOURCE.mp4 --cameras camera_tracks_dense.npz \
  --room-cameras room_cameras.npz --out RUN/mesh
```

The exporter fits one rigid basis transform from paired camera observations,
including camera orientation, and checks their residuals and common timing. It
inverts the saved scene-ground transform, preserves native body scale, and keeps
inactive tails hidden. It does not derive room placement from the optimizer's
per-person camera. A matching coordinate basis alone does not establish accurate
absolute human placement.

When reviewed source-depth targets are available, optional
`--placement-observations surface_corrected.npz` adds one constant horizontal
translation using only valid training observations. The source video, actor and
room-camera hash must match. Floor height, body dimensions and temporal motion
remain unchanged; these targets are estimates, not surveyed pelvis positions.
`export.json` records exact inputs, transforms and placement observations.

For explicitly requested contact refinement, use the exported mesh with the
[finite-surface contact workflow](HUMAN_TRACKING_AND_TRAJECTORY.md#4-add-requested-contact-constraints).
Review and bind the contact manifest to this exact mesh; preserve source contact
intervals and the existing room geometry. Measure contact gap and penetration,
all-active floor/object intersections, body-shape invariance and motion changes.
Show both the placed v2 baseline and the refined candidate when acceptance fails;
a solver exit or a browser playback check does not establish physical correctness.
The Blender inspection importer and browser exporter can display either candidate
without regenerating the room or changing its materials.

Run the explicit contact stage after binding the reviewed manifests:

```bash
python -m tools.gvhmr.world_scene refine --body RUN/mesh/body_room.npz \
  --contacts reviewed_contacts.json --scene-constraints frozen_scene.json \
  --foot-export RUN/mesh/export.json \
  --out RUN/contact
```

This solves smooth root translation against the declared contact keys and
all-active finite floor/listed convex-object constraints. Shape and articulation
stay fixed. It writes before/after metrics, the corrected mesh and a report with
separate solver, contact and scene-geometry flags; visual review and delivery
selection remain separate. Use the declared training cohorts transparently. A
final fit using all reviewed keys has no heldout-contact validation claim.

The contact solve also retains **GVHMR-predicted foot contacts** when the v2 mesh
export receipt is supplied or `export.json` exists beside the input mesh. It
loads the exact export's static-head logits; no new contact detector or manual
stance annotation runs. Both endpoints must exceed confidence 0.5 for a velocity
constraint, and inactive frames never vote. The four channels are left ankle,
left foot, right ankle and right foot. Sole patches use the same dominant SMPL-X
skinning regions as v2. Support height uses the reconstructed room floor.

The joint scene solve penalizes foot-joint velocity and sole height together with
the existing hand/support and collision objectives. Defaults are velocity weight
1000, height weight 100, and a 2 mm sole target, configurable with
`--foot-velocity-weight`, `--foot-height-weight`, and `--foot-target-gap`. These are
soft constraints; local articulation remains fixed. Reports include identical
predicted-contact samples before/after, slip mean/P90 regression flags, sole gap
and penetration, finite-floor coverage and corresponding-patch velocity. The
predictions are model evidence, not manually verified stance ground truth.

This addresses the clip32 audit: the previous hand/collision-only pass increased
predicted-contact mean speed from 3.20 to 6.13 cm/s because it omitted foot velocity.
Floor nonpenetration and smooth root motion alone do not preserve planted feet.
Any remaining contact conflicts and slip regressions must remain visible in the
candidate report rather than being hidden behind optimizer convergence.

The September 17 joint rerun converged on both reused scenes. Predicted-contact
mean speed changed from 6.13 to 3.21 cm/s in clip32 and from 10.92 to 5.07 cm/s in
clip42; placed-v2 baselines are 3.20 and 4.86 cm/s. The rerun removes most of the
later sliding regression but does not improve every metric: worst floor/object
penetrations are 23.3/17.3 mm and 66.1/61.5 mm, respectively, and reviewed hand or
support limits fail. These are diagnostic candidates, not accepted replacements.
Fixed articulation makes simultaneous contact targets conflict; increasing a foot
weight alone does not solve that limitation. Evidence and before/after previews:
`runs/_tools/gvhmr-foot-scene-contact/20260917/`.


The owner's pending [GPT-6 optimization TODO](REAL2SIM_PIPELINE.md#todo-gpt-6-guided-postprocessing-optimization)
requires improving the optimizer code with additional constraints/objectives and
revised loss weights based on actual scene-mesh review, followed by controlled
before/after motion-quality comparisons. Retaining GVHMR foot contacts is the
first scoped change; broader objective/pose refinement remains pending.

## SAM3 source tracking default

`world_pipeline.py prepare` graphs use SAM3, followed by
the unchanged `world_tracking` trailing-maximum bbox gate, centre/extent gap
interpolation, two five-frame moving averages and 1.2x GVHMR crop. PMPose-h
remains the default detector and consumes the original SAM3 masks with the
smoothed boxes. Reusable tracking inputs must explicitly record SAM3 provenance;
old tracker graphs and unidentified mask archives are not current-route inputs.

Set `SAM3_ROOT`, `SAM3_PYTHON`, `SAM3_CHECKPOINT`, or the corresponding
`--sam3`, `--sam3-python`, `--sam3-checkpoint` flags. SAM3 uses its own interpreter
and requires 1280x720 at 30 Hz. Late prompts run fresh forward and reversed
prefix states, preserving source timestamps. Track provenance records SAM3;
`--person-masks` supplies masks to reconstruction. This route does not establish
tracker superiority.
Mask/crop support is separate from physical exits and pose identity: the real
clip32 pilot showed foreground pose contamination during heavy occlusion.
