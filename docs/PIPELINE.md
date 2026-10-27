# Shared scene pipeline

Read the [complete updated real2sim pipeline](REAL2SIM_PIPELINE.md) for source,
room, camera, PMPose/GVHMR, v2, planned Kimodo gap completion and delivery routing.

Use `bash tools/indoor` from the project root after [setup](SETUP.md). It uses
the installed project Python environment selected by `KIMODO_ENV`; recipes also
require their configured stage runtimes. This CLI executes authored recipes. For
first-video geometry inference, use the [geometry reference guide](PI3X_GEOMETRY_REFERENCE.md). Execution rules live in the
[runtime policy](../MACHINE.md). Claim recipe, code and run paths through the
[coordination protocol](COORDINATION.md) before writing them.

The deployed preflight, full-duration preview and layout-review interfaces are
part of the mainline. See [scene workflow](SCENE_WORKFLOW.md#submission-preflight-and-full-clip-preview-acceptance)
and [layout inspection](LAYOUT_INSPECTION.md#pipeline-and-gvhmr-review-gates).

## Commands

For new high-level scene requests, begin with `bash tools/indoor intake` and the
[requirements guide](SCENE_REQUESTS.md). It collects missing scope and emits a
durable authoring plan without loading runtime paths or starting work. It is
separate from `plan`, which reads an already authored executable recipe. Existing
recipe run/resume commands retain their current behavior.

```bash
bash tools/indoor scenes
bash tools/indoor plan living_room_kitchen_g0025 --recipe walk_generate
bash tools/indoor run living_room_kitchen_g0025 --recipe walk_generate --preview
bash tools/indoor status --live
```

`plan` reads configuration without loading Blender or models. `run` prepares
and executes the run in the foreground on the GPU; use `batch` (below) to start a
detached background worker instead. Code and source
files are snapshotted when preparation starts. `--preview` limits resolution to 640 × 360 and samples
to eight; frame count and duration remain unchanged.

```bash
bash tools/indoor batch living_room_kitchen_g0025:walk_replay \
  minimal_living_g0070:v4_replay pool_lounge_0a8d46c9:revised_stills \
  --preview
```

Each pair receives its own run directory. One background worker runs the tasks
sequentially on the single GPU, continuing past failures. Batch recipes must use
the same runtime profile. The batch folder `runs/batches/<id>/` holds `job.sh`,
`batch.log`, per-task `<i>.log` and `<i>.log.exit` (exit code), and
`submission.json` (`job_id`, `pid`, `runs`). A started batch is not a completed
result; `status --live` reports whether the worker pid is alive and each task's
exit code. Inspect those and the run reports.

## Recipes and stages

For requested source-motion comparisons, the optional
[full world-postopt pipeline](WORLD_POSTOPT.md) runs fresh SAM3 tracking,
bbox/lifecycle processing, PMPose/GVHMR, dense Pi3X, adaptation, v2 and review
through the resumable human-experiment runner. It compares GVHMR global postprocessing, camera-lifted initialization and corrected
v2 using foot slip, contact height and root acceleration. Per-person placement
camera paths are coordinate diagnostics, not motion-quality criteria.
It does not replace the default native-motion route or automatically import
experimental cameras into authored rooms. A shorter postopt-only command remains
available for explicitly requested cached-input experiments.

Each scene has `scene.json`, `STATE.md` and `recipes/<name>.json`. Recipes select
a saved source `.blend`, timing, body inputs/placement, assembly settings, render
settings and validation policy. Unknown fields and inconsistent timing fail.
Source scenes must localize linked libraries and pack file textures before use.
The [initial recipes](../scenes/README.md) are examples; the
[configuration validator](../src/aha3d/config.py) defines accepted fields.

| Body mode | Stages before assembly |
| --- | --- |
| `keep` | Preserve the source body's baked animation; optional cache supplies joints |
| `cache` | Import a supplied skinned body cache |
| `native` | Resample native rotations, then skin with the deployed SMPL-X add-on |
| `generate` | Generate Kimodo motion, resample rotations, then skin |

All modes assemble a new scene, export calibration/tracks where applicable,
reopen the saved scene to verify it, and render. Video recipes also encode and
fully decode the complete frame sequence.

Timing uses explicit `frames`, rational `fps` and optional `duration_seconds`.
For example, 120 frames at `24` last five seconds; 450 at `30000/1001` last
15.015 seconds. New motion uses rotation SLERP before skinning. Existing animated
sources must already match the requested timing. `cache_timing: match_scene_frames`
is an explicit adoption of an existing baked scene's playback rate, used by the
archived g0070 replay.

Prefer authoring only `frames` and rational `fps` (plus optional `start`). An
explicit `duration_seconds` is checked for exact equality, so a rounded decimal
for 208 frames at `24000/1001` is invalid. The runner's frozen recipe includes
derived duration/end fields; downstream tools should use its count and cadence,
not pass normalized timing back as a new authoring config.

Render mode `preserve` requires an existing Cycles scene. For a Workbench source,
select `clay` or `material`; conversion defaults to AgX. Recipes can explicitly
set `view_transform`, `look` and `exposure`. The pool recipe uses its original
preview's AgX / -0.30 exposure settings.

No smoothing is applied unless the recipe asks for it. `anchor: first_pelvis_xy`
places the initial pelvis at the requested horizontal offset; the default
`cache_origin` translates the existing cache origin. `ground_clearance` applies
per-frame vertical correction and records that correction in the output cache.

## Run contents and reproducibility

```text
runs/<scene>/<run-id>/
  run.json                       status, fingerprints, provenance and review
  snapshot/                      frozen Python code, recipe and runtime profile
  inputs/                        copied source scene and supplied motion inputs
  stages/motion/                 generated native motion, when requested
  stages/resample/                target-timed AMASS motion and timing report
  stages/skin/                    skinned body cache and editable rig copy
  stages/assemble/                baked scene, body cache, tracks and calibration
  stages/verify/validation.json   saved-scene verification
  stages/render/                  PNG frames, receipts and completeness report
  stages/video/video.mp4          canonical encoded video
  stages/video/validation.json    full decode/timing report
  logs/                          one log per stage attempt
```

Installed runtimes and
licensed models stay in local storage outside Git. Do not mutate installed dependencies while a run or batch is active.

Saved output scenes pack file textures and reject linked libraries. Body playback
uses baked shape keys. Resuming a run requires its frozen directory and installed
runtime; sharing the final `.blend` for playback does not require Kimodo.

## Resume and reuse

Resume a prepared or interrupted run in a new background worker:

```bash
bash tools/indoor submit runs/SCENE/RUN_ID
```

Execution verifies frozen inputs and skips completed stages whose fingerprints
and artifacts still match. Rendering reuses only frames that still match the recorded outputs. Changed or corrupt frames are rendered again. Logs retain every attempt.
An execution lock prevents concurrent writers to one run. Locks have no automatic
expiry; after a killed process, confirm its recorded pid is no longer alive
before removing that exact stale lock. Do not edit a run's snapshot to repair code: create a new run.

For camera/material variants, create another recipe and reuse compatible motion:

```bash
bash tools/indoor run SCENE --recipe VARIANT --reuse-run runs/SCENE/PREVIOUS_RUN
```

Camera/render changes
leave motion fingerprints unchanged; seed, timing, smoothing, code or runtime
changes invalidate affected reuse. Reuse requires the source run to be idle.

In the foreground, `prepare SCENE --recipe NAME --run-id ID` creates a snapshot
without executing it. `execute RUN --until STAGE` stops after a named stage.
`execute RUN --frame-budget 2` deliberately interrupts rendering after two new
frames to test resume. These are diagnostic controls, not validated completion.

## Validation and review

Checks cover finite geometry, cache/topology correspondence, camera calibration,
requested timing, saved-scene playback, sampled floor/framing/intersections,
complete rendering, video frame count/duration and full decoding. Collision
policy is `report`, `error` or `off`; sampling defaults to five frames, with `all`
available for changed motion paths. Reports state their actual sampling scope.
Triangle intersections do not certify containment, self-collision, inter-frame
clearance or foot sliding. Track `in_frame` flags test the camera frustum, not
occlusion. Replays without a joint cache export vertex tracks with zero joints.

`validated` means the configured automated checks passed. Inspect representative
rendered frames before recording a review:

```bash
bash tools/indoor review runs/SCENE/RUN_ID --reviewer NAME \
  --frames 1 60 120 --note 'Concrete observations from these rendered frames.'
bash tools/indoor promote runs/SCENE/RUN_ID --name selected
```

Both commands verify artifacts locally. Promotion requires a recorded review and
writes `deliveries/<scene>/selected.json`; it does not copy or overwrite sources.
Report-only warnings remain review obligations. Historical replay recipes do not
become accepted scene revisions simply by passing technical checks.

## Adding a scene or asset

Create a unique scene ID, add `scene.json` and `STATE.md`, and write recipes that
reference saved sources in `scenes/<scene>/blender/`, with original footage in
that scene's `references/`. Put scene-specific choices in recipes and reusable
behavior in `src/aha3d/`; avoid another copied runner per scene. Register
portable furniture/material libraries under [assets](../assets/README.md).

Existing scene builders remain compatibility and provenance references. Historical scene variants now live under
`scenes/<scene>/blender/`. [Path migrations](../configs/path_migrations.json) resolve
old input paths through the current path resolver and task-claim helper. Frozen
run inputs, code and reports retain their original bytes; their own snapshots
remain the execution source when resumed.

## Checks

Run `python -m unittest discover -s tests -v` with the Kimodo interpreter after
`source kimodo_blender/env.sh`, as in the [runtime policy](../MACHINE.md), and
`python tools/check_docs.py`. Unit tests use tiny synthetic
inputs; real GPU validation evidence belongs in the
refactor handoff (historical or external input; omitted from this bundle).

## Room-free motion and complete scene coordination

`plan`, `prepare` and `run` accept `--motion-only` for `generate` or `native` body
modes. Preparation omits the source `.blend` entirely; its prospective recipe path
need not exist. Only motion/resample/skin stages execute, with ordinary snapshots,
locks and fingerprints. Even after successful skinning the run is `staged`, not
`validated`, because scene/video checks have not run. Resume keeps that scope.
The flag is not currently exposed by `batch`; `submit` can resume a prepared run.

Use the [scene workflow](SCENE_WORKFLOW.md) to coordinate Pi3X, white-room/camera
review, independently generated people and final delivery. Its parameterized
ensemble verifier supplements this pipeline's single-body validation/tracking
interface with checks for every declared person in the final saved scene.
The workflow also provides CPU camera preflight, native root diagnostics before
skinning, constant ensemble `z_offset`, and `workflow.compare --run RUN` to resolve
the recorded video and frozen timing without task-local filename conventions.


## Scene-ground GVHMR and occlusion completion

The [world-motion graph](WORLD_POSTOPT.md) now takes an accepted scene floor/upright
before GVHMR, then proceeds directly to camera lifting and v2. Kimodo completion
is skipped by default following the owner's visual quality feedback. Low-evidence
frames retain GVHMR estimates; this does not establish their accuracy.

Explicit `--kimodo-completion --completion-prompt "Observed action description."`
adds a `complete` stage. In this experiment, PMPose frames with fewer than five
reliable mask-supported keypoints are invalidated. Kimodo uses the preceding and
following three reliable frames (two minimum), with no invalid GVHMR poses or
roots as conditions. Unresolved/rejected intervals stop this opt-in graph;
completion evidence and diagnostics stay in the failed stage's `result/work`.
Existing immutable runs keep their original stage definitions. Prepare a new run
to use the new default; never modify an old snapshot to change its algorithm.
