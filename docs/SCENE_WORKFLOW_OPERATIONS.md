# Scene workflow operations

Read the relevant section when executing the [scene workflow](SCENE_WORKFLOW.md).
These are existing helper interfaces, not a separate runner or acceptance policy.
Use installed configured runtimes from `kimodo_blender/env.sh`; from the project root,
export `PYTHONDONTWRITEBYTECODE=1` and `PYTHONPATH="$PWD/src"`. Blender executes `bpy`.
Shell variables below stand for task inputs and distinct claimed outputs under
`runs/<scene>/<run>/`. Preserve source scenes and frozen run snapshots.

## Camera and independent static cross-view checks

After Pi3X and reviewed alignment:

```bash
"$PI3X_PYTHON" -m aha3d.workflow.camera \
  --bundle "$ALIGNED_BUNDLE" --config "$CAMERA_CONFIG" --out "$CAMERA_OUTPUT"
"$PI3X_PYTHON" -m aha3d.workflow.crossview \
  --bundle "$ALIGNED_BUNDLE" --config "$STATIC_REGION_CONFIG" --out "$CROSSVIEW_OUTPUT"
"$PI3X_PYTHON" -m aha3d.workflow.preflight \
  --root "$PWD" --config "$ENSEMBLE_CONFIG" --out "$CAMERA_PREFLIGHT"
```

Camera configuration requires rational timing, `[width,height]`,
`continuous_shot_reviewed: true` and an endpoint policy: `error` rejects samples
outside observations; explicit `hold` records their count. `source_start_seconds`
optionally offsets the sample clock. Split shots and map variable-rate footage
explicitly. The helper uses PCHIP translations, rotation SLERP and linear intrinsics,
converting integer-index pixel centers to boundary centers before resizing.

Preflight checks the full timeline, raster, rigid poses, skew and predicted fixed
pixel-aspect residual without Blender; future person-cache paths are allowed.
It writes `camera.json` and exits 2 above the configured tolerance. Investigate
conventions/scale and document any deliberate tolerance change. Assembly repeats
preflight and separately compares all evaluated Blender cameras with the cache.
These checks establish representation consistency, not source-camera accuracy.

Static-region configuration uses native source frame IDs, excluded processed-pixel
rectangles `[x0,y0,x1,y1]`, frame pairs, descriptor ratio and confidence thresholds.
Exclude moving people, reflections, graphics and unreliable regions. ORB matches
are independent of predicted 3D; report per-pair support/error because repeated
patterns and weak turns can invalidate a pooled median. Review static floor patches
separately from camera-up or higher surfaces. Keep `world_transform` rigid; calibrated
uniform scale goes through the measurement interface for points and camera translations
together. Assumed camera height does not supply measured scale.

## Parameterized room and multi-person assembly

Ensemble configuration contains timing, raster, optional camera cache, unique integer
person IDs/labels, cache paths, constant scale/yaw, initial pelvis `anchor_xy`, optional
`z_offset` or `ground_z`, preview frames and validation policy. Paths resolve against
`--root`; ID `1` maps to `Person_001_Body`. Validation rejects unknown fields,
duplicate IDs, incompatible timing and unlisted existing people during extraction.
Source-human scale and correction permissions follow [motion controls](MOTION_CONTROLS.md),
regardless of the generic placement fields exposed by this helper.

- `z_offset`: one constant world-metre Z translation after scale; omit both vertical
  fields to preserve original placement. Apply it once rather than also shifting caches.
- `ground_z`: per-frame translation of the lowest vertex to a chosen height, only for
  a reviewed flat-support case. It is unsuitable for stairs, jumps or precise contacts.
- The two vertical fields are mutually exclusive. Uniform stature scaling is not
  fitted pediatric morphology; neither a scale nor a floor offset certifies gait.

```bash
"$BLENDER" -b "$SAVED_SOURCE" -t 8 --python-exit-code 1 \
  --python src/aha3d/blender/ensemble.py -- --root "$PWD" \
  --config "$ENSEMBLE_CONFIG" --stage room --out "$ROOM_OUTPUT" --preview
"$BLENDER" -b "$ROOM_OUTPUT/scene.blend" -t 8 --python-exit-code 1 \
  --python src/aha3d/blender/ensemble.py -- --root "$PWD" \
  --config "$ENSEMBLE_CONFIG" --stage assemble --out "$ENSEMBLE_OUTPUT" --preview
"$BLENDER" -b "$ENSEMBLE_OUTPUT/scene.blend" -t 8 --python-exit-code 1 \
  --python src/aha3d/blender/ensemble.py -- --root "$PWD" \
  --config "$ENSEMBLE_CONFIG" --stage verify --assembly "$ENSEMBLE_OUTPUT" \
  --out "$ENSEMBLE_CHECKS"
```

`room` removes only declared people from an existing scene, preserving its geometry
and materials. A newly authored reviewed room can go directly to `assemble`.
Review furniture footprints, facing, aisles and actual support before committing
motion. Keep inspection-only cameras/visibility in separate outputs.

Blender cannot animate render pixel aspect. The importer uses median `fx/fy`, keys
lens/principal-point shifts and checks every evaluated camera; default intrinsic
tolerance is 2 px. Ratios below one use `(x,y)=(1/ratio,1)` to respect Blender's lower
bound of one per axis. Resolve convention errors before accepting approximation.
Each preview process configures its GPU devices.

Reopened verification covers every person's topology, stable vertex IDs, geometry,
placement, framing, floor/intersections and camera poses/intrinsics at output frames.
Sampling is explicit (`all` for changed paths). Self-collision, containment,
between-frame clearance, foot sliding and exact hand contacts still require their
own evidence; inspect actual meshes and supports.

## Final render and comparison

After preview review, point a normal recipe at `ENSEMBLE_OUTPUT/scene.blend`, retain
baked bodies and follow [render execution](RENDER_EXECUTION.md). Use the pipeline
for rendering, resume and full decode. Repeat ensemble `verify` on the final saved
pipeline scene to cover all actors; the pipeline tracking interface selects one body.

```bash
bash tools/indoor run "$SCENE_ID" --recipe "$FINAL_RECIPE"
"$PI3X_PYTHON" -m aha3d.workflow.compare \
  --reference "$SOURCE_VIDEO" --run "$FINAL_RUN" --out "$COMPARISON_OUTPUT"
```

`compare --run` resolves the recorded video/frozen timing and refuses a live execution
lock. For external videos, use `--render VIDEO --config TIMING_CONFIG` instead.
The forms are mutually exclusive. Resolve source aliases from `scene.json.reference`
and verify the actual source path when updating catalogs.

Comparison uses FFmpeg with configurable tiles (default 640x360), strict full decode
and timing checks. Inputs require matching constant-rate timelines starting at their
respective first frames; cuts/VFR need explicit alignment. Retry a failed comparison
into a new output after fixing inputs; completed rendering remains reusable.
Comparison does not record visual acceptance or replace all-person verification.
The final baked `.blend` supports playback without Kimodo or a body add-on.

## Submission preflight and full-clip preview acceptance

`plan`, `prepare`, `run` and `batch` share recipe/input preflight. Recipe names are
identifiers, not JSON paths. Optional SAM selection/camera inputs must be paired:

```bash
bash tools/indoor preflight "$SCENE_ID" --recipe "$PERSON_RECIPE" --motion-only \
  --sam-selection "$SAM_SELECTION" --sam-cameras "$SAM_CAMERAS"
```

Preflight checks schema, prompt segmentation/10-second limits, native
`int(duration * 30)` frame counts, constraint bounds, conflicting root/heading targets,
dependencies and optional SAM raster/endpoint bounds. Kimodo `global_root_heading`
is `(cos(angle), sin(angle))` (+Z forward `[1,0]`); SAM `facing_xz` is an XZ vector
(+Z `[0,1]`). Inspect processed full-body overlays and mesh facing for identity,
visibility and coordinate meaning. Preflight does not load models, open Blender or
inspect linked scene assets; saved-scene and source-fidelity checks are separate.

New video runs stop at `preview_required` after assembly/verification. Motion-only
and still-image runs are exempt; historical manifests without `preview_policy`
retain their behavior. For a prepared video run:

```bash
bash tools/indoor preview runs/SCENE/RUN --backend gpu --max-width 320 --preview-fps 5
bash tools/indoor accept-preview runs/SCENE/RUN --evidence /absolute/path/review.json
bash tools/indoor submit runs/SCENE/RUN
```

`submit` resumes final stages in a background worker; `execute` is the foreground
alternative. Existing generation, skinning, assembly and verification are reused.
GPU execution requires visible usable devices; CPU-supported verify/resample/video
and `preview --backend cpu` remain available. Assembly currently configures GPU
rendering. CPU previews use headless Cycles; GPU previews use the configured raster
engine or explicit Cycles GPU. The older `run --preview` quality flag is not this gate.

### Sampling and review

Default preview is **5 FPS across the full source duration**, including endpoints.
Use 1 FPS for coarse inspection, higher rates for fast events, or `source` when every
frame is needed. `sampling.json` records exact scene-frame indices/timestamps;
consecutive encoded indices and rational presentation FPS preserve clip duration.
The final scene timing/camera remain unchanged. Old snapshots without sampling need
a new run or explicit `--preview-fps source`.

Each `RUN/previews/<id>/` contains the sampled MP4, original recipe-resolution
start/middle/end stills and an all-frame evaluated mesh/contact report for every
tagged `person_id`. Reduced previews preserve saved pixel aspect and use an integer
raster divisor, so unusual rasters may exceed `--max-width`. Encoding validates
sampled count/rate/duration; final delivery still needs every requested frame decoded.

Inspect the entire sampled preview, full-resolution keyframes against source frames,
and all-person placement/contact evidence. A montage supplements this inspection.
Numerical mesh checks cover integer frames independently of preview rate; `collisions:
error` rejects intersections while `off`/`report` keep diagnostic scope. Untagged
people, identity, inter-frame clearance and self-collision are not established by
this report. Preserve intended supports and report approximation limits.

Record actual review in JSON with absolute paths. This schematic example abbreviates
frame lists; use the exact full `sampling.json.scene_frames`, real keyframe numbers
from `render_report.json` and every enumerated person ID:

```json
{
  "reviewer": "scene-agent",
  "preview_reviewed": true,
  "reviewed_scene_frames": [1, 13, 25],
  "sampling_limitations": "Describe between-sample risks and higher-rate follow-up.",
  "note": "Describe observed camera, timing, framing and approximations.",
  "source_video": "/absolute/path/to/registered/source.mp4",
  "source_comparisons": [
    {"frame": 1, "note": "Opening source/render proportions."},
    {"frame": 450, "note": "Midpoint layout and action."},
    {"frame": 900, "note": "Endpoint layout and action."}
  ],
  "people": [
    {"id": 1, "placement_contact_note": "Observed support/contact and approximations."}
  ],
  "people_report_limitations": "Integer sampling and intended or unresolved overlaps."
}
```

Sampled acceptance requires `preview_reviewed`, the exact `reviewed_scene_frames`
and `sampling_limitations`. Original-rate previews retain `whole_clip_reviewed`;
that legacy field alone cannot accept sampled previews. Use an empty `people` list
for no people. The default `people_report.json`, or an explicitly stronger report,
must match the assembled scene SHA256, all person IDs and complete sampled frames.

### Source provenance and resume

Review the preview's lossless `source_keyframes/` PNGs; receipts record source video
hash, zero-based decoded indices and image hashes. Default source/output mapping
requires matching frame count, FPS and duration. Trimmed/retimed scenes require
`preview --source-frames INDEX_FIRST INDEX_MIDDLE INDEX_LAST`; align these with the
output keyframes in `render_report.json`. An external `source_image` must hash-match
the receipt-extracted PNG.

Text-authored scenes without a registered reference use `no_reference_rationale`
and `render_keyframes` (frame/note entries) instead of `source_video` and
`source_comparisons`. Frozen reference provenance determines the requirement.
Acceptance binds scene, recipe/runtime/code/input identities, preview artifacts and
review evidence. Changed source, timing, camera, output or evidence needs applicable
regeneration/review in a new immutable run. Keep evidence durable; preview acceptance
records the agent's technical review, while final selection uses the task's
[acceptance contract](WORKFLOW_ACCEPTANCE.md).
