# Generate and adapt body motion

Use scene recipes through `bash tools/indoor`; see project-root `docs/PIPELINE.md`
for the shared stages and `kimodo_blender/RUNTIME.md` for model provenance.
Use the configured GPU runtime for inference/skinning. Capacity was validated on
96 GB; consult the runtime's 24 GB options when needed rather than inferring capacity.
The human branch is body-only SMPL-X without facial expression animation; G1 uses
its native checkpoint, skeleton and rigid-link meshes.

## Motion configuration

Select controls using the [motion control policy](../../../../docs/MOTION_CONTROLS.md).
For source-video actions the accepted priority is root > hands >= feet > text-only.
Establish the root route and review sparse source hand/foot observations before
the final generation; text supplies action semantics. Record unavailable or rejected references
instead of silently delivering a text-only baseline. Required contacts use native
end-effector constraints; the foot compiler still needs its experimental review.
Text-only tasks without source people retain a text entry point. Do not route new
work through retired post-generation arm IK/facing corrections.

Start from `examples/new_scene/recipes/walk.json`, then copy it into a new scene workspace. Set the
prompt, duration segments, seed and reviewed native constraints for reference-guided motion. Timing
specifies the final frame count and rational FPS. Generate and inspect on the local GPU.

The generic example is text-only and contains no fixed route. Add native root
constraints when the requested action requires a path; choose indices and points
for its duration and scene coordinates. Root-path constraints are not furniture-aware
navigation, and different seeds/prompts may change limb clearance.

Use `body.mode: native` with a native Kimodo NPZ to resample an existing motion,
`cache` to import already-skinned geometry, or `keep` to preserve a saved body.
New motion is quaternion-resampled before skinning, preserving all 300 shape
coefficients and the deployed SMPL-X asset's pose corrections. The skin stage also
saves an editable rig `.blend`. Optional smoothing must be explicit in the recipe.

### Duration and native calls

The deployed model supports at most **10 seconds per prompt** (300 native frames
at 30 fps). Longer clips use multiple native segments. Direct Python calls take
complete prompt strings and matching integer frame counts; the recipe wrapper
requires one duration per period-separated prompt and validates the limit before
generation. Check the actual converted calls when using another wrapper.

Choose splits by action changes and observed drift/joins, without a universal
five-second rule. Hand/foot keys can also encode root height and heading; reconcile
them with the route. Inspect segment boundaries in the mesh and source-camera view.
See [call validation](call-and-validation.md) for native frames, implicit constraints
and continuation handling.

## Placement and appearance

Body offsets are metres in Blender Z-up; yaw rotates about Z. The example uses
`anchor: first_pelvis_xy` to place the initial pelvis at the specified horizontal
coordinates. The default `cache_origin` translates the original cache origin.

`ground_clearance` moves all vertices and joints vertically each frame until the
lowest vertex reaches the chosen height. The corrected cache records this offset.
This flat-floor heuristic is unsuitable for jumps, stairs and intended airborne
motion without action-specific handling.

Assembly recipes may select a camera, hide named collections and park cabinet
controls closed. Otherwise existing source animation and camera remain in place.
New `.blend` outputs are separate from their sources. Blender-specific extensions
should import `aha3d.blender.body.import_cache` and `export_tracking`.

For material or camera variants, use `--reuse-run` with the same motion settings;
matching generation/resampling/skinning stages are reused. Render mode `preserve`
requires a saved Cycles scene. Workbench sources need an explicit `clay` or
`material` conversion with appropriate color settings.

## Validation

The example is 120 frames at 24 fps: frame 1 is time zero, frame 120 is 119/24 s,
and playback ends at five seconds. The shared encoder handles recipe timing and
verifies frame count, duration, dimensions and full decoding.

For changed paths, use validation sampling `all`, review floor contact, framing
and actual furniture clearance, then inspect representative rendered frames.
Triangle intersections at integer frames do not establish inter-frame clearance,
self-collision avoidance, absence of foot sliding or temporal smoothness.
Synthetic tracks are exported from the final evaluated scene; their frustum flags
do not test occlusion. See [tracking schema](tracking.md).

The scripts under `kimodo_blender/` remain legacy demonstrations and provenance;
new reusable behavior belongs in `src/aha3d/`.

## Independent actor stages

Use `--motion-only` on `plan`, `prepare` or `run` with a `generate`/`native` recipe
when only generation, rotation resampling and skinning are needed. The prospective
source `.blend` path is retained in recipe metadata but need not exist and is not
copied. Status remains `staged`; it does not certify room contact or video delivery.
Resume preserves this scope. Use compatible caches in a later scene run.

```bash
bash tools/indoor plan "$SCENE_ID" --recipe "$PERSON_RECIPE" --motion-only
bash tools/indoor run "$SCENE_ID" --recipe "$PERSON_RECIPE" --motion-only
```

For uncertain travel or sustained support, inspect native output before skinning:

```bash
bash tools/indoor prepare "$SCENE_ID" --recipe "$PERSON_RECIPE" --motion-only --run-id "$MOTION_ID"
bash tools/indoor execute "$MOTION_RUN" --until motion
"$KIMODO_ENV/bin/python" -m aha3d.motion.diagnostics \
  --motion "$MOTION_RUN/stages/motion/motion.npz" \
  --recipe "$MOTION_RUN/snapshot/recipe.json" --out "$NATIVE_DIAGNOSTICS"
bash tools/indoor execute "$MOTION_RUN" --until skin
```

`MOTION_RUN` is `runs/$SCENE_ID/$MOTION_ID`; claim it and the diagnostic output.
For `native` mode inspect the supplied file directly; there is no generation stage.
Diagnostics report Y-up root travel, displacement, height, steps and segment joins.
Optional `--max-travel-m`, `--max-root-step-m`, `--min-root-height-m` and
`--max-height-range-m` are action-specific limits (violations write evidence and
exit 2). Native FPS follows recipe `source_fps` or defaults to 30, with an override.
Review diagnostics before the skin command; support/facing still need mesh review.
`submit` can resume prepared motion-only runs; `batch` does not expose this flag.

The [ensemble workflow](../../../../docs/SCENE_WORKFLOW.md#parameterized-room-and-multi-person-assembly)
provides stable per-person IDs, independent caches, explicit uniform stature scale,
constant placement and optional flat-floor grounding. It verifies all people in
the reopened final scene. The original shared pipeline tracks one selected body;
its successful report alone is insufficient for an ensemble.

A root path does not specify forward/backward walking. Review mesh facing at
approach, pause and departure. When an essential turn needs native heading control,
inspect the installed constraint schema rather than adding hand/full-body keys
solely for facing. Changes to an intended route require task-specific judgment;
collision diagnostics are not an automatic motion-correction instruction.

## Mesh previews and contact experiments

`python -m aha3d.motion.preview --motion ... --out ...` defaults to actual human
mesh, resampling rotations before skinning and following pelvis XY for display
while preserving heading/vertical motion. Use `--baseline` for comparison;
`--representation skeleton` is an explicit diagnostic option. Claim the video and
its sibling assets directory. A room-free preview does not establish furniture contact.
Final scene playback uses baked shape keys; keep the separate rig when skeletal
editing is useful. Regenerate synthetic tracks after pose, placement or camera edits.

For authorized generated contact edits, `tools/kimodo/contact_edit.py` accepts native
joint targets, mandatory text, optional warm-start motion and a hashed clean-prediction
callback. Inspect decoded mesh contact, visible-part preservation and transitions;
conditioning alone does not guarantee exact contact. Source-contact repair stays under
the current [motion policy](../../../../docs/MOTION_CONTROLS.md).

## Experimental source-assisted legs

The SAM skill now exposes [reviewed sparse leg directions](../../sam3d-motion-reference/references/legs.md)
through native foot constraints. Follow its original-camera, visibility and
identity guards. An ankle/foot key also carries root height, smooth root and heading.
Keep a matched baseline and check the actual skinned sole before any grounding;
better keyed limb directions alone do not establish better gait. Use withheld
source times and a shared constant floor policy for comparison. Promoting a foot
variant requires new ensemble/furniture/camera validation.
