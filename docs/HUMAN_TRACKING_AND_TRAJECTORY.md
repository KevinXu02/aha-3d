# Human tracking, trajectory and contact constraints

## Working-mode policy update — September 16, 2026

The owner separated normal use from debug experiments. The authoritative mode
contract is [Human motion: working and debug modes](HUMAN_PIPELINE_GUIDE.md).
Working mode processes all people in the full supplied video, uses tracker validity
to hide missing tracks without a GPT-6 exit checkpoint, filters ViTPose observations
by a confidence threshold, and proceeds from rigid placement to contact-position
correction. It has no mandatory raw-motion diagnostic stage, manual observation
intervals, or agent decision about whether to apply extra trajectory correction.
Independent trajectory scaling/smoothing and Kimodo are outside the current scope.

The reviewed single-actor interfaces below remain implemented **debug interfaces**;
this documentation change does not make them an automatic all-person pipeline.
In particular, their reviewed terminal-prefix lifecycle and required observation
review/quality files still need working-mode integration. Preserve historical run
provenance; do not fabricate reviews to satisfy a legacy schema.

## Existing debug workflow and interfaces

Use this guide for source-person reconstruction with SAM3 and GVHMR in an
authored room. The user requested the lifecycle and correction capabilities below
on 2026-09-13. Room geometry continues through Pi3X references and Blender asset
authoring. This guide does not turn raw Pi3X meshes into the delivered room.

The objective for contact work is **contact > source action/style > image
alignment**. Retain walking, waving, sitting and contact timing; use reprojection
as a diagnostic. Keep original estimates and rejected candidates as separate
artifacts.

## 1. Select an actor and establish its track lifetime

Review the source before initializing SAM3. Choose the actual
requested person, including when initially distant or partially hidden. Save the
original masks, actor ID, initialization frame/box and full-video SHA256. Reviewed
mask suppression is a separate derivative with reasons; never overwrite raw masks.

Supply SAM3 masks before fresh PMPose (or explicitly selected ViTPose), image
features and motion inference. `tools/gvhmr/person_tracking.py` uses the pinned upstream crop
conversion and separates observed mask support from interpolated crops.
`tools/gvhmr/person_reconstruct.py` records actual inputs, preprocessing hashes
and extraction counters. A mask overlay on an existing motion cache is not a rerun.

Use three distinct concepts:

| State | Motion treatment |
| --- | --- |
| Visible or partially cropped actor | Continue the active track; use only supported body parts as observations |
| Interior occlusion or initial overlap | Keep identity/lifetime separate from missing observation support; interpolation is not observed motion |
| Confirmed fully out of frame | Stop this track segment and motion inference; hide its body from the first fully absent frame |

An empty mask alone cannot distinguish exit from occlusion or tracker failure.
Review edge approach and neighboring source frames. Automatic edge/absence cues
are proposals, not permission to terminate an internally occluded person. A tiny
edge sliver can be unusable for pose cropping while the person is still present.
If the person returns, create a new reviewed segment with explicit identity and
source timing rather than silently reviving a terminated segment.

The cache contract is `track_active: bool[T]`. One segment has an active prefix
and terminal inactive tail; legacy caches without this field mean all active.
Keep source frame indices, exact timestamps, actor/video binding and the reviewed
exit event. Full-scene storage may repeat the last active pose after exit, but
those rows are explicitly **inactive padding, not estimated or generated motion**.
Fresh motion inference processes only the active prefix. Do not send the exited
tail to Kimodo for completion. Preserve scene, video and camera duration.

The current launcher expects an exact **30Hz, zero-origin normalized source**;
retain the original video and normalization mapping. From the repository root,
using the dedicated GVHMR Python and configured GPU:

```bash
python -m tools.gvhmr.person_reconstruct \
  --video /absolute/normalized_source.mp4 --actor-id reviewed_actor \
  --person-masks /absolute/masks.npz \
  --track-lifecycle /absolute/lifecycle_review.json \
  --pi3x-bundle /absolute/same_shot_camera_bundle --gpu 0 \
  --output /absolute/new_gvhmr_run
```

Select the GPU for the actual machine. The lifecycle JSON requires `schema_version:1`,
`source_video_sha256`, `actor_id`, exact full `time_seconds`, `image_size:[W,H]`,
`terminal_exit_frame` (integer first absent frame, or null), `reviewer`, `reason`
and supporting `evidence_frames` before/after an exit. Review is performed by the
agent from source evidence; it does not require another user approval.

`track_lifecycle.edge_exit_proposal(masks)` suggests an edge/terminal-absence
candidate. Source review determines the active inference prefix. Masks NPZ must
contain `masks:bool[T,H,W]`, `image_size:[W,H]`, exact `time_seconds`,
`source_video_sha256`, `actor_id` and explicit SAM3 tracker provenance.
`prepare_gvhmr_boxes` retains full evidence but creates model crops only for the
active prefix. The launcher records the losslessly verified inference prefix,
actual model inputs and full source/lifecycle mapping. For a matched control,
replace `--person-masks` with `--baseline-track-id` for the same reviewed actor.

## 2. Establish one room basis and depth observation set

Keep SMPL-X body dimensions at native nominal metric scale. Record the rigid
native-to-room rotation/placement, the Pi3X-to-room transform and their provenance.
Pi3X depth and camera translation already contain its predicted metric factor;
do not apply it twice. Absolute metric scale remains approximate without a
separate physical scale observation.

For trajectory anchors, use reviewed ViTPose COCO hips, or shoulders when hips
are cropped. Select the target-person mask interior and a small robust depth
neighborhood. Reject low-confidence/edge pixels, broad or mixed depth clusters,
wrong-person landmarks and anatomically inconsistent projections. Confidence is
not proof of visibility. Do not query an occluder's depth for the hidden pelvis.

Keep pixel grids and camera conventions explicit:

- Map source UV to the Pi3X raster using the recorded resize convention, including
  the half-pixel offset where applicable. Resize masks with nearest-neighbor sampling.
- `local_points[..., 2]` is axial camera-Z depth, not distance along a unit ray.
  Backproject with the matching processed-image intrinsics.
- Use OpenCV camera-to-world (right/down/forward); invert a world-to-camera input
  before use. Apply the recorded room transform once to both points and cameras.
- Compare source-shaped COCO mesh regressors with ViTPose anatomy. A skeleton's
  first 17 joints are not automatically COCO17.

Visible clothing depth is a body surface, not an internal pelvis measurement.
Compare a raw-surface baseline with a surface-to-root corrected target. The latter
uses the source-shaped camera-space body and the front ray/mesh intersection to
estimate the surface-to-pelvis offset; it inherits body/camera error. Preserve raw
and corrected targets, rejection reasons and exact source rows.

Freeze training/held-out rows before filtering invalid observations. Fit only
supported, active training rows. Binding must include video SHA256, actor ID,
room-basis hash, exact source-body hash and timestamps. Held-out depths evaluate
agreement with estimated observations, not independent ground truth. If the room
or camera uses those source frames, state that shared conditioning.

The public observation producer is:

```bash
python -m tools.gvhmr.depth_observations \
  --config /absolute/depth_config.json --output /absolute/new_depth_anchors
```

The config declares `schema_version:1`, `actor_id`, `source_video_sha256`,
`raw_to_room` (rigid4x4), a frozen `train_row_mask` for every Pi3X row, and `paths`
resolved relative to the config. Required path keys are `native`, `ground`,
`ground_report`, `observations`, `quality_report`, `cameras`, `pi3x_inputs`,
`pi3x_predictions`, `pi3x_input_report`, `tracking`, `review`, `masks`, `coco_regressor`, `smplx_to_smpl`
and `smplx_joint_names` (the pinned official joint-name Python source).
Use the native parameter mesh/export reports from `audit_native_ground.py` and
`export_motion_quality.py`. `review` binds the actor/video and declares half-open
`torso_visible_intervals` and optional `shoulder_visible_intervals`. `masks` is the
full-rate boolean NPY; camera NPZ uses `dense_c2w_room_cv`/`dense_time_seconds`.
The producer checks raw Pi3X cameras against this room camera basis, derives
raster sizes from the inputs, and uses reviewed training frames for its fixed yaw.
If the room-camera cache lacks an embedded source-video hash, also supply
`paths.room_basis_report` binding the source, raw-to-room transform, Pi3X prediction
hash and Pi3X input-report hash. A geometrically consistent camera from another
video is still invalid input.

Outputs are `rigid_body_room.npz`, `raw_hip_depth.npz`,
`surface_corrected_hips.npz`, `surface_corrected.npz` and `report.json`. Body caches
include full geometry/topology, `fps`, exact times, joint names, source bindings
and lifecycle for the shared Blender importer. Observation NPZs contain
`frame_indices`, `time_seconds`, `root_targets`, `valid`, `weights`, `train` and
the actor/video/room/body hashes. Review the report's source samples and rejection
reasons before consuming the fit.

## 3. Correct the trajectory without changing body size

For the original room-space pelvis `q(t)`, fit a positive trajectory scale `s`
and translation `b` around a training-only anchor `q0`:

```text
q_fit(t) = q0 + s * (q(t) - q0) + b
delta(t) = q_fit(t) - q(t)
V_new(t) = V_original(t) + delta(t)
J_new(t) = J_original(t) + delta(t)
```

This scales travel distances, not pelvis-relative anatomy. For near-stationary
motion, scale is poorly identifiable: use translation-only or retain the baseline.
Do not fit body size to projection or resize furniture to meet the estimated hand.

If supported drift remains, fit a low-dimensional smooth residual `d(t)` to the
training observations. Penalize correction acceleration and magnitude, use smooth
endpoint conditions, and hold the correction constant outside observed support.
Report correction speed/acceleration and inspect acquisition/release and clip
boundaries. Smoothing the added path does not repair jitter already in the pose.
Do not fit every noisy depth independently or snap each frame to a camera ray.

A separate constant vertical translation `z_offset` moves the actor in room Z,
positive upward. Estimate it from a reviewed standing/walking support interval or
set an explicit reviewed value. It preserves velocities and articulation. A
lowest-vertex rule over the entire clip can erase intentional raised feet/jumps;
do not confuse it with measured support. Recheck all contacts after the shift, and
do not apply an additional implicit grounding step on import.

Every stage stores its immediate-source hash and **its own** translation offsets.
Archive native/world-derived parameters that no longer describe the corrected
geometry; stale `transl`, foot positions or cumulative offsets must not appear as
the authoritative output. Retain local shape/articulation and lifecycle metadata.

Run an ordered, source-bound protocol with the core Python environment:

```bash
python -m tools.gvhmr.align_depth_trajectory \
  --body /absolute/new_depth_anchors/rigid_body_room.npz \
  --observations /absolute/new_depth_anchors/surface_corrected.npz \
  --workflow /absolute/workflow.json --output /absolute/new_corrections
```

`workflow.json` has `schema_version:1`, the exact `source_body_sha256` and
`observations_sha256`, optional `contacts_file`, and ordered `stages`. Each stage
has a unique simple `name`, `method`, optional `options` and optional `limits`.
Paths are relative to the workflow. Each stage writes a distinct body/report;
the top-level `report.json` records the stages and their scope of acceptance.

| Method | Options / intended use |
| --- | --- |
| `scale_translation` | `fit_scale:false` for translation-only; positive scale bounds otherwise |
| `smooth_residual` | `knot_spacing_seconds`, `acceleration_penalty`, `offset_penalty`; estimated-depth drift |
| `smooth_contact` | Same smooth settings, with reviewed finite contact targets from `contacts_file` |
| `constant_contact` | One contact normal translation; finite extent is checked after fitting |
| `vertical_origin` | Reviewed `vertex_indices`, half-open `intervals`, `reviewed_by`, `review_scope`, optional `floor_z_m`/`target_gap_m` |
| `vertical_offset` | Explicit constant `offset_z_m`; record the placement review in the protocol/handoff |

Stage limits can bound `max_root_offset_m`, `max_added_velocity_m_s` and
`max_added_acceleration_m_s2`. Pick tolerances before comparison; do not relax a
failed gate solely to relabel the candidate successful. All later stages recheck
declared contacts, including after vertical placement. `accepted` means only the
declared numerical/geometry/contact gates passed; source style and the complete
authored scene still require review.

## 4. Add requested contact constraints

First locate the event in the source: body side/part, object identity, acquisition,
contact interval and release. Associate it with a finite authored surface in the
same room basis. For example, clip32 uses the anatomical right-hand patch and the
pantry panel; walking support uses separate left/right sole patches and reviewed
stance periods. A low/slow foot heuristic is only a contact proposal. Unobserved
hand area, finger articulation and hidden support remain uncertain.
Distinguish intermittent fingertip touches from continuous palm adhesion, and
foot support from a proven stationary plant. Do not pin a whole patch or impose
zero support-relative velocity unless the source supports that stronger claim.

Validate contact-sensitive scene geometry from static source evidence. If a dark
panel has unstable depth, independently triangulated edges may support a revised
room candidate. Keep the old room and compare geometry separately from motion.
Matching an infinite plane outside the actual object is not a successful contact.

Add targets to the smooth root correction for contact gap and nonpenetration;
for a reviewed planted event, constrain support-relative drift as well. Ramp
contact acquisition and release so a hard frame switch does not cause a jump.
Keep depth observations as placement evidence while honoring the contact-first
objective. Record target weights/tolerances and evaluate all simultaneous targets
on final body geometry, including the feet when correcting a hand contact.

`contacts_file` binds `schema_version:1`, `source_body_sha256`,
`source_video_sha256`, `source_actor_id`, `room_basis_sha256`, `reviewed_by` and
`review_scope`, plus `contacts`. Each contact declares:

- `name`, `landmark:{kind:vertices|joints, indices:[...]}`, half-open `intervals`
  and explicit `train_frame_indices`. Select real anatomical mesh indices;
  joint-only constraints do not establish surface contact.
- `surface_file`/`surface_sha256`: NPZ with finite room-space `vertices` and
  triangle `faces`; outward unit `plane_normal` and `plane_offset` define
  `n·x+d=0`. Without solid geometry the finite mesh must be planar. Optional
  `convex_planes_0`, etc. in the NPZ permit declared convex-solid penetration checks.
- `target_gap_m`, `max_gap_m`, `max_penetration_m`, `weight`, and optional
  `ramp_frames` for raised-cosine acquisition/release weighting in the objective.
- Optional `plant:{reviewed_by, review_scope, weight, max_slip_m_s}` only for a
  source-reviewed stationary contact. It constrains patch-centroid speed and gates
  all event-frame pairs. It does not prove every contacting vertex is stationary.
  An optional `plant.train_frame_indices` declares its separate training cohort;
  report that split explicitly rather than calling those frames held out for plant fitting.

For intermittent touch hypotheses, `diagnostic_intervals` can retain the wider
uncertain windows for measurements while `intervals` contains the supported
contact keys/windows used by acceptance. Declare those cohorts from source review
before inspecting fits; do not remove failed frames retrospectively to pass a gate.

The current public solver supports **static** room surfaces. Moving objects need
time-varying support targets; do not treat static-room centroid speed as motion
relative to an animated door or chair. Plant constraints never bridge a release
interval. Ramps change fitting weights; final gap, penetration and slip gates
still inspect all declared active event frames, including held-out and ramp-end
frames. Near-surface vertex counts are not measured contact area.

Root-only correction preserves articulation but may be unable to satisfy a hand,
seat and both feet simultaneously. Report that conflict and retain the candidate
as rejected/partial. When a pose change is required, use the documented experimental
[Kimodo contact editor](../tools/kimodo/contact_edit.py) with action text,
visible-part preservation and native joint targets or contact guidance. Check
contact after decoding/skinning and source-relative transition smoothness.
Generated motion stays labeled; stock conditioning is not an exact preservation
or contact guarantee. Retired external arm IK remains retired.

## 5. Bake and evaluate the result in the authored scene

Use the shared `src/aha3d/blender/body.py` importer on a saved room and
write a distinct `.blend`. Bake terminal activity into constant visibility keys,
with no runtime callbacks needed for playback. Exclude inactive padding from
contact/collision cohorts and synthetic visible observations. A frustum test alone
does not establish source visibility.

Reopen the saved scene and check all frames, especially the frame before exit,
the first absent frame and the final scene frame. Check fixed body-relative
dimensions, unchanged local articulation, trajectory changes and camera/scene
preservation. Resample rotations before skinning if output FPS changes; preserve
visibility event timing separately. Review source style and contact patches in
both a readable room overview and close views of the support surface.

Measure finite-surface gap, penetration and support-relative slip on fixed reviewed
cohorts; report empty/unknown cohorts explicitly. A floor improvement can worsen
hand contact, and contact alignment can worsen estimated depth agreement. Deliver
the tradeoff, not a single projection score. Fully decode full-duration videos
and verify FPS/frame count/PTS. The shared web comparison accepts multiple cases
and evidence links via `tools/human_experiments/build_portal.py`.

Evaluate simulated ground clearance and unintended object intersection over the
**entire active track**, even when the source feet are cropped or occluded. Their
true motion is unknown, but penetration of the authored floor is still measurable.
Reject a contact candidate that improves selected support cores but substantially
regresses the rest of the active scene. Clip42's soft-support experiment passed
its local patch gates yet increased full-active floor penetration; it is not an
adopted replacement. Keep source-observation uncertainty separate from simulated
geometry validity.

For repeatable execution and compact review, follow
[Human automation](HUMAN_AUTOMATION.md). The `smooth_scene_contact` workflow adds
finite full-active floor and explicitly listed convex-object penalties directly
to the root solve. Bind its `scene_constraints_file` to the frozen scene geometry
and source/body identifiers. Keep global scene gates as well as event gates;
do not append an unreported vertical shift after the joint solve.

Plant optimization currently penalizes patch-centroid motion. Acceptance also
checks corresponding patch-point speed, because a stationary centroid can hide
rotation-induced sliding. Preserve derivative endpoint semantics in exported
metrics. Report contact and plant training masks separately: a heldout contact
frame may still participate in plant training. Neither this proxy nor a passing
numerical gate establishes exact contact area or source-style fidelity.
