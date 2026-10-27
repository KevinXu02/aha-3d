# Source motion operations

Read the section needed for tracking, room alignment, contact fitting or import.
The [full pipeline](../../../../docs/REAL2SIM_PIPELINE.md) owns the SAM3/PMPose/GVHMR
route; [motion controls](../../../../docs/MOTION_CONTROLS.md) owns correction and
completion boundaries. Preserve compatible caches, original estimates and rejected
candidates. Use [installation](../../../../docs/install/gvhmr.md) only for runtime setup.

## Actor tracking and track termination

Review the actor, source cuts, visibility and action coverage before initializing
SAM3. Preserve raw masks, actor ID, source hashes, frame IDs and timestamps; record
reviewed suppression as a separate derivative. `tools/gvhmr/person_reconstruct.py`
consumes person masks before fresh PMPose, image features and GVHMR inference.
A new mask overlay on cached motion is not fresh tracking-conditioned inference.

Separate detection support, interpolated crop boxes and physical activity. A
confirmed exit ends the segment and hides the actor while the scene/camera retain
full duration. Interior occlusion, initial missing masks and partial cropping do
not establish exit. Reentry needs a reviewed segment/identity association. Preserve
`track_active` and inspect frames immediately around each visibility transition.
The current single-actor terminal-prefix interface is not automatic all-person
association; reconstruct every declared actor explicitly.

For source-fidelity diagnosis, use `tools/gvhmr/motion_quality.py` with upstream
mesh-regressed COCO17 joints, full-image coordinates and exact timestamps. Report
body-part support and time-window errors separately. PMPose agreement measures input
consistency, not independent truth; hidden observations remain unknown. Keep native-
camera and independently aligned room-camera checks separate.

## Source motion and room basis

Use the reviewed same-shot Pi3X reference and authored Blender room, preserving
raw predictions, depth/confidence, intrinsics and frame mappings. Choose static or
moving camera from the actual shot. Pi3X geometry and the global preview are reference
material, not the delivered room or a calibrated common frame.

Initialize with one fixed whole-clip rigid transform at body scale 1, then apply
default v2 and contact refinement. Record the Pi3X-to-room transform; uniform scene
scale applies consistently to geometry and camera translations, never twice.
Review floor/upright against static source surfaces; camera-up alone is not a floor.
Keep one common scene/camera basis across actors.

Only an explicit trajectory-refinement request enables the
[source-bound depth workflow](../../../../docs/HUMAN_TRACKING_AND_TRAJECTORY.md):
fit trajectory scale/translation and supported smooth residuals while preserving
body dimensions, articulation and heading. Actor-mask interior depth near reviewed
hips/shoulders describes visible clothing, not the pelvis center. Check axial depth,
pixel grids and camera conventions. Near-stationary motion cannot identify travel
scale reliably; retain the baseline or use translation-only correction.

A separate constant room-Z translation may correct placement. Choose it from
reviewed support and recheck simultaneous contacts; apply it once, including during
import. Preserve native-parameter reproduction and source-placement evidence.

## Multi-person scale gate

Body scale is **1** for every source actor. Trajectory scale changes travel distance,
not anatomy. Keep native dimensions, local articulation and source shape fixed;
projection error does not authorize fitting individual or shared body size.

## Full-pipeline contact refinement

Prioritize **contact > source action/style > image alignment**. Review acquisition,
contact and release intervals; associate each anatomical body patch with its finite
authored support in the common room basis. Distinguish observed, inferred and unknown
contact. Validate the room from static source evidence before using it as a target.
Source finger articulation is not established by GVHMR or the 22-joint Kimodo skeleton.

Use `smooth_scene_contact` with a source-bound `scene_constraints_file` under
[contact refinement](../../../../docs/CONTACT_REFINEMENT.md). Smooth whole-body XYZ
translation may improve support/penetration while preserving local articulation.
Do not modify joints to remove collisions, run human 6D leg repair/leg IK or use
per-frame viewing-ray floor snapping. Regularize correction velocity/acceleration
and ramp acquisition/release; evaluate source projection and continuity afterward.

Measure finite-surface gap, penetration and corresponding patch-point slip over
all active frames, including occluded/cropped intervals. A stationary centroid can
hide rotational slip; raising a body can reduce penetration while worsening support.
Report finite/convex proxy limits and compare the complete active track with the
baseline. Reject local improvements that introduce larger floor/contact failures.
Root-only correction may not satisfy hand, seat and feet simultaneously; retain
that conflict rather than resizing the body, moving furniture or dropping targets.

## Import into the modeled room

Open the saved authored room in background Blender and save a distinct output.
Use `src/aha3d/blender/body.py:import_cache` or the
[ensemble interface](../../../../docs/SCENE_WORKFLOW_OPERATIONS.md#parameterized-room-and-multi-person-assembly).
Preserve materials, furniture, cabinet animation, other actors and the requested
camera. The pilot `import_room_blender.py` build mode clears the scene for a Pi3X
reference demonstration; use the shared importer for an authored room.

Apply recorded room transforms to all vertices/joints, use stable person IDs and
bake `track_active` with constant visibility keys. Exclude inactive padding from
contact and synthetic observations. Resample rotations/translations before skinning
when FPS changes; preserve source duration and visibility timing. Save fixed-topology
caches and baked playback without model weights, add-ons or startup handlers.

Reopen the final scene and check every actor and camera over all output frames,
including exit boundaries. Inspect source/room views for scale, heading, trajectory,
contact and framing; fully decode delivered videos with frame-count/FPS/duration
checks. Finish [acceptance](../../../../docs/WORKFLOW_ACCEPTANCE.md) and selected
results with numerical checks, visual review and remaining uncertainty separate.

## V2 and optional generated motion

Use [world-postopt v2](../../../../docs/WORLD_POSTOPT.md) for full reconstruction.
Compare against GVHMR global motion, fixed scale-1 native alignment and the camera-
lifted initialization. Preserve camera-path, fixed-camera reprojection and foot-slip
metrics: v2's per-person camera transform is not the shared room camera, and its
own-camera reprojection can conceal placement drift. Verify the final scene through
the common camera; cached post-optimization alone does not complete reconstruction.

Retain GVHMR estimates in low-evidence intervals with uncertainty explicit.
Generated replacement/completion requires an explicit request and observed-action
text; contact failure or low keypoint counts alone do not authorize it. Read
[completion modes](../../kimodo-body-motion/references/completion.md) only for that
experiment: partial-body preservation and whole-body gap replacement have distinct
contracts. Preserve source shape/timing, outside-interval motion and visible-part
constraints as required; check decoded contacts and seams and label generated frames.
Confirmed departures never become generated continuations. Use
[Kimodo](../../kimodo-body-motion/SKILL.md) for authorized changed/new actions and
[SAM references](../../sam3d-motion-reference/SKILL.md) only when useful to that route.
