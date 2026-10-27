# Human motion control policy

## Full-pipeline completion contract

The owner clarified on 2026-09-20 that full source-person reconstruction must
continue through scene anchoring, v2 world post-optimization, human-object and
foot-ground contact refinement,
source-projection/collision/slip validation, and final integrated scene delivery.
Reuse verified scene and model caches. All visible people remain in scope.
A diagnostic label, failed placement gate or completed GVHMR/v2 run does not
satisfy the task; repair the failure and continue. An actual external blocker
must be reported specifically with the task still incomplete.

V2 is the default for full source-person reconstruction (owner decision, 2026-09-22).
Keep native GVHMR as a comparison baseline; running v2 does not automatically
accept its result. Standalone reference, placement-only and motion-only requests
retain their declared scope. See [the v2 route](WORLD_POSTOPT.md).

Initial placement uses one fixed rigid transform at body scale 1 in a shared
scene-camera basis. Contact correction follows placement and includes predicted
GVHMR foot contacts in the same solve as supported finite-object contacts.
Contacts need source evidence and actual scene surfaces; do not invent contact
for a walking person who never touches furniture. Inspect all active frames for
floor/object penetration and support slip, plus representative source projections.
Do not confuse per-person v2 camera consistency with shared-room placement.
If the entire clip lacks sufficient strong keypoint anchors, retry using low-
confidence predictions with smaller nonzero weights and robust multi-frame fitting.
A low score is not a reason to omit anchoring. Keep the weak-evidence status and
validate it with shared-camera projection and contacts; missing predictions are
not fabricated.
Depth/trajectory anchoring prioritizes supported pelvis/hip and shoulder anchors,
then falls back to any visible keypoint of the same person. Do not discard a
whole frame merely because both hip/shoulder pairs are unavailable. Convert each
body-part surface observation into an overall translation constraint through the
posed mesh, with confidence weights; never treat a wrist/ankle depth as pelvis
depth. Reject wrong-person/background samples explicitly.
Dense Pi3X camera tracks alone are not dense depth evidence; preserve matching
depth, confidence, intrinsics, source frame IDs and camera transforms for anchoring.
If a required intermediate was not saved, rerun its producer and persist it.

This full-pipeline authorization supersedes the historical opt-in language below
for complete reconstruction requests. Explicit placement-only, motion-only, review
or debug requests retain their narrower scope. Kimodo completion stays disabled
by default. The current translation-only comparison keeps all GVHMR/V2 joint
poses fixed; this is its experimental scope, not a prohibition on upper-body
contact refinement. The owner forbids leg repair and collision-driven joint-pose modification.
Whole-body translation during contact optimization may also reduce penetration.
Source-supported shoulder/elbow/wrist refinement may be considered with explicit
rotation-velocity/acceleration regularization and source-pose preservation.
It has not been run in this comparison. Separate trajectory
scaling and unrelated action changes remain outside the default route.
See [whole pipeline](REAL2SIM_PIPELINE.md).

## Initial native-motion placement

Keep source-supported hand/seat/foot contact refinement. Whole-body translation,
including a constant room-Z adjustment, is allowed to improve contact and
penetration. Apply the same translation to every joint and vertex; preserve body
size and local joint poses. Regularize time-varying corrections and inspect source
projection, velocity/acceleration, contact transitions and foot slip together.
Do not change joint poses to remove collisions or run human 6D leg repair/leg IK.
When translation cannot satisfy simultaneous contacts, report the remaining
conflict instead of distorting the action. Historical leg-repaired deliveries
remain withdrawn.

Initial alignment uses one fixed whole-clip rigid transform at scale 1. This is
the starting placement, not a restriction against default v2 optimization or the
subsequent contact translation. Preserve timing and resample rotations before
skinning. Synthetic in-frame flags do not establish source visibility.

The configured frontend supplies reviewed actor tracking before GVHMR inference.
Use the current SAM3 pipeline routing. Confirmed
exit ends/hides the track; temporary occlusion does not. Keep native body dimensions
and local articulation. Pi3X remains reference geometry; keep the editable Blender
room workflow. Kimodo is for new/changed actions or documented generated fallback.
Retired external arm IK, viewing-ray floor snapping and per-frame facing edits
remain retired.

## Explicitly requested refinement

Full reconstruction requests include contact refinement under the contract above.
Separate trajectory experiments require an explicit request for that work.
For a task already authorized to implement or use the
[all-person working-mode contract](HUMAN_PIPELINE_GUIDE.md), retain its declared
contact-position stage; do not ask again or silently remove that requested scope.
That contract distinguishes requested behavior from pending integration.
For authorized refinement, use [tracking and trajectory](HUMAN_TRACKING_AND_TRAJECTORY.md),
[contact repair](CONTACT_REFINEMENT.md) or [trajectory experiments](TRAJECTORY_REFINEMENT.md).
Keep whole-body scale 1; any requested trajectory-only fit must preserve native
body-relative geometry. Review actual source intervals and finite authored supports.
The experimental priority contact > source action/style > image alignment applies
only to requested contact-driven work. Report incompatible targets and remaining
errors; optimization convergence is not source or contact acceptance.

## Historical generated-motion decisions

The following records explain earlier experiments and rejected approaches. They
do not override the current default or authorize refinement in a new task.

Status: updated on 2026-09-09 local / 2026-09-10 UTC from the user's explicit
request to replace the text-first reference-video workflow. The accepted control
priority is **root > hands >= feet > text-only**. This governs input selection and
conflict resolution; it is not a claim that numerical model weights were changed.
That historical generated-motion priority is now scoped to Kimodo; the GVHMR
source-person default above takes precedence.

The earlier prohibition on external IK and keyed facing edits remains in force.
A subsequent native right-hand experiment (historical or external input; omitted from this bundle)
demonstrated target control but did not establish a smoothness improvement over
the accepted motion with fewer constraints.

## Rejected room-motion adapter (2026-09-11)

The user explicitly prohibited replacing the GVHMR global trajectory with
camera-space positions followed by per-frame viewing-ray floor correction.
Do not use this adapter or reuse its baked caches for subsequent reconstruction.
Preserve native global motion as the starting point for reviewed room alignment;
the current supported world-trajectory correction is documented separately above.
A replacement must check contact consistency and source projection together.
The ref42 v5 integrated motion is withdrawn; the room-only model remains available.
The scene state and stage audit are historical source-workspace evidence;
scene outputs and their baked motion caches are excluded from this bundle.

## Historical rigid-placement policy: body-size restriction retained

The user rejected independently scaled people and rejected adding a new shared
human-scale fit as a workaround. The audit found Pi3X already applies predicted
metric scale. Preserve SMPL-X body dimensions, nominal metric units, and native
motion; use rigid room alignment with scale fixed to 1. Verify source placement
and contact together. Neither individual nor shared scale fitting belongs in this
repair stage under that earlier policy. The current route permits trajectory-only
scale; any body/scene metric recalibration still needs independent evidence.

## Historical native-only default (2026-09-12; superseded for the current route)

At that time the user simplified the source-person route: retain GVHMR native global motion,
body dimensions and rotations; use one constant rigid alignment over the whole
clip, with scale fixed to 1. Do not run per-frame root/floor correction, trajectory
optimization, smoothing or IK by default.

If a particular person's legs penetrate the floor, first apply a constant vertical
translation to that actor only. Set `actors[i].vertical_translation_m` in the
manifest (metres along room Z, positive upward, default 0). Apply the same offset
to every vertex/joint and frame of that person after room alignment. Other actors,
room geometry and cameras stay in their aligned positions. Record the actor ID,
offset and source/room review evidence; never encode named-person offsets in code.
A constant shift preserves the person's trajectory differences, jumps and poses.
Review the full clip for remaining penetration or floating; do not escalate to
frame-dependent corrections or IK automatically.

The existing `run_contact.sh` and `contact_refine.py` entrypoints now default to
this native-preserving mode. They still verify native-parameter cache reproduction
and source placement. The previous reusable contact solver remains available only
with `--refine-contact` when separately requested; historical authorization does
not make automatic support-height repair or leg IK the current default. See
[invocation and validation scope](CONTACT_REFINEMENT.md).

## Reference-video default and priority

For source-informed Kimodo generation, use **root > hands >= feet > text-only**.
Use the reference video for root routes and useful hand/foot guidance, with text
supplying action semantics. Keep reference selection sparse and judged from the
actual source. Record material limitations when reliable reference data is missing.

Each native prompt lasts at most 10 seconds; longer clips use multiple segments.
Deliver continuous motion and review the rendered result against the source,
including gestures, support and framing. Preserve original candidates and report
remaining approximations. Let the agent choose the implementation using the
[SAM skill](../.agents/skills/sam3d-motion-reference/SKILL.md) and
[scene workflow](SCENE_WORKFLOW.md). Text-only requests retain their text entry point.

## Choose controls by task

| Requirement | Preferred control |
| --- | --- |
| Actions based on people in a source video | GVHMR estimation, room alignment, import into authored Blender room |
| Explicitly text-only walking, turning or gesture requests | Text with root guidance for the requested room/path |
| Touching a specific shelf/object or placing a hand/foot at a target | Sparse native Kimodo end-effector constraints at the necessary times |
| A particular whole-body pose at an important moment | Sparse native full-body keyframes, only when end-effector guidance is insufficient |

For video-assisted actions, inspect the source and divide it at meaningful action
changes before selecting text segments. The accepted 10-second walk plus
5-second gesture remains a historical example, not a default segmentation for
new footage or a claim of source correspondence. Write down approximate action
start/end times, walking versus pausing, facing changes and the active arm.
Use the reference-video priority above. SAM estimates require review before
becoming targets; adding more observations does not establish temporal correspondence.

On September 9, 2026, the user explicitly prioritized preserving native Kimodo
results over person/furniture collision avoidance for the
g0070 redo (historical or external input; omitted from this bundle), and
authorized inferred person placement. For that workflow, collisions are allowed:
do not reroute or repair generated poses to pass furniture checks. Disable
per-frame floor snapping unless specifically needed and authorized for the
action. Preserve raw generated motion, apply timestamp resampling before skinning,
and keep any display-only recentering distinct from the saved room animation.
This preference concerns generated people; it does not remove static room support
and attachment checks for furniture and props.

Historical scene-specific exception: in the final g0070 redo, even sparse native root guidance produced opposing
facing and travel direction in rendered views. The delivered generation therefore
uses text only and a single constant rigid placement of its complete native
trajectory. When native trajectory preservation is the priority, remove path
constraints that distort the action instead of adding more pose constraints.
A hidden-room preview can make the resulting motion visible without rerouting it.
This historical text-only delivery is not the default for new reference-video tasks.

Review generated actions at matching source times, including transitions and
release/lowering phases. Separate approximate action correspondence from finite
geometry, frame count and video decoding. When revising a mismatched action,
revise the native input description or necessary sparse guidance and regenerate;
do not fix the output with external IK or keyed facing edits. Report remaining
hand-height, facing and timing differences without calling the result exact
reconstruction. An unobstructed SMPL-X preview can reveal native motion when the
room view hides it.

Do not add contact or pose constraints solely to make a recipe more detailed.
Leave room for natural weight shifts and gestures; avoid fixing the root
throughout an ending that does not require a fixed position. Approximate motion
does not require fitting or reconstructing the reference person's poses.

## Native 3D targets for contact

Kimodo supports sparse full-body and end-effector constraints during generation.
For a hand, the native constraint includes wrist position/rotation and the hand
end position. It also constrains the smoothed root at the selected frames. The
model can generate the rest of the body around these goals. See the
[official constraint description](https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/constraints.html).

For a required shelf touch, begin with only the necessary contact keyframes
(for example, two or three well-chosen moments). Author feasible targets in the
room, transform them into Kimodo's Y-up metre coordinates with the same world
alignment and scale as the route, and generate the surrounding movement. This
is a starting experiment, not a validated number of constraints or a guarantee
of smooth motion.

The installed JSON interface stores pose/root data plus a constraint type such
as `right-hand` or `end-effector`; the latter selects the relevant end-effectors.
Providing a pose-shaped array in that interface does not automatically constrain
the whole body. Do not assume it accepts only an arbitrary wrist XYZ triple.
Check the selected skeleton's joint names and
[JSON fields](https://research.nvidia.com/labs/sil/projects/kimodo/docs/user_guide/constraints.html)
when authoring targets. Avoid accidentally fixing the root at every gesture frame
through the root information carried by hand constraints.

Use initial heading, text and route guidance for broad direction. Reserve a
native pose keyframe for an essential facing/pose event. Inspect approach,
contact, release and segment joins before adding more constraints. Verify hand
reachability, floor contact, clearance and temporal continuity in the actual room.

## Retired external correction method

The first version's generate-then-edit workflow, which changed arm IK and keyed
body facing after generation, is removed from the supported workflow. Do not use
`refine_body.py` as a recipe, fallback or starting point for new scenes. Existing
artifacts and historical records document an earlier result, not instructions
for new work.

This decision does not disable Kimodo's built-in motion correction. It is a
separate part of the generation workflow, and was enabled in the accepted 10+5
example. Sparse native constraints may still conflict or produce visible joins;
they have not been established as a universal cure for jitter. The recorded test
reached three wrist goals within about 0.9–1.9 cm but had higher ending jerk
metrics than the accepted baseline. Keep spatial accuracy and smoothness as
separate checks; see the experiment's scope and evidence before generalizing.

## Reusable placement and contact acceptance

The user requires shared methods across scenes and actors. Do not encode a named
person, a reference-specific time interval or a hand-tuned depth offset in a shared
correction. Source-specific observations and reviewed exclusions belong in the
manifest, with evidence; generic policy applies equally to all actors.

Actual source-observation placement is checked before and after contact repair.
Low coverage is unverified; bad time windows cannot be hidden by aggregate medians.
Body scale stays 1 and source shape remains fixed. Contact translation may change
root XYZ while preserving local articulation; review the resulting trajectory.
Temporal diagnostic probes do not authorize per-frame camera-ray grounding or
claim to solve trajectory drift. See [shared method](CONTACT_REFINEMENT.md).
