# Real2sim indoor pipeline: room, cameras and source people

Updated 2026-09-20. This is the current end-to-end routing and its limits, not a
claim that every stage is one fully automatic command. The agent reviews source
identity, geometry and visual quality; models estimate references and motion;
Blender produces the editable scene. See [entry commands](PIPELINE.md),
[scene workflow](SCENE_WORKFLOW.md) and [world experiments](WORLD_POSTOPT.md).

The source-person frontend is **SAM3 → existing bbox postprocessing → PMPose-h → GVHMR**, with same-shot
Pi3X camera evidence. New full world runs reuse the accepted scene floor/upright
inside GVHMR, then run **v2 by default**, comparing it against scene-guided GVHMR. Kimodo gap
completion is **skipped by default** following the owner's visual quality feedback;
it remains an explicit experimental opt-in. The native rigid
alignment route remains available separately. Candidate validation does not
silently select an authored-room delivery.

```text
Source video + requested scope → shot, timing and actor review
  ├─ Pi3X geometry/cameras → reviewed floor, upright and scale → editable room/assets
  │                              │
  └─ SAM3 → bbox support/interpolation/smoothing → PMPose COCO-17 + image features
                                 │
                         GVHMR using scene floor/upright
                                 │
                dense whole-clip cameras → common scene-ground lifting
                                 │
                     shared-scene position anchoring → v2 with fixed floor/upright
                                 │
              finite-object + foot-ground contact refinement and validation
                                 │
Accepted motion + authored room + requested camera/object animation
  → rotation resampling → skinning/baking → saved-scene review → full decode
  → versioned delivery selection → Gallery / requested browser demo
```

## Completion requirement

Full reconstruction defaults to every requested stage, including reliable shared-
scene position anchoring, human-object and foot-ground contact correction, final
validation and integrated visualization. Reusing existing rooms does not remove
these human-scene stages. Intermediate results are debugging artifacts; failed
anchors or contact checks require repair and continuation, not a reduced delivery.
Use the [completion contract](MOTION_CONTROLS.md#full-pipeline-completion-contract).

Preserve dense Pi3X depth/confidence and matching intrinsics/cameras with explicit
source-frame mapping. A full-rate camera-only export cannot supply full-rate depth
anchors. Validate the crop-to-original-pixel transform and shared-camera projection
before contact solving; v2's per-person fitted camera is not the room-camera gate.

## 1. Establish source, timing and scope

Record the exact source, shot boundaries, frame timestamps, requested duration,
actor identities, important interactions, room coverage and intended deliverables.
Keep attribution and licenses with source videos. Separate camera motion, people,
furniture articulation and movable objects: a source video does not itself request
an animated output camera. Reuse established decisions for a narrow revision.

Use `bash tools/indoor intake` for a new high-level request and scene/run directories
for work. Claim writable code, scene and output paths before writing or launching
jobs. Cross-scene validation belongs under `runs/_tools`. Keep installed models and
machine paths in their existing external/local runtime configuration. For a new
scene reconstruction, initialize or bind the declared acceptance scope before
downstream work and retain user findings in its ledger; see
[reconstruction acceptance](WORKFLOW_ACCEPTANCE.md). Code/documentation experiments
such as this detector update are not scene reconstruction acceptance tasks.

The current full world harness requires normalized **1280×720, 30 Hz** video.
Reject other rasters at this interface; the motion tracking interface has fixed dimensions.
Preserve a source-frame/time mapping when normalizing a different input. Do not
truncate the scene to the last visible person or join unrelated camera shots.

## 2. Estimate same-shot scene references and cameras with Pi3X

Run Pi3X on appropriate source observations to obtain reference geometry, camera
poses and source-linked views. Establish one reviewed world basis, floor direction
and metric-scale hypothesis from reliable static evidence. Record uncertainty:
monocular geometry alone does not guarantee correct real-world dimensions.

Use source-colored bird's-eye/side views and measurements to review wall spans,
openings, furniture extents and spatial relations. A raw point cloud or mesh is a
reference, not the final authored room. Moving people, reflections, glass, hidden
bounds and unsupported depth are not reliable static measurements.

For world-postopt comparisons, run a **fresh whole-clip dense Pi3X forward** and
align its camera positions to the reviewed same-shot basis. This is distinct from
interpolating sparse reference cameras. The integrated external harness uses its
255,000-pixel inference budget with dimensions rounded to multiples of 14.
Record camera conventions and transforms. Planar/near-stationary camera positions
alone may not determine a stable 3D rotation alignment; inspect orientation and
static-scene projections as well. Do not recenter frame zero in a way that cancels
the global rotation. Recovered cameras remain estimates.

## 3. Author the editable room and furniture

Search `assets/INDEX.md` or `tools/asset_index.py` before building new furniture.
Use registered assets where their shape, dimensions, orientation and materials
match the source. Author missing geometry as editable semantic objects with
complete roots; place complete objects rather than independently moving children.
Preserve the declared front, support plane, texture dependencies and stable IDs.

Build walls, floor, openings and major furniture in the shared basis. Review early,
middle and late source views so later-revealed spaces are represented. Compare
silhouettes, facing, dimensions, circulation and occlusion. Validate material scale
and finish in actual renders. A plausible blockout is not a finished textured room.
Resolve contact-sensitive furniture before final motion placement. Inspect each
main object in its isolated X-ray focus packet, alongside source and plan/side
views, including structural walls; keep visible mismatches open until corrected
source-linked evidence closes them.

Open saved source scenes in background Blender and save distinct output scenes.
Preserve source materials; a clay comparison can use an output render override.
Asset replacement, when requested, follows the stable-ID library contract.

## 4. Track each source actor with SAM3

The full graph uses SAM3. Review an anchor image and actor bbox, then propagate
instance masks. Configure SAM3 with `--sam3`,
`--sam3-python`, `--sam3-checkpoint` (or `SAM3_ROOT`, `SAM3_PYTHON`,
`SAM3_CHECKPOINT`). SAM3 runs in its own runtime on the local GPU; PMPose retains its
separate interpreter. The remaining 30 Hz graph and scene-ground scope are unchanged.

Keep the existing bbox postprocessing below: SAM3 is only the tracker replacement.
PMPose receives the smoothed XYXY boxes and original masks, while GVHMR uses
the 1.2x centre/scale crops. Never replace partial masks with invented full-body masks.
The clip32 SAM3/PMPose pilot completed 260 frames, but a heavily occluded
7.8-second observation followed the foreground distractor. Crop smoothing does
not certify identity or pose fidelity; review these intervals before adoption.
Initialize a fresh predictor per propagation direction. An anchor after frame zero
requires both forward and reversed-prefix tracking. Save raw masks, raw boxes,
mask area, frame indices and actor/source provenance independently for each actor.

Review identity through crossings, occlusion and reappearance. A nonempty mask or
high-confidence skeleton can belong to the wrong person. The previous clip42 run
illustrates this: reverse tracking followed another person in early frames.
PMPose is conditioned on the selected mask and cannot repair a wrong identity.

## 5. Separate crop support from physical track activity

In `world_pipeline.py`, gate each mask bbox against the maximum of recently
accepted observations over roughly one second: bbox scale must retain at least
0.5 of the reference and actual mask fill at least 0.25. Keep the last accepted
history when a long failure would otherwise restart warmup. These checks reject
collapsed crops; scale is not bbox area, and fill uses the actual mask area.

Interpolate unsupported bbox centers/extents, apply the existing two 5-frame crop
averages and use 1.2× crop enlargement. These are **crop operations**, not smoothing
of the recovered 3D root trajectory. Keep the raw masks unchanged for PMPose and
review. The standalone reconstruction adapter retains its own bbox policy; use
the full world graph for this explicit trailing-maximum policy.

Mask support does not define a person's physical exit. Require source-reviewed
lifecycle metadata. Partial bodies/hands at the image edge remain active; hide the
body only at confirmed complete exit. The current adapter supports an active
prefix and terminal inactive suffix. Interior occlusion remains active; reentry
needs a new reviewed segment. Unsupported leading identity cannot be accepted by
silently extending an interpolated crop.

## 6. Extract mask-conditioned 2D pose with PMPose (new default)

`tools/gvhmr/person_reconstruct.py` and `world_pipeline.py prepare` default to
`--pose-detector pmpose`, variant `PMPose-h`. Use the actual per-frame source-person mask
and the same prepared bbox consumed by GVHMR. PMPose's merged COCO/AIC/MPII output
has 23 joints; retain its first 17 in canonical COCO order, full-image pixel
coordinates and confidence, producing `(active_frames, 17, 3)`.

A partial/collapsed mask is still the actual identity evidence even when its bbox
was interpolated. Never replace it with an invented full-body mask. An empty mask
produces zero-confidence keypoints. Maskless PMPose is rejected; there is no silent
ViTPose fallback. A requested detector control can explicitly use
`--pose-detector vitpose`. Legacy generic upstream demo/pilot entry points remain
upstream tools; they are not the default source-person route described here.

Run PMPose in the separate BBoxMaskPose environment and GVHMR in its compatible
GVHMR environment. Configure paths with `--pmpose-python`, `--pmpose-root`,
`--pmpose-checkpoint`, `--pmpose-variant`, and optional `--pmpose-ld-preload`.
Environment defaults are `PMPOSE_PYTHON`, `BMP_ROOT`, `PMPOSE_CHECKPOINT` and
`PMPOSE_LD_PRELOAD`; otherwise paths use the checkout's `.runtime/bmp` deployment.
No packages or other project's environments are modified.

Extract PMPose **before the first GVHMR inference**. The upstream compatibility
filename remains `preprocess/vitpose.pt`; its name is not detector provenance.
`provenance.json` records the detector, runtime, actual mask/bbox paths, confidence,
empty-mask frames and extraction counts. `actual_model_inputs.npz` contains the
keypoints actually supplied to the model. This avoids copying an old run and
leaving its native-motion/model-input exports stale after replacing a pose cache.

## 7. Infer source motion with GVHMR

Extract fresh image features and feed reviewed crops, PMPose keypoints, camera
rotation evidence and features to GVHMR. The mask-injection route forbids default YOLO
tracking and uses Pi3X camera evidence instead of DPVO. Upstream intrinsic-camera
estimation remains approximate: `estimate_K` assumes a diagonal-length focal, which
for a wide lens inflates in-camera depth (g0027: fx 1469 vs Pi3X 538 px, ~3x depth).
`world_pipeline.py prepare --intrinsics pi3x` substitutes the reviewed same-shot Pi3X
intrinsics (`camera_tracks.npz` `K_fullimg_pi3x`); the default remains `upstream`. Fresh-run guards reject existing pose, feature or
HMR caches; expected calls are PMPose=1, ViTPose=0, image features=1.

Save camera-frame and native global SMPL parameters, actual model inputs, body
shape, activity and source timing. New full world runs replace the inferred
gravity/world rollout and body-derived floor origin using the accepted scene prior
(section 9B), while retaining the estimator's contact-aware velocity correction
and local-pose processing. Record these changes in provenance. Only the active prefix enters feature extraction
and inference. Full-timeline exports may repeat the last body pose for storage,
but mark those frames inactive; padded poses are not observations or visible
motion. Preserve exact full-clip timing in final output.

For the separate native-only scene-import route, retain native articulation and global motion under **one fixed
whole-clip rotation and translation, scale 1**. Keep body dimensions fixed. Do not
automatically add per-frame floor/root corrections, temporal smoothing or IK. For
penetration, first consider one constant room-Z offset for the affected person.
Requested contact refinements are separate, evidence-based candidates.

## 8. Compare the scene-guided v2 candidate

Lift the GVHMR camera-space bodies through dense Pi3X cameras using explicit
camera-to-world conventions. Optional Kimodo completion precedes this step only
when explicitly enabled (section 9). Use the
same fixed scene-to-ground transform as GVHMR; floor height and upright come from
the reviewed scene fit, not the human. Historical cached comparisons without a
scene prior retain the old body-derived floor estimator and must be labeled accordingly.

Run **v2 only**, preserving shape/local articulation and disabling trajectory
scale optimization. The corrected implementation uses GVHMR's global-branch
orientation, per-person placement parameters, one learning rate with cosine decay,
and an acceleration weight of 0.1. The explicit config uses 1,000 iterations,
`postopt_lr_v2=0.01`, contact velocity weight 1000 and height weight 10. The flat
support target is 2 cm with a Huber height loss. No final root smoothing or frame-0
realignment is added.

Compare against **scene-guided GVHMR global output**, with its recording camera
recovered by per-frame Kabsch. If Kimodo completion is explicitly enabled, both
arms use the same accepted completion and the mixed baseline must be labeled.
Keep original raw inference separate. The former `native_rigid` result remains a
scene-alignment diagnostic, not the primary postprocessing comparator. Measure
foot slip, contact height/penetration and root acceleration. Keep model-contact
labels and body-derived floor uncertainty explicit.

Each v2 person has their own `camera_world`: this is a placement representation,
not a shared metric scene camera. Reprojection through that person's transform is
invariant in the mathematical construction; serialized SMPL/rotation evaluation
has small numerical error. The integration records the maximum camera-space joint
residual and checks a 2 mm numerical bound. Camera path length and projection
through the original Pi3X camera are coordinate diagnostics, **not criteria for
rejecting the motion estimate**. A fixed-world review removes only constant yaw
and horizontal translation for display, leaving scale and height unchanged.

The earlier conclusion that native is better because v2's camera path grows is
superseded. It combined an obsolete optimizer with an unsuitable camera criterion.
Current scene import still needs a reviewed common room basis, support surfaces
and visual/contact validation; changing the motion estimator does not select a
scene delivery automatically.

Detector confidence is not independently measured visibility. Scoring each model
against its own input detector is biased. Any cross-detector comparison must score
both candidate bodies against the same keypoint sets and source frames.

## 9. Optional Kimodo completion of low-evidence intervals

Kimodo completion is disabled by default. The owner found the generated motion
unsatisfactory on September 17, 2026; numerical seam checks did not establish
acceptable visual quality. The default graph retains GVHMR estimates in occluded
frames and proceeds to camera lifting/v2. These estimates can still be unreliable;
skipping generation does not repair missing motion evidence.

Explicit `--kimodo-completion --completion-prompt "Observed action description."`
adds a `complete` stage after `bodies` and before camera lifting/v2. No Kimodo
runtime, weights or action prompt are required by the default graph. The following
constraints and failure gates apply only to the opt-in experiment. `tools/gvhmr/occlusion_completion.py` counts COCO17 PMPose
keypoints with confidence at least 0.5, finite in-raster coordinates and source-person
mask support (5-pixel margin). Fewer than five marks the active frame invalid.
These are evidence thresholds, not measured visibility or an identity guarantee.

Consecutive invalid frames form a half-open interval `[start, end)`. The default
constraints are the three consecutive reliable frames immediately before and
three immediately after it. Each side can fall back to two; fewer than two leaves
an unresolved interval. The window including anchors must fit Kimodo's 300-frame
limit. An inactive terminal tail is never generated or used as an anchor.

Native `FullBodyConstraintSet` conditions use source-shaped world joints and global
rotations at those exact frames. The rejected interval's GVHMR pose AND root are
excluded from conditioning. Kimodo regenerates both. Original shape, timestamps,
all endpoint frames and every frame outside the interval remain unchanged. The
original inference run is retained separately; generated frames carry explicit
labels in the NPZ, prediction and report. Generated foot-contact labels replace
invalid GVHMR contact logits only inside completed intervals; wrist contacts there
are unobserved. Text describes the observed action and is recorded with the seed.

The decoded native candidate is archived before splice correction. A C2 endpoint
bridge is built exclusively from the reliable windows; Kimodo supplies a residual
whose envelope and first two derivatives vanish at both boundaries. Root positions
use a cubic spline and rotations use SO(3) interpolation, never matrix/Euler
blending. Only the invalid interval changes. Spliced foot contacts are additionally
gated by FK speed/height rather than blindly retaining pre-splice labels. The final
candidate must pass endpoint-error and seam-kinematics gates before export.
Three-frame context alone does not guarantee smoothness. Failed candidates and missing endpoints block subsequent optimization,
with a diagnostic candidate/report retained. Inspect actual source-shaped meshes,
source projections, floor support and finite object contact before scene selection.
Wrong identities require tracking repair, not generation.

## 9B. Reuse the reconstructed scene floor and upright

`tools/gvhmr/scene_ground.py` exports a source-bound prior from the preceding reviewed
scene floor fit. The plane `n dot x + d = 0` and upright are expressed in the exact
Pi3X aligned camera world. The source-video hash, camera-bundle hash, fit evidence,
scale and coordinate transforms are recorded. Camera-up alone is rejected as a
floor. Current flat-ground support requires the floor normal and upright to agree;
sloped or nonplanar supports need a different support model.

GVHMR receives this prior during world-motion reconstruction. Its scene-camera
rotations replace inferred gravity-view orientation in the world rollout. The
contact-aware trajectory keeps the first pelvis anchored to the scene camera;
its body-minimum ground-origin shift is cancelled. This preserves the scene height
instead of declaring that the lowest predicted body point defines the floor.
GVHMR jointly predicts a gravity branch: this installed implementation has no
separate gravity model to skip, so no saved model invocation is claimed.

The adapter applies one fixed rigid transform to bodies and cameras, making the
supplied floor Y=0 and upright +Y. No body-based floor fit runs. V2 uses that fixed
plane; its constant placement rotation is yaw-only, so fitting cannot tilt the
supplied upright. Scene-guided global orientations already share this world frame
and are not registered again with a free rotation. The inverse ground-to-world
transform is retained for scene export. Predicted metric scale remains uncalibrated.

These features do not establish correct actor identity, exact hidden source motion,
or furniture contact. Numerical checks, generated candidates and scene acceptance
remain separate. Earlier comparison runs retain their original body-derived floors.

The v2-to-room export is available through `python -m tools.gvhmr.world_scene export`;
see [the export command](WORLD_POSTOPT.md#export-v2-into-an-existing-authored-room).
It reuses saved room cameras and preserves scale. Optional reviewed depth targets
provide a single horizontal placement translation before contact fitting.

For requested scene-contact refinement, carry GVHMR's existing four foot-contact
channels into the same solve as source-supported hand/seat contacts. Keep all
GVHMR/V2 joint poses fixed in the current translation-only comparison. This
experiment has not tried upper-body contact fitting; the user prohibition concerns
leg repair and collision-driven joint modification. Smooth whole-body contact
translation may reduce intersections; check the resulting motion and source fit.
The human 6D leg repair stage is retired (owner correction, 2026-09-21). The v2 mesh
export receipt supplies source-bound logits and SMPL-X foot patches automatically;
no manual foot-contact detection is required. Penalize predicted-contact velocity
and sole-to-floor distance jointly, with the same reconstructed floor. Measure
slip again after the final correction: a smooth root shift can otherwise undo v2's
foot stabilization. See [scene-contact settings](WORLD_POSTOPT.md#export-v2-into-an-existing-authored-room).

## TODO: GPT-6-guided postprocessing optimization

Owner request, September 17, 2026. Status: broader optimization pending. The first
scoped change retains GVHMR-predicted foot velocity/height in scene-contact fitting;
this does not establish that motion quality or all simultaneous contacts are solved.

- [ ] GPT-6 should inspect source motion, final human meshes in the reconstructed
  scene, and frame-level diagnostics, then improve the postprocessing optimization
  code based on specific observed failures.
- [ ] Add or revise constraints for finite hand/object contact, foot support and
  slip, contact acquisition/release, and reliable
  source-pose preservation. Reuse the reconstructed floor/upright and keep native
  body scale. Evaluate collisions and allow whole-body contact translation to reduce penetration;
  do not modify joint poses for collision removal. Treat
  occluded evidence and uncertain contact hypotheses explicitly.
- [ ] Add motion-quality objectives for smooth root correction with fixed joint poses, stable
  support, plausible transitions and preservation of source action/style. Check
  positive foot clearance as well as penetration so lifting the body cannot count
  as a successful contact repair.
- [ ] Adjust loss weights, normalization, temporal schedules and convergence
  criteria jointly. If root-only correction cannot satisfy simultaneous contacts,
  report the conflict; do not enable leg repair or collision-driven joint modification.
  Kimodo completion remains skipped by default.
- [ ] Compare each change against the current v2 and contact-refined baselines on
  both clips using identical scene geometry, source timing, body scale and reviewed
  contact cohorts. Report gap/penetration, support-relative slip, root/joint
  velocity and acceleration, source-pose consistency and visual action quality.
- [ ] Keep fitting and evaluation cohorts distinct; label a fit using all reviewed
  keys as such. Save code/configuration, loss weights and per-frame before/after
  evidence. Select an improved candidate only after actual-room mesh review and
  applicable acceptance checks; numerical gains alone are insufficient.

Current findings to investigate: clip32 still has small object intersections;
clip42's left-foot support remains too high in part of its reviewed interval.
The occluded poses and clip42's uncertain opening identity require separate source
review and cannot be certified by tuning contact losses.

## 10. Integrate, validate and deliver

Resample rotations to delivery timing before skinning, then bake bodies and requested
object/cabinet animation into the authored Blender scene. Keep independent actor
IDs, fixed body shape and explicit hide-on-exit behavior. Use the reviewed scene
camera; importing an experimental v2 camera requires a coherent scene candidate.

For requested contact refinement, use actual finite support geometry and source
contact/release intervals. A floor solver is not a chair/hand solver. Measure gap,
penetration and support-relative slip and inspect occluded contacts honestly.

Reopen the saved `.blend`. Inspect representative actual renders, worst metric
frames, crossings and exit boundaries. Verify full video decoding, frame count,
rate and duration. Synthetic `in_frame` flags do not prove rendered visibility or
occlusion. Skeleton overlays are diagnostic and do not validate full mesh contacts
or furniture collisions. Keep numerical checks and visual review separate.

For a scene reconstruction, bind the exact final candidate and required evidence
to its acceptance scope; only a passing acceptance evaluation permits completion
and selection. Successful subprocesses or a registered manifest do not establish
acceptance. Register exact artifacts and evidence under `deliveries/<scene>/versions`, select
the accepted delivery, and rebuild/check the claimed Gallery. Export a browser
demo when requested from the selected manifest, never by newest filename. New
cross-scene algorithm comparisons do not select or replace a scene delivery.
See [results contracts](RESULTS.md) and the local run reports for measured scope.

## 11. Corrected v2 validation

The updated implementation reduces measured sliding on both retained PMPose
inputs, with higher clip32 root acceleration and worse clip42 penetration. See
[the corrected comparison](WORLD_POSTOPT.md#corrected-implementation-validation-2026-09-17)
for exact numbers, provenance and validation scope. The earlier blanket native
preference is withdrawn; room acceptance remains separate from the implemented ground/completion stages.

## 12. Historical PMPose integration validation (old v2, 2026-09-17)

The v2 numbers below predate the corrected optimizer. They must not be used to
rank the current v2 against GVHMR; see [current comparison](WORLD_POSTOPT.md).

These historical runs used the retired SAMURAI frontend; they do not describe
the current SAM3 interface. Fresh full graphs completed for clip32 (260/260 active frames) and clip42
(245/302 active frames), including new SAMURAI, PMPose, features, GVHMR, dense
Pi3X, v2 and review. The previous ViTPose runs serve as recorded controls. New
masks, actual bboxes and dense cameras were exactly equal to those controls.

| Clip | Motion | ViTPose slip (m/s) | PMPose slip (m/s) |
| --- | --- | ---: | ---: |
| 32 | native rigid | 0.0461 | 0.0380 |
| 32 | v2 | 0.1883 | 0.2161 |
| 42 | native rigid | 0.0558 | 0.0539 |
| 42 | v2 | 0.2224 | 0.1621 |

These historical results are model-contact-based diagnostics, not ground-truth accuracy. PMPose
changes are mixed, not universal gains. With PMPose, v2 camera travel remains
5.535 m versus the original 1.437 m for clip32, and 7.154 m versus 0.868 m for
clip42. Clip42's wrong-identity prefix remains rejected. No scene import was
selected, and no authored-room physical validation is claimed.

Verified exact PMPose/model-input equality, native-export/model-result equality,
PMPose=1 / ViTPose=0 / features=1 extraction calls, inactive suffix zero
keypoints, and all frames of both keypoint and world diagnostic videos. Reviewed
representative source-keypoint and skeleton sheets, including exit boundaries.
Twenty-five focused tests passed. A clean remote-main tracking adapter reproduced
the clip32 inference boxes exactly despite unrelated shared working edits.

Local evidence: `runs/_tools/pmpose-default/20260917/{REPORT.md,summary.json,
visual_review.json,comparison.html}` and `clip32-r2/run`, `clip42-r2/run`. The
report distinguishes reused controls from fresh inference and records the
remaining occlusion, identity and camera-drift limitations.

## Portable motion installation

The [human-motion installation guide](install/human-motion.md) records the
GVHMR/dense-Pi3X environment and separate SAM3 and PMPose environments. The
owner-authored v2 optimizer and helpers now live in `tools/gvhmr/world_backend/`;
`world_pipeline.py prepare` uses them by default without a PromptHMR checkout.
Upstream model libraries, weights and licensed body models remain external.
