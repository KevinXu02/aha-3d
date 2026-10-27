# Staged reference-to-scene workflow

Use this workflow for complete scene requests and cross-stage revisions. Start with
[indoor-scene-workflow](../.agents/skills/indoor-scene-workflow/SKILL.md); load only
current-stage references. Existing runners, recipes and acceptance records implement
execution and delivery. The sections below define their order and handoffs.

## Plan once, consume results at the appropriate stage

Resume the source, scene `STATE.md`, request and acceptance scope. For new work,
claim scene/delivery paths, record [requirements](SCENE_REQUESTS.md) with
`bash tools/indoor intake`, then [initialize or bind acceptance](WORKFLOW_ACCEPTANCE.md#declare-scope-before-downstream-work).
Prefill known choices and ask only missing decisions; a brief is separate from
acceptance registration and executable recipes. Narrow revisions reuse scope.

Record source/timing, cuts, actor IDs, important action/contact intervals, visible
layout and uncertainties. Preserve requested appearance, fidelity, camera, timing
and independent people/cabinet/object motion. A source video does not itself request
camera animation. Video delivery includes the editable scene and preview stills;
an interactive demo is optional. Requested variants remain obligations: deferred
intake integration still permits authorized standalone [variant work](SCENE_VARIANTS.md).

## Stage contracts

Each stage consumes compatible inputs, produces saved artifacts and records the
review needed by its consumers. Repair failures at the affected stage, then resume
its dependents. Reviews are agent work within the authorized scope; they add no
user approval pause. Keep user findings open until current evidence resolves them.

| Stage and inputs | Output | Pass condition and procedure |
| --- | --- | --- |
| Source analysis: original video and scope | Reviewed source frames, exact timing, scene/actor inventory | Cover cuts and newly revealed spaces; use [Pi3X](../.agents/skills/pi3x-scene-reference/SKILL.md) for same-shot predictions |
| Reference alignment: source observations and Pi3X | Dimensions, floor/scale status, common world transform and matching cameras | Source-linked static endpoints and cross-view evidence; mark unsupported or inferred geometry explicitly |
| Room/camera: reviewed reference basis | Editable saved room, requested camera and comparison views | [RoomKit](../.agents/skills/blender-roomkit/SKILL.md), camera preflight and per-object layout review; Pi3X stays reference geometry |
| Appearance: reviewed room/camera and requested materials | Saved material scene and comparison renders | [RoomKit materials](../.agents/skills/blender-roomkit/references/assets.md#material-behavior) and [rendering](RENDERING.md); inspect actual source/render appearance and preserve whitebox or motion-only scope |
| People: reviewed identities/actions and target supports | Motion, activity/timing and refined scene contacts | Observed people use [GVHMR](../.agents/skills/gvhmr-body-reconstruction/SKILL.md); new/changed actions use [Kimodo](../.agents/skills/kimodo-body-motion/SKILL.md) |
| Integration: saved room, compatible cameras and all actor caches | Baked ensemble and reopened scene checks | Every declared active actor present, departed actors hidden, actual floor/objects/framing reviewed |
| Delivery: reviewed final candidate | Scene, requested media and selected manifest | [Acceptance](WORKFLOW_ACCEPTANCE.md) and [results](RESULTS.md), including actual visual review and full final-video decoding/timing |

Analyze people early; stabilize the room/camera and contact-sensitive target assets
before costly motion refinement. Preserve completed compatible stages. An explicitly
independent rebuild retains fresh source-based inference. Source-human defaults and
tracker routing live in [the full pipeline](REAL2SIM_PIPELINE.md), with constraints in
[motion controls](MOTION_CONTROLS.md): fixed whole-clip rigid initial alignment at
body scale 1, shared-scene anchoring, default v2 and contact refinement through final
delivery. Generated completion remains an explicit opt-in.

## Furniture layout review entrypoint

From the first blockout, use [RoomKit inspection](../.agents/skills/blender-roomkit/SKILL.md#choose-the-inspection-tool)
and the [object review contract](LAYOUT_INSPECTION.md#object-review-default-and-repair-loop).
Inspect each main and user-flagged item's isolated X-ray/source and plan/side views;
use visible contours, front details and depth where silhouettes are ambiguous.
Record exact object IDs, source/render views and before/after findings against the
saved candidate. The pipeline generates the focus packet; the linked contract
provides standalone commands, filter coverage checks and evidence requirements.

## Layout inspection before people and final rendering

Start with the [early source-relation check](LAYOUT_INSPECTION.md#early-source-relation-check):
repair user corrections, shared planes, counts, silhouettes and facing with local
views. Freeze the repaired candidate, then complete its common-frame layout review.
Exclude moving people from static reference and distinguish glass/reflections and
missing coverage. Keep actual decoded frame IDs/PTS and texture-source provenance;
geometry and camera agreement are separate from visual source fidelity.

## Quality before batch expansion

Complete one representative scene through room, materials, source-camera comparison
and requested preview review before expanding a new batch. Thereafter keep one
unfinished authored scene per worker and resolve its visible findings before the next.
Bounded reference preparation and rendering of reviewed frozen scenes may run in
parallel when requested, with separate outputs and one owner for shared catalogs.
Search registered assets before custom authoring; review a complete custom object
before duplication. Use complete semantic roots and source-observed facing under
[asset orientation](ASSET_ORIENTATION.md) and [reuse conventions](SCENE_VARIANTS.md).

## Reuse boundaries

Record changed inputs, object IDs, affected neighbors/supports and dependent stages.
Use immutable variants and the [existing resume/reuse interface](PIPELINE.md#resume-and-reuse).
Let the runner check identity; preserve original sources and frozen snapshots.

| Change | Reuse | Recompute or review |
| --- | --- | --- |
| Labels, colors or display slices | Raw predictions, RGB and unchanged geometry/cameras | Presentation; preserve measurement endpoints |
| Geometry or object placement | Matching source predictions and reviewed camera | Affected layout/support and appearance; final candidate review covers wider occlusion, shadow and reflection changes |
| Material or lighting | Matching geometry, reference and body caches | Affected appearance renders and final acceptance |
| Floor basis, scale or camera | Only compatible upstream inputs | Dependent measurements, points/cameras, placements, assembly and evidence |
| One actor's action or native keys | Room, camera, other actors and suitable source observations | That actor's generation/resample/skin, ensemble and affected contact/framing checks |
| SAM selection or facing interpretation | Suitable same-shot observations | Native guidance, affected motion and checks |
| Source shot, cadence or body shape | Only inputs with matching identities/timing | Dependent reference, camera, motion, skinning and assembly stages |

For local repairs, hold reviewed cameras fixed unless camera error is established;
inspect changed objects and neighbors before the complete candidate review. Recheck
contacts after target-asset replacement. Resume encoding/reporting from intact frames
and malformed review responses from unchanged image packets. A comparison-only
failure needs a corrected comparison, not another render. Keep source-layout,
placement, camera, appearance and preview findings separately visible.

## Integration and delivery

Open the saved source in background Blender and save a distinct output, preserving
materials, unrelated animation, common world coordinates and requested timing.
Resample body rotations before skinning. Reopen the final saved scene and verify
all declared people and cameras; the single-body pipeline report covers only its
selected body. Complete preview review before final video rendering. Then bind
checks and actual visual observations to the exact candidate and finish
[acceptance](WORKFLOW_ACCEPTANCE.md) and [selected delivery](RESULTS.md).
Report remaining approximations and unresolved findings explicitly.

## Shared commands and configuration

Read only the relevant section of [scene operations](SCENE_WORKFLOW_OPERATIONS.md)
when executing the helpers below. Recipes, stage snapshots and resume are documented
in [pipeline](PIPELINE.md); compute placement and waiting in [runtime](../MACHINE.md)
and [render execution](RENDER_EXECUTION.md).

### Camera and independent static cross-view checks

Use [camera operations](SCENE_WORKFLOW_OPERATIONS.md#camera-and-independent-static-cross-view-checks)
after reference alignment and before combining room/motion. Review source accuracy
separately from camera-cache representation checks.

### Motion without a room copy

Use [Kimodo motion operations](../.agents/skills/kimodo-body-motion/references/motion.md#independent-actor-stages)
for `--motion-only` and generation/diagnostic stage stops. Source-person reconstruction
uses its own [full pipeline](REAL2SIM_PIPELINE.md).

### Parameterized room and multi-person assembly

Use [ensemble operations](SCENE_WORKFLOW_OPERATIONS.md#parameterized-room-and-multi-person-assembly)
for stable actor IDs, placement, assembly and all-person saved-scene verification.

### Final render and comparison

Use [delivery operations](SCENE_WORKFLOW_OPERATIONS.md#final-render-and-comparison)
for the final recipe and source/video comparison, retaining independent visual review.

## Execution and repeatability

Use existing runner locks, immutable snapshots and failure propagation. Standalone
helpers save configuration/provenance in distinct claimed outputs. Pause at a stage
boundary during extended authoring; follow runtime/job policy for resource ownership.
Record run/job IDs, selected/rejected candidates, evidence and next actions in the
handoff when stopping or transferring work. Preserve the full requested scope on retry.

## Submission preflight and full-clip preview acceptance

New pipeline video runs stop at `preview_required` after assembly/verification.
Review a sampled preview across the complete source duration, original-resolution
source/keyframes and all-person checks, then resume final rendering. Still-image and
motion-only runs retain their scope. Historical manifests keep their existing behavior.
Use [preview operations](SCENE_WORKFLOW_OPERATIONS.md#submission-preflight-and-full-clip-preview-acceptance)
for commands, exact evidence fields, sampling and provenance requirements. A sampled
preview review and full final-video decoding are separate completion requirements.
