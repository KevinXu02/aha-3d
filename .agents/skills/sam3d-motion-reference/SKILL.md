---
name: sam3d-motion-reference
description: Review sparse SAM 3D Body observations and compile arm or experimental leg directions into native Kimodo guidance. Use for reference-assisted generated gestures and motion keyframes; source-person reconstruction uses GVHMR.
---

# SAM 3D Body motion references

Extract observations and compile sparse guidance for generated new/changed actions
or an authorized fallback. Source reconstruction uses [GVHMR](../gvhmr-body-reconstruction/SKILL.md).
A reference-only request ends at reviewed observations/guidance; it does not authorize
motion generation or complete scene reconstruction.

## Workflow

| Stage | Action and output | Before advancing |
| --- | --- | --- |
| Select | Same-shot Pi3X bundle and useful actor/event frames | Preserve `inputs.json`, `inputs.npz`, `cameras.json` and exact source frame identities |
| Observe | SAM poses and full-body/source overlays | Review identity, visibility and plausible limbs; record rejected or missing evidence |
| Compile | Reviewed events and native baseline into `constraints.json` | Inspect world rotation and implicit root/heading fields; preserve baseline proportions |
| Compare, when requested | Kimodo baseline/guided mesh and scene evidence | Review changed motion, withheld source times and actual support/contact under the requested scope |

Use **root > hands >= feet > text-only** under [motion policy](../../../docs/MOTION_CONTROLS.md).
Keep observations, reviewed selection, compiled guidance and generated motion as separate
artifacts. Identify useful events during source analysis; reuse valid observations
when root/facing interpretation changes, then recompile affected guidance and motion.

## Load for the current operation

| Need | Reference |
| --- | --- |
| Inference or cached compilation | [Workflow and runtime](references/workflow.md); isolated SAM GPU inference, installed Kimodo CPU compilation |
| Author events and world rotation | [Selection schema](references/schema.md) |
| Jitter diagnosis or optional arm refit | [Temporal guidance](references/temporal.md) |
| Cached full-body overlays or experimental native feet | [Leg review and comparison](references/legs.md) |
| Missing same-shot bundle | [Pi3X](../pi3x-scene-reference/SKILL.md) |
| Requested generation, skinning or mesh previews | [Kimodo](../kimodo-body-motion/SKILL.md); [mesh preview operations](../kimodo-body-motion/references/motion.md#mesh-previews-and-contact-experiments) |

Use shared `src/aha3d/motion/sam3d.py` and `reference.py`; the compiled JSON enters
an ordinary recipe through `body.constraints`. Scheduling and reuse follow the
[scene workflow](../../../docs/SCENE_WORKFLOW.md).

## Guidance boundaries

The proper world rotation maps Pi3X Z up to Kimodo Y up. Retain baseline root placement
unless explicitly overridden; estimated SAM depth is not a root target. Native hand
keys also fix root height, smooth root and heading: keep events sparse and inspect
those fields so facing overrides do not suppress intended turns. Post-generation
arm IK and keyed facing edits remain retired.

Inspect actual full-body overlays: in-frame flags describe bounds, not occlusion
confidence, and reprojection checks adapter conventions rather than pose accuracy.
Use `motion.observations` to redraw cached observations. Direction retargeting retains
existing proportions; exact contacts, wrist/ankle twist, full-body conversion and
path fitting require separate work. Experimental feet require the linked baseline,
withheld-time and skinned-mesh comparison before adoption. Show human motion as
actual SMPL-X mesh by default; room-free previews do not establish furniture contact.
