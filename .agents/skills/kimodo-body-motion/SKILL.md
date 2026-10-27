---
name: kimodo-body-motion
description: Generate and review human or native G1 motion with Kimodo. Use for new or changed actions, scene interactions, and model-call debugging; use GVHMR for reconstructing observed people.
---

# Kimodo body motion

Deliver the requested motion scope: a complete interaction includes scene placement,
simultaneous hand/support contact refinement, validation and the updated viewer.
Generation-only and debugging requests retain their narrower scope. Preserve the
source scene and motion in distinct output variants.

## Workflow

| Stage | Action and output | Before advancing |
| --- | --- | --- |
| Plan | Record action phases, timing, contact/release intervals and actual room supports | Derive root travel from reach, stance and clearance; keep text and constraints consistent |
| Generate | Use the configured runtime and [motion recipe](references/motion.md), saving native motion and inputs | Check travel, height drift and segment joins; each prompt is at most 10 seconds in the deployed model |
| Review and refine | Resample rotations before skinning; inspect actual mesh facing, support, contact and transitions | Compare simultaneous contacts and clearance against the intended action; retain unresolved findings |
| Integrate and deliver | Use [scene assembly and delivery](../../../docs/SCENE_WORKFLOW.md#integration-and-delivery) for scene work | Reopen and check every actor, review actual images and fully decode requested videos with their timing |

For source-informed generation, use **root > hands >= feet > text-only** under
the [motion policy](../../../docs/MOTION_CONTROLS.md). Review same-shot observations
and record missing or unreliable references; text supplies semantics. Text-only
requests retain their text entry point. A route or constraint match still needs
mesh review for facing, natural motion and finite-surface contact.

## Load for the current operation

| Need | Reference |
| --- | --- |
| Recipes, duration, independent actors, stage stops or mesh previews | [Motion operations](references/motion.md) |
| Native calls, segmented interactions, G1 or implausible output | [Call and validation](references/call-and-validation.md); shared frame adapter: `src/aha3d/motion/kimodo_frame.py` |
| Revise interaction planning | [Interaction planning lesson](../../../docs/lessons/interaction-motion-planning.md) |
| Explicit source-occlusion completion research | [Completion modes](references/completion.md) |
| Mesh/joint correspondence or synthetic tracks | [Tracking schema](references/tracking.md) |
| Runtime configuration or capacity failure | [Kimodo/Blender runtime](../../../kimodo_blender/RUNTIME.md) |

## Source-motion boundaries

Observed-person reconstruction and source-contact repair use
[GVHMR](../gvhmr-body-reconstruction/SKILL.md) and the current
[motion policy](../../../docs/MOTION_CONTROLS.md), including when an older lesson
describes a different correction. Kimodo replacement/completion requires explicit
authorization; contact failure or low-confidence observations alone do not enable it.
Confirmed exits hide the actor; preserve `track_active` through generation/import.
Temporary occlusion follows the selected completion contract.

For source-contact work, retain **contact > source action/style > image alignment**,
body scale 1 and source timing. Smooth whole-body translation may improve contact
and penetration while preserving local articulation. Do not modify joint poses to
remove collisions or run human 6D leg repair/leg IK. Report infeasible simultaneous
contacts rather than changing stature, dropping targets or replacing the action.

Use actual SMPL-X mesh previews for humans; G1 uses its native checkpoint, skeleton
and rigid-link meshes. For scene delivery, finish the shared
[acceptance](../../../docs/WORKFLOW_ACCEPTANCE.md) and select the exact reviewed result.
Keep numerical checks, visual judgment and approximation limits separate.
