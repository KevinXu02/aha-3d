---
name: indoor-scene-workflow
description: Coordinate reference-to-room reconstruction and scene delivery. Use for complete scene requests or resuming a scene across reference, room, appearance and human-motion stages.
---

# Indoor scene workflow

Use the [scene workflow](../../../docs/SCENE_WORKFLOW.md) for stage inputs,
outputs, review gates and reuse. Load specialist guidance only for the active stage.
Scene geometry, actor decisions and source selections belong in the scene workspace;
shared implementation belongs in `src/aha3d/`.

## Establish scope

Resume the source, scene state and existing task scope. For new reconstruction,
claim scene/delivery paths, record the [request](../../../docs/SCENE_REQUESTS.md)
with `bash tools/indoor intake`, and initialize or bind
[acceptance](../../../docs/WORKFLOW_ACCEPTANCE.md#declare-scope-before-downstream-work).
The request brief and acceptance registration are separate records.
Prefill known choices; ask only missing product decisions and wait for their answers.
Narrow corrections reuse established scope and preserve unrelated components.
Source video alone does not request camera animation; a static request stays static.

## Route by stage

| Active work | Use | Result needed next |
| --- | --- | --- |
| Source, timing and reference basis | [Pi3X](../pi3x-scene-reference/SKILL.md) | Reviewed source coverage, floor/scale and matching geometry/cameras |
| Editable room, furniture and requested camera | [RoomKit](../blender-roomkit/SKILL.md) | Saved room with per-object source/layout review |
| Requested materials and lighting | [RoomKit materials](../blender-roomkit/references/assets.md#material-behavior) and [rendering](../blender-roomkit/references/workflows.md#rendering-and-resuming) | Saved scene with actual appearance renders reviewed against the source; preserve whitebox and motion-only scope |
| Reconstruct observed people | [GVHMR](../gvhmr-body-reconstruction/SKILL.md) | All requested actors through shared-scene anchoring, default v2, contact refinement and scene validation |
| Generate new or changed actions | [Kimodo](../kimodo-body-motion/SKILL.md) | Reviewed motion and requested scene interactions; load [SAM references](../sam3d-motion-reference/SKILL.md) only when useful for native guidance |
| Integrate and deliver | [Scene workflow](../../../docs/SCENE_WORKFLOW.md#integration-and-delivery) | Reopened final scene, requested media, accepted and selected delivery |

Analyze people and contact-sensitive assets early; stabilize room/camera and target
assets before costly motion refinement. Reuse compatible reviewed outputs; an
explicitly independent rebuild retains its fresh-inference scope. For repeated
source-motion comparisons use [human automation](../../../docs/HUMAN_AUTOMATION.md).

## Advance, repair and finish

At each stage, inspect its output against the declared scope, record findings,
and repair failed work before dependent stages. Technical and visual reviews are
agent work within existing authorization. Keep user corrections open in acceptance
until current evidence resolves them; retain the full requested pipeline after failure.

For batches use [quality before expansion](../../../docs/SCENE_WORKFLOW.md#quality-before-batch-expansion).
When parallel work is requested, assign separate component outputs and one integrator
for the combined scene. Follow existing runtime/job policy when launching compute.
Complete the declared [acceptance and selected delivery](../../../docs/WORKFLOW_ACCEPTANCE.md),
with automated checks, actual visual review and remaining approximations recorded separately.
