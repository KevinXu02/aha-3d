---
name: blender-roomkit
description: Build or refine editable Blender interiors from references; reuse, place or replace furniture and materials. Use for room modeling, reusable interior assets, and requested cabinet or camera animation.
---

# Blender RoomKit

Use registered assets and `scripts/roomkit.py` to author editable rooms. New reference
reconstructions default to a static scene and inspection stills; animate only within
requested scope and preserve unrelated existing animation. Pi3X supplies references,
not the authored room. `demo_scene.py` is a helper example, not a starting layout.

## Workflow

| Stage | Action and output | Before advancing |
| --- | --- | --- |
| Establish basis | Use source observations and reviewed geometry/cameras via [reconstruction](references/reconstruction.md) | Preserve scope, common coordinates and source-linked dimensions |
| Author or revise | Search the [asset index](../../../assets/INDEX.md), then use [asset operations](references/assets.md) | Move complete stable-ID roots; preserve materials, joints, controls and supports |
| Review layout | Save a distinct scene and inspect individual objects against matched source views | Repair findings and recheck affected neighbors; use the inspection tools below |
| Refine appearance | Use [material operations](references/assets.md#material-behavior) and [rendering](references/workflows.md#rendering-and-resuming) after layout review | Inspect actual material/lighting renders against the source; preserve reviewed geometry/cameras and whitebox scope |
| Deliver | Save the edited `.blend`, requested stills/clips and relevant metadata | Complete [acceptance](../../../docs/WORKFLOW_ACCEPTANCE.md), actual visual review and requested video decoding/timing |

Open saved sources in background Blender and write distinct outputs. Obtain a saved
copy for required UI edits. Use Blender Python for `bpy`, with `--python-exit-code 1`
before `--python`/`--python-expr`; consume only successfully saved outputs. Canonical
semantic placement uses metres, **-Y front, Z up**; imports normalize verified source
axes. Repair within existing authorization and state remaining approximations.

## Choose the inspection tool

From the first blockout, individually inspect every main item—at least beds, tables
and shelves—plus dominant and user-flagged objects. Compare isolated X-ray focus
images with matched source and plan/side views. Record object-specific findings,
repair and recheck; a generated packet alone is not visual review.

| Need | Use |
| --- | --- |
| Visible contours or isolated authored geometry | `quick_check --object-id <exact-id>`, `object_outline`, `agent_review --all-objects` |
| Source pixels and occlusion evidence | Reviewed SAM3 masks or source landmarks |
| Correct a mismatch | Hold reviewed cameras fixed; edit complete object roots or structural planes with separate stable IDs; compare affected objects |

Follow [layout inspection](../../../docs/LAYOUT_INSPECTION.md#object-review-default-and-repair-loop)
for commands and evidence. Run [placement checks](../../../docs/PLACEMENT_CHECK.md)
after saved placement/replacement edits; reuse the layout renderer's included check.

## Load for the current operation

- Requested cabinet/camera animation: [animation](references/workflows.md#animation-json).
- Stills or clips: [rendering](references/workflows.md#rendering-and-resuming).
- Source cameras or room/person assembly: [camera operations](references/cameras.md).
- Runtime failures: [runtime](../../../kimodo_blender/RUNTIME.md) and [render settings](../../../docs/RENDERING.md).
