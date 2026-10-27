---
name: pi3x-scene-reference
description: Generate Pi3X geometry, cameras and source-linked measurements. Use for room reference views, approximate dimensions or scale calibration from video or cached predictions.
---

# Pi3X scene references

Produce reference geometry, cameras and dimensioned views for editable room authoring.
Keep observed surfaces, fitted geometry and inferred hidden bounds distinct.

## Workflow

| Stage | Action and output | Before advancing |
| --- | --- | --- |
| Select source | Continuous video shot with original frame IDs/PTS | Use 32 frames by default or the 64-frame tier across the full shot, every frame for shorter clips |
| Reconstruct and align | Same-shot predictions, reference mesh and matching cameras | Review floor/wall support; retain one shared rigid basis for points and cameras |
| Measure | Source-linked endpoints, dimensions and diagrams | Use eligible source geometry and explicit floor/scale evidence; label missing or inferred bounds |
| Deliver | Reviewed views, measurement files and camera handoff | Check source agreement, labels and coverage; state scale/floor uncertainty |

## Choose the operation

| Need | Read or run |
| --- | --- |
| New video or cached reference rebuild | [Reconstruction](references/reconstruction.md), `python -m tools.layout_inspection.build_reference` |
| Floor/up/wall axes | [Structural alignment](../../../docs/STRUCTURAL_ALIGNMENT.md); preserve an existing room basis unless migrating room and cameras together |
| Cached dimensions/diagrams | [Measurements](references/measurements.md), `scripts/measure.py` with final aligned `--cameras` |
| Surface or furniture dimensions | [Mesh measurements](references/mesh-measurements.md), [furniture measurements](references/furniture.md) |
| Author a room from the reference | [RoomKit](../blender-roomkit/SKILL.md), then [matched layout inspection](../../../docs/LAYOUT_INSPECTION.md) |
| Requested human depth/camera handoff | [Tracking and trajectory](../../../docs/HUMAN_TRACKING_AND_TRAJECTORY.md), under [motion policy](../../../docs/MOTION_CONTROLS.md) |

## Geometry and measurement contract

Default to the [3 cm TSDF core plus visible context](../../../docs/PI3X_GEOMETRY_REFERENCE.md).
Semantics and meshing consume all inference frames; display selection does not reduce
coverage. Measurements use source triangles and `measurement_valid`, excluding people;
glass, mirrors and missing geometry remain uncertain. Reuse predictions for cosmetic
changes to diagrams. SAM3 labels source pixels; native outlines show authored objects.

Retain actual source pixels, timestamps and endpoint-selection rationale; reject
unreliable depth edges. Visible extents are not complete furniture bounds, and height
requires a reviewed floor. Uniform metric correction applies to geometry, dimensions
and camera translations together, separately from the rigid transform. Pi3X scale is
predicted; assumed sizes, camera-up, confidence and reprojection alone do not establish
physical scale or a floor. Source pixel IDs record provenance, not physical tracks.

## Delivery

Provide plan/front/side views and 3–5 native source overlays (five by default), covering
distinct regions and early/middle/late video times. Once a room exists, add matched
model-edge comparisons and label the geometry shown.

For measurements, deliver `MEASUREMENTS.md`, diagrams, JSON/CSV and `cameras.json`.
Inspect arrows, endpoints, units and timestamps against retained source RGB. Sparse
camera keys require an explicit interpolation policy and playback review before
full-rate animated-camera use.
