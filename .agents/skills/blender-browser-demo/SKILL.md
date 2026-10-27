---
name: blender-browser-demo
description: Convert a saved aha3d Blender scene into an interactive browser demo with furniture dragging, cabinet controls and existing animation. Use for scene-to-web conversion and requested tabletop or chair swaps.
---

# Blender browser demo

Use the shared `tools/roomkit_browser/` implementation from the aha3d checkout.
Convert the requested saved scene; preserve its frame range/FPS and existing motion.
Source files open in background Blender and are never saved or overwritten.

## Workflow

| Stage | Action and output | Before advancing |
| --- | --- | --- |
| Prepare | Select the scene from current state and claim a new run directory | Use configured Blender and installed Node/Playwright; [setup](../../../tools/roomkit_browser/README.md#setup-and-conversion) only when needed |
| Convert | Run the shared converter below locally on CPU in the foreground | Inspect `run.json`, `discovery.json` and `validation.json` |
| Review | Inspect `orbit.png`/`plan.png` and requested-mode evidence | Check actual framing, grouping and playback; resolve failed QA in shared implementation and retry a new output |
| Deliver | Validated HTML and a reachable preview URL | Confirm HTTP success and retain the serving process; report unsupported or approximate behavior |

```bash
python3 tools/roomkit_browser/demo.py \
  --source scenes/SCENE/blender/scene.blend \
  --out runs/SCENE/browser-RUN --blender /path/to/blender
```

No per-scene config or HTML patch is needed. Automatic discovery uses semantic roots,
complete furniture groups, supported cabinet controls and shape-key/armature animation.
Preserve semantic/support IDs; unowned loose meshes remain static. Inferred grouping
is not proof of completeness. Nested/unsupported articulation and changing animated
topology fail export. Existing actor motion is replayed, not regenerated or replanned
around furniture edits. Use [scope and metadata](../../../tools/roomkit_browser/README.md#scope-and-metadata)
for exact export, material and collision limits. Failed runs retain `pipeline.log`.

## Requested modes

Load only the matching [converter reference](../../../tools/roomkit_browser/README.md) section.

| Request | Mode and evidence |
| --- | --- |
| Tabletop physics/seeded prop swaps | [`--tabletop`](../../../tools/roomkit_browser/README.md#tabletop-physics-and-seeded-swaps); inspect `tabletop-validation.json` and `tabletop-original.png`/`tabletop-final.png`. Requires explicit supported table geometry; swaps cover tabletop objects |
| Whole-chair swaps | [`--chairs`](../../../tools/roomkit_browser/README.md#constrained-chair-replacement); inspect `chair-validation.json` and `chairs-*.png`. Trusted roots preserve seat count, type, native scale, heading and current anchors; unavailable alternatives retain the layout |
| Hierarchy/layout presentation changes | [Scene graph](../../../tools/roomkit_browser/README.md#live-scene-graph-and-layout); inspect `scene-graph-validation.json` and graph screenshots. Assigned support edges do not establish physical contact |

Mode checks cover their documented proxies and samples; they do not certify continuous
contact or ergonomics. Fix reusable behavior rather than hand-editing a scene to pass QA.

## Access

Reuse an existing preview server. `--serve /absolute/server-root/scene-name.html`
atomically mirrors validated HTML; claim that exact output too. The flag does not
start a server. Otherwise serve the output directory with Python HTTP on an available
remote port and retain the process. Open the remote service address through the app;
use the actual service port, not a browser-local forwarded port. See
[results and access](../../../tools/roomkit_browser/README.md#results-and-access).
