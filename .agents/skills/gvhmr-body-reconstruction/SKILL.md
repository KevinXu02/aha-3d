---
name: gvhmr-body-reconstruction
description: Reconstruct observed people with GVHMR, shared-scene anchoring, v2 optimization and contact refinement. Use for source-person reconstruction or its stages; generated new actions use Kimodo.
---

# GVHMR source-person reconstruction

Full reconstruction includes all requested people through scene integration and
selected delivery. Stage-only, review and debugging requests retain their scope.
Reuse compatible caches unless fresh inference is requested; preserve a native
GVHMR comparison. Current defaults and execution live in the
[full pipeline](../../../docs/REAL2SIM_PIPELINE.md) and [motion policy](../../../docs/MOTION_CONTROLS.md).

## Workflow

| Stage | Inputs and output | Before advancing |
| --- | --- | --- |
| Track and infer | Source video and reviewed actor identities → masks, PMPose and native GVHMR motion | Review masks and exact source frames; [tracking operations](references/source-motion.md#actor-tracking-and-track-termination) distinguish confirmed exits from occlusion |
| Anchor and optimize | Reviewed room floor/upright and same-shot dense Pi3X depth/cameras → shared-scene placement and default v2 | Compare native and optimized bodies through the shared room camera; use [room basis](references/source-motion.md#source-motion-and-room-basis) and [v2 commands](../../../docs/WORLD_POSTOPT.md) |
| Refine contacts | Reviewed contact/release intervals and finite authored supports → corrected trajectory | [Contact refinement](references/source-motion.md#full-pipeline-contact-refinement): check support, penetration, source projection, continuity and foot slip |
| Integrate | Authored room and aligned caches → baked scene with actor visibility | [Scene import](references/source-motion.md#import-into-the-modeled-room): preserve materials, other actors, requested camera and timing |
| Deliver | Reopened final scene and requested media → accepted selected result | Check all actors over all active frames, inspect source/room mesh views, fully decode videos and verify timing under [acceptance](../../../docs/WORKFLOW_ACCEPTANCE.md) |

## Essential constraints

- Initialize with one fixed whole-clip rigid transform at **body scale 1**, followed
  by default v2. Trajectory scaling requires an explicit request. Contact translation
  may adjust root XYZ while preserving local articulation and body shape; regularize
  temporal changes. Do not modify joints to remove collisions or run human 6D leg
  repair/leg IK. Keep infeasible simultaneous contacts explicit.
- Preserve dense depth, confidence, intrinsics and frame mappings. V2's per-person
  camera is not the shared scene camera; own-camera reprojection alone cannot
  establish scene alignment.
- Confirmed exits hide actors with explicit activity metadata; temporary occlusion
  does not end a track. Resample rotations before skinning when FPS changes.
- Open the authored room in background Blender and save a distinct output. Use the
  scene importer; the pilot reference-mesh builder clears its scene.

## Conditional work

For repeated controlled comparisons use [human automation](../../../docs/HUMAN_AUTOMATION.md).
Kimodo completion stays disabled unless explicitly requested with observed-action text;
contact failures or low keypoint counts alone do not authorize replacement. For such
experiments use [completion details](references/source-motion.md#v2-and-optional-generated-motion)
and [Kimodo](../kimodo-body-motion/SKILL.md). Keep generated intervals, numerical
checks, actual visual review and remaining approximations explicit.
