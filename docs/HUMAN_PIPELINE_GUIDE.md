# Human motion: working mode and debug mode

## Accepted working-mode contract

The owner clarified this contract on September 16, 2026. It supersedes the earlier presentation that mixed diagnostic experiments with normal use. This document distinguishes requested behavior from implemented interfaces; a policy update is not evidence of a new end-to-end model run.

Working mode takes a video and reconstructs **all people appearing in it** for its full requested duration. Selecting a difficult clip, a particular actor, or a matched control belongs to debug mode. Room reconstruction remains the existing Pi3X-reference and editable-Blender-room workflow.

Use these responsibility labels:

- **Fixed computation:** implemented calculations consume explicit inputs/configuration.
- **Configured computation:** a named stage runs when included in the configuration; it does not decide to add itself.
- **Agent judgment:** GPT-6 interprets source/scene semantics or reviews final results. State this explicitly instead of presenting it as a script.
- **Integration pending:** existing components do not yet implement the complete working-mode behavior. Do not claim an executable production command exists.

## Working-mode stages

| Stage | Who decides / normal behavior | Input → output | Existing implementation and boundary |
| --- | --- | --- | --- |
| 1. Discover every person | Automatic working-mode stage; no agent selection of a preferred person or test excerpt | Full source video → person tracks and initializations, including later entrants | SAM3 requires initialization. All-person discovery, initialization and association are integration work; the existing `tools/gvhmr/person_reconstruct.py` consumes one actor and supplied masks. Repeated single-actor calls alone do not establish discovery of all people. |
| 2. Track and control visibility | Automatic tracker-driven policy; no GPT-6 exit/occlusion checkpoint | SAM3 results → masks and active/inactive frame ranges | Use valid SAM3 tracking output as the operational visibility signal: hide the actor when tracking has no valid result; do not invent a visible continuation. Restoring a detected person is an association/segment operation. `tools/gvhmr/track_lifecycle.py` currently has a reviewed terminal-prefix contract, so it needs integration for arbitrary automatic visibility intervals. |
| 3. Estimate human motion | Fixed model execution for each active person segment | Masks/crops + video + same-shot Pi3X camera rotations → ViTPose keypoints and GVHMR body motion | `tools/gvhmr/person_reconstruct.py`, with `person_tracking.py` and `camera_tracks.py`. Call this estimation in working mode. Re-estimation describes a debug rerun of an existing result, not a mandatory second inference pass. |
| 4. Filter 2D observations | Fixed threshold, no GPT-6 visibility-interval authoring | ViTPose keypoint confidence → usable keypoint mask | Accept keypoints at or above a configured confidence threshold. Existing `tools/gvhmr/depth_observations.py` has `min_keypoint_score` (current default 0.5), but still requires reviewed intervals and diagnostic inputs; those dependencies must be separated for working mode. Retain numeric validity and matching actor-mask/depth checks without adding a semantic review gate. |
| 5. Place the person in the room | Fixed computation, required | Body, valid observations, Pi3X depth/cameras and common room basis → one rigid rotation/translation per person segment | `tools/gvhmr/depth_observations.py` implements fixed-size depth anchors and rigid placement. Keep body scale 1. Room authoring/basis preparation is a separate scene responsibility. |
| 6. Correct contact position | Included in the currently requested working route; no agent decision about whether extra trajectory repair is needed | Aligned body + contact events/body patches + finite room surfaces → contact-position candidate | `tools/gvhmr/align_depth_trajectory.py`, `root_constraints.py`, `full_scene_constraints.py`. Existing `smooth_contact` / `smooth_scene_contact` solve position offsets; temporal regularization may smooth that contact correction. Do not prepend an independent trajectory-scale or drift-smoothing experiment. Contact-event/target preparation still requires explicit source/scene interpretation; no fully automatic contact detector is established. |
| 7. Assemble and deliver | Scripted assembly/rendering, final agent review stated separately | Room + all people + activity intervals → saved scene, full-duration video, optional comparison page | `src/aha3d/blender/body.py`, `ensemble.py`; `.agents/skills/blender-roomkit/scripts/render_clip.py`; `tools/human_experiments/build_portal.py`. Keep saved-scene, video timing and full-decode checks. Contact and action/style checks belong after the result, not in a mandatory pre-alignment diagnostic stage. |

A missing SAM3 result is the requested **operational hide rule**. It is not a claim that tracking loss scientifically proves physical departure; working mode does not ask GPT-6 to resolve that distinction before continuing. Do not replace this rule with a mandatory manual lifecycle review. Tracker confidence/validity and segmentation/association behavior must be explicit in implementation.

Confidence thresholding is the accepted operational observation filter. It is not a calibrated guarantee of anatomical visibility. Do not reintroduce a GPT-6 review of each hip/shoulder interval under another name.

For the current scope, preserve GVHMR articulation and body size. After initial rigid placement, run contact-position correction only. Separate trajectory scaling, depth-drift smoothing, a standalone vertical-offset search, Kimodo generation/editing and joint IK are outside this working route unless requested again. A smooth contact-position curve is part of the contact solver, not a separate decision stage.

Contact priority remains **contact, then source action/style, then image alignment**. Source-relative contact targets still need a body patch, object surface and acquisition/release interval. Existing code consumes supplied declarations; this remaining interpretation must be reported as agent judgment, not silently called fully automatic. Contact-position solving does not prove correct articulation, exact contact area, friction or balance.

## Debug mode

Use debug mode for method development or investigating a reported failure. It can select one or two challenge-specific clips and actors, compare tracking or camera alternatives, inspect identity/occlusion mistakes, or evaluate optional repair branches. These operations are not mandatory working-mode steps.

| Debug task | Script / responsibility |
| --- | --- |
| Select a test actor or difficult time window | Agent judgment for the experiment |
| Re-estimate after changing a tracker or camera input | `tools/gvhmr/person_reconstruct.py`; fresh inputs and inference, not relabelling an old cache |
| Raw ViTPose reprojection/native-ground diagnostics | `tools/gvhmr/export_motion_quality.py`, `audit_native_ground.py`; debug or post-result analysis, not mandatory immediately after estimation |
| Explicit tracker/lifecycle and visible-interval audits | Source inspection plus the existing reviewed schemas in `track_lifecycle.py` and `depth_observations.py` |
| Trajectory scaling / standalone smooth drift correction / constant vertical-offset comparisons | `tools/gvhmr/align_depth_trajectory.py`; explicit experiment configuration only |
| Kimodo completion or contact editing | `tools/kimodo/complete_motion.py`, `contact_edit.py`; explicit experiment only. Require visible-part constraints throughout the edit and action text. |
| Numerical packages and actual candidate previews | `src/aha3d/workflow/human_inspection.py`, `tools/human_experiments/render_inspection.py` |
| Source / control / candidate webpage | `tools/human_experiments/build_portal.py`; comparing alternatives is a debug presentation, not a prerequisite for reconstructing a video |

Historical 32/42 experiments belong here. Four full videos rendered on September 15 show existing rejected fixed-articulation contact-position trials, not new Kimodo samples. Their remaining contact, floating and pose defects stay recorded; the new working-mode policy does not change those outcomes.

## Implemented components versus missing integration

The existing single-actor SAM3/GVHMR launcher, Pi3X camera adapter, depth-alignment mathematics, contact-position solver, Blender importer and experiment runner are implemented. The following are **not yet a validated automatic all-person pipeline**:

1. Discovering, initializing and associating every person throughout the input.
2. Feeding automatic tracker-valid intervals through motion estimation, reappearance and final visibility instead of the legacy reviewed terminal-prefix contract.
3. Using confidence-only observation selection without the depth producer's existing manual interval and diagnostic-report prerequisites.
4. Producing contact events and surfaces without the explicit semantic preparation currently required by the solver.

Do not fabricate a working `--mode working` switch or claim that a document change removed these code dependencies. The current change records the accepted workflow and separates debug interfaces; implementation and real-video validation must report their own evidence.

`bash tools/human_experiments/run.sh` can execute, resume and cache a graph of specified commands. It does not discover actors, select correction methods, create contact semantics or provide missing working-mode integration. See [runner interface](../tools/human_experiments/RUNNER.md).

The technical [tracking/trajectory guide](HUMAN_TRACKING_AND_TRAJECTORY.md) and [automation guide](HUMAN_AUTOMATION.md) preserve existing debug interfaces and evidence. Their legacy review requirements must not be presented as the accepted working-mode behavior above.
