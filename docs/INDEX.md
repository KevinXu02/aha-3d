# Documentation index

Read [coordination](COORDINATION.md) once at task start, then choose the active
stage below. This index is a router, not a required reading list. Existing runtime
configuration does not require rereading installation guides.

| Active task | Read now |
| --- | --- |
| Code or documentation | [Git workflow](GIT_WORKFLOW.md), the affected interface/tests |
| New or resumed reconstruction | [Acceptance intake](WORKFLOW_ACCEPTANCE.md#declare-scope-before-downstream-work), current scene/task state, [scene skill](../.agents/skills/indoor-scene-workflow/SKILL.md) |
| Room, furniture or material edit | [Blender RoomKit](../.agents/skills/blender-roomkit/SKILL.md); its references only for the operation being performed |
| Source-person reconstruction | [Motion policy](MOTION_CONTROLS.md), [GVHMR skill](../.agents/skills/gvhmr-body-reconstruction/SKILL.md) |
| Launch or resume a long job | [Execution policy](RENDER_EXECUTION.md), [runtime policy](../MACHINE.md) |
| Publish a scene/demo | [Acceptance](WORKFLOW_ACCEPTANCE.md), [results](RESULTS.md) |
| Configure or repair an environment | [Setup](SETUP.md), the relevant component's installation guide |

Each policy has one maintained authority: ownership in coordination, synchronization
in Git workflow, job waits in render execution, motion defaults in motion controls,
and acceptance in workflow acceptance. Skills link to these policies; historical
lessons and experiment reports do not change defaults. User instructions take
precedence. When editing a policy, update conflicting entrypoints in the same change.

## Reference lookup

| Need | Reference |
| --- | --- |
| Develop, run and synchronize the fixed Git checkout | [Git workflow](GIT_WORKFLOW.md) |
| Browse assets, skills, pipeline and videos | [Generated catalog](catalog/README.md), local [HTML interface](catalog/index.html) |
| Browse scene results and batch-export demos | [Scene results and Gallery](RESULTS.md) |
| Core models in one environment | [Shared installation](install/core.md) |
| Where to put your own reference footage | [References](../references/README.md) |
| Generic input files and first commands | [Examples](../examples/README.md) |
| Install only the components you need | [Component installation](INSTALLATION.md) |
| Project skills | [Skill index](../TOOLS_AND_SKILLS.md) |
| Reusable source-motion contact repair | [Contact refinement](CONTACT_REFINEMENT.md) |
| Current scene/human pipeline and where GPT-6 participates | [Scripts and GPT-6 review](HUMAN_PIPELINE_GUIDE.md) |
| Actor exits, fixed-body trajectory and contact constraints | [Human tracking and trajectory](HUMAN_TRACKING_AND_TRAJECTORY.md) |
| Resumable human experiments and automatic inspection | [Human automation](HUMAN_AUTOMATION.md), [Runner interface](../tools/human_experiments/RUNNER.md) |
| Experimental bounded trajectory optimization | [Trajectory refinement](TRAJECTORY_REFINEMENT.md) |
| TSDF core, retained background, measurements and native-view overlays | [Pi3X geometry reference](PI3X_GEOMETRY_REFERENCE.md) |
| One-command room collision, support and stability report | [Placement check](PLACEMENT_CHECK.md) |
| Common-frame room/reference overlays and semantic uncertainty | [Layout inspection](LAYOUT_INSPECTION.md) |
| Inspect misplaced furniture with simplified silhouettes and per-object highlights | [Object review entrypoint](SCENE_WORKFLOW.md#furniture-layout-review-entrypoint) |
| Video-to-room stages | [Scene workflow](SCENE_WORKFLOW.md) |
| Record requested appearance/motion/timing | [Scene requests](SCENE_REQUESTS.md) |
| Enforce task scope, evidence, findings and agent completion | [Workflow acceptance](WORKFLOW_ACCEPTANCE.md) |
| Recipes, immutable snapshots, stage resume | [Pipeline](PIPELINE.md) |
| Workbench / EEVEE previews and explicit Cycles quality | [Rendering](RENDERING.md) |
| Wait for rendering and review completed outputs | [Render execution](RENDER_EXECUTION.md) |
| Motion guidance and approximation limits | [Motion controls](MOTION_CONTROLS.md) |
| Reusable furniture/material discovery | [Assets](../assets/README.md) |
| Seeded PBR textures and material comparison | [Procedural materials](PROCEDURAL_MATERIALS.md) |
| Asset placement and material/model variants | [Variants](SCENE_VARIANTS.md) |
| Front direction and mounting points | [Orientation](ASSET_ORIENTATION.md) |
| Seating and foliage quality | [Quality](ASSET_QUALITY.md) |
| Optional SAM3 object segmentation | [Object segmentation](OBJECT_GENERATION.md) |
| Mesh caches and synthetic track schema | [Integration](../kimodo_blender/INTEGRATION.md) |
| Blank state/handoff documents | [Templates](templates/STATE.md), [handoff](templates/HANDOFF.md) |
| Checks performed on this bundle | [Validation](VALIDATION.md) |

The historical lessons retain methodological observations but exclude their
scene files and videos. Missing evidence references are recorded explicitly in
`omitted_references.json`; do not treat these historical reports as a fresh run.
