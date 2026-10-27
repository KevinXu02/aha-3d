# AHa-3D: Agentic Tool Use for Real2Sim with GPT-6 Astra

**[Project page](https://kevinxu02.github.io/real2sim-indoor-site/)** ·
[Getting started](docs/GETTING_STARTED.md) · [Documentation index](docs/INDEX.md)

AHa-3D reconstructs indoor videos as editable Blender scenes, with room geometry,
furniture, recovered cameras and optional people. Models estimate geometry and
motion; a coding agent authors the room, reviews intermediate results and refines
contacts. Saved scenes can be rendered or exported as interactive browser demos.

Start by giving your coding agent a reconstruction request. It handles setup,
geometry estimation, scene authoring, review and delivery.

| Original video | Astra without the harness | Astra with the harness |
| --- | --- | --- |
| [![Original video](https://kevinxu02.github.io/real2sim-indoor-site/media/motivation-office48/source.jpg)](https://kevinxu02.github.io/real2sim-indoor-site/#kitchen-refinement) | [![Astra without the harness](https://kevinxu02.github.io/real2sim-indoor-site/media/motivation-office48/before.jpg)](https://kevinxu02.github.io/real2sim-indoor-site/#kitchen-refinement) | [![Astra with the harness](https://kevinxu02.github.io/real2sim-indoor-site/media/motivation-office48/after.jpg)](https://kevinxu02.github.io/real2sim-indoor-site/#kitchen-refinement) |

[Play the synchronized video comparison](https://kevinxu02.github.io/real2sim-indoor-site/#kitchen-refinement).

## News

- **2026-10-27** — Added an [office reconstruction walkthrough](https://github.com/KevinXu02/aha-3d/blob/main/docs/GETTING_STARTED.md), with a bundled source video, agent prompts and Gallery/X-ray refinement. For the previous code, see the [legacy branch](https://github.com/KevinXu02/aha-3d/tree/legacy).

## Getting started

Follow [Getting started](docs/GETTING_STARTED.md) to explore the office example
in the Gallery, ask the agent to reconstruct it, and request changes. It provides
copyable prompts; the agent runs the workflow.

<details>
<summary>Technical reference: setup, commands and troubleshooting for agents and developers</summary>

## Setup

Run from the repository root on Linux (Ubuntu or WSL2), with Python 3.11,
FFmpeg and an NVIDIA GPU/driver compatible with the installer’s CUDA 12.8 wheels.
Use Blender 5.2 for the supplied scene libraries; human skinning additionally
uses Blender 4.5 and its SMPL-X extension. See [prerequisites](docs/SETUP.md#1-prerequisites).
A 24 GB GPU is the target; prior validation used 96 GB, so smaller-card coverage
and runtime are not guaranteed. See [runtime limits](MACHINE.md).

Choose the installation route for your task:

| Task | Installer level | Additional setup |
| --- | --- | --- |
| Room/reference geometry and cameras | `static` | [Pi3X weights](docs/install/pi3x.md#download-authorized-weights), [SAM3 weights/runtime](docs/PI3X_GEOMETRY_REFERENCE.md#runtime), [Blender](docs/install/blender.md) |
| Reconstruct people from the source video | `human` | Static plus [PMPose, GVHMR, body assets and skinning](docs/install/human-motion.md) |
| Generate new robot motion | `robotics` | Static plus [Kimodo, G1 and text models](docs/install/kimodo.md) |

For the reference-inference example below:

```bash
bash tools/setup.sh --level static --plan  # Inspect without installing
bash tools/setup.sh --level static
```

The installer prepares code and Python environments. **Download the Pi3X and SAM3
weights using the links above before inference**; weights and Blender are separate.
Human setup also needs the separate PMPose/GVHMR installation steps. Generated
human actions use Kimodo's SMPL-X model, described in the Kimodo guide.

Default runtime locations are under `.runtime/`; use [setup](docs/SETUP.md) for
custom paths and keep machine configuration in ignored local files. Model access
and distribution terms are listed in [third-party dependencies](THIRD_PARTY.md).

## Inference: build a geometry reference

Entry point: `python -m tools.layout_inspection.build_reference`.
It runs Pi3X, SAM3 masks, structural alignment, meshing and source-camera overlays.
Use **one continuous video shot** with at least two frames and valid decoder
timestamps. Automatic shot splitting is not implemented. The source-person
world pipeline has additional [input normalization requirements](docs/REAL2SIM_PIPELINE.md#1-establish-source-timing-and-scope).

### Minimal example

After Static setup and both model downloads, run the example video or
replace `--video` with your own footage. These commands assume the default
installation locations. In a shared checkout,
[claim the output path](docs/COORDINATION.md#commands) before running.

```bash
export PI3X_PY="$PWD/.runtime/pi3x-inference/venv/bin/python"
export PI3X_MESH_PY="$PWD/.runtime/pi3x-mesh/venv/bin/python"
export PI3X_CHECKPOINT="$PWD/.runtime/pi3x/checkpoint/model.safetensors"

"$PI3X_PY" -m tools.layout_inspection.build_reference \
  --video references/office48/source.mp4 \
  --checkpoint "$PI3X_CHECKPOINT" \
  --out runs/office48_tutorial/reference-001
```

Choose a **new** output directory; the command creates it and refuses overwrites.
The default Pi3 checkout is `external/Pi3`; SAM3 uses
`.runtime/sam3-segmentation` with its prepared `runtime.json` and checkpoint.

| Argument | Required for a new video? | Meaning/default |
| --- | --- | --- |
| `--video PATH` | Yes | Input video; mutually exclusive with cached `--bundle` |
| `--checkpoint PATH` | Yes | Explicit Pi3X model file; setting the environment variable alone does not pass it |
| `--out PATH` | Yes | New output directory |
| `--num-frames 32\|64` | No | Default 32, spanning both shot endpoints; shorter clips use all frames |
| `--pixel-limit N` | No | Inference pixel budget; default 255000 |
| `--reference-views 3\|4\|5` | No | Native-camera overlay count; default 5 |

For custom installs, use `--upstream`, `--pi3x-python`, `--semantic-runtime`,
`--semantic-python` and `--mesh-python`. Get all options with
`"$PI3X_PY" -m tools.layout_inspection.build_reference --help`.
[The geometry guide](docs/PI3X_GEOMETRY_REFERENCE.md) covers complete-cache reuse.

### Expected outputs

```text
runs/office48_tutorial/reference-001/
├── pi3x/                 Frames, predictions, input provenance and initial cameras
├── masks/                Per-frame semantic masks
├── alignment/            Proposed structural basis and aligned cameras
├── mesh/                 Reference geometry layers and manifest
├── source_views/         Native-camera overlays and view manifest
└── reference_build.json  Final artifact paths and build status
```

The report status is `built; visual review required`. Open the source overlays
and review geometry/alignment before authoring the room. Use the camera path in
`reference_build.json` downstream. This command produces no final room `.blend`
or rendered scene video; inferred dimensions and hidden geometry remain approximate.

## Reconstruct a complete scene

Start with the [office reconstruction prompt](docs/GETTING_STARTED.md#2-ask-the-agent-to-reconstruct-it).
The agent records scope and acceptance, executes the requested stages and selects
the final delivery. See [other tasks](docs/GETTING_STARTED.md#4-try-another-task)
for source people, materials and browser demos.

### Run an existing recipe

The scene CLI is `bash tools/indoor`. Select an installed project environment
before using it; after default Static setup:

```bash
export KIMODO_ENV="$PWD/.runtime/pi3x-inference/venv"
source kimodo_blender/env.sh
bash tools/indoor --help
bash tools/indoor scenes
```

For an **already authored** scene and configured recipe runtime:

```bash
bash tools/indoor plan SCENE_ID --recipe RECIPE_NAME
bash tools/indoor run SCENE_ID --recipe RECIPE_NAME
bash tools/indoor status --scene SCENE_ID --live
```

Replace both placeholders with your scene ID and recipe name. `SCENE_ID` and
`--recipe` are required; the recipe supplies inputs, timing, stages and runtime.
A clean clone does not contain your authored scene. The CLI's `intake` command
records structured requirements; it does not parse a chat prompt or launch
inference. See [recipes and resume](docs/PIPELINE.md), or the separate
[generated-human-motion example](examples/README.md#new-motion-workspace), which
requires Kimodo SMPL-X and skinning setup.

## Common troubleshooting

| Symptom | Next step |
| --- | --- |
| `Runtime interpreter not found` / `Set KIMODO_ENV` | Check the installed interpreter paths. Set the reference runtime flags or `KIMODO_ENV` for `tools/indoor`; see [shell setup](docs/SETUP.md#3-per-shell-environment). |
| Missing checkpoint, SAM3 access error or missing `runtime.json` | Finish the separate model downloads and SAM3 runtime preparation linked under Setup. |
| CUDA unavailable or out of memory | Check `nvidia-smi` and the selected environment. Use 32 frames and reduce `--pixel-limit` if needed; record the change. See [Pi3X troubleshooting](docs/install/pi3x.md#troubleshooting). |
| Output directory already exists | Choose a new reference output path. For an existing recipe run, use [resume](docs/PIPELINE.md#resume-and-reuse). |
| Invalid timestamps, incomplete decoding or poor alignment | Inspect the source shot and decoder output; review source overlays and [alignment diagnostics](docs/STRUCTURAL_ALIGNMENT.md). |
| A recipe pauses for layout/preview review | Read `run.json` and stage logs, complete the requested visual review, then resume via the [review workflow](docs/SCENE_WORKFLOW.md#submission-preflight-and-full-clip-preview-acceptance). |

The example arguments above match the code interfaces. Documentation checks are
separate from a fresh GPU run: see [validation coverage](docs/VALIDATION.md).
There is no published time/VRAM benchmark for this exact first-run example.

</details>

## Which documentation should I read?

| Audience | Start here | Read as needed |
| --- | --- | --- |
| Users | This README, [Getting started](docs/GETTING_STARTED.md) | [Installation](docs/INSTALLATION.md), [setup](docs/SETUP.md), [reference inference](docs/PI3X_GEOMETRY_REFERENCE.md), [examples](examples/README.md), [results](docs/RESULTS.md) |
| Developers | [Git workflow](docs/GIT_WORKFLOW.md), [coordination](docs/COORDINATION.md) | [Pipeline](docs/PIPELINE.md), `src/aha3d/`, `tools/`, `tests/`; run `python tools/verify_package.py` for source/package checks |
| Agents | [AGENTS.md](AGENTS.md), [task routing](docs/INDEX.md), [skill index](TOOLS_AND_SKILLS.md) | The current skill under `.agents/skills/`, [scene workflow](docs/SCENE_WORKFLOW.md), [acceptance](docs/WORKFLOW_ACCEPTANCE.md) |

Use the [documentation index](docs/INDEX.md) for specific operations such as
asset replacement, materials, motion refinement and rendering. The
[asset catalog](assets/INDEX.md) lists reusable furniture and materials.
Historical reports in `docs/lessons/` provide context, not current execution defaults.

## License

Project-authored code, skills and assets are licensed under the
[Apache License 2.0](LICENSE). See [distribution notes](DISTRIBUTION.md) and
[third-party dependencies](THIRD_PARTY.md) for separately acquired models and assets.
