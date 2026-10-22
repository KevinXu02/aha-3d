# Installation levels

Choose a level by the output you need. A new video with an editable static room
uses Static. Source-person reconstruction adds Human. Robotics motion generation
extends Static with Kimodo. Human and Robotics can be combined when both
capabilities are requested.
Model weights, body assets and Blender binaries are acquired separately from
Python packages. Keep the checkout intact.

| Level | Install | What it enables |
| --- | --- | --- |
| **1. Static 3D scene** | Pi3X, SAM3 segmentation, Open3D and compatible Blender; Pi3X/SAM3 checkpoints | First-video `build_reference`, reviewed cameras/geometry and an editable room. No GVHMR, PMPose or Kimodo. [Reference setup](PI3X_GEOMETRY_REFERENCE.md#runtime), [Blender](install/blender.md) |
| **2. Human reconstruction** | Static plus SAM3 tracking/PyAV, PMPose with MMCV, GVHMR with PyTorch3D, body models/weights and Blender SMPL-X skinning | Default source-person motion, v2, contact refinement and baked scene delivery. [Motion stack](install/human-motion.md), [GVHMR](install/gvhmr.md) |
| **3. Robotics motion generation** | Static plus Kimodo, the native G1 checkpoint, text models and G1 mesh dependencies | Native robot motion generation in an editable room. GVHMR/PMPose are needed only when source-person reconstruction is also requested. [Kimodo](install/kimodo.md) |

SAM 3D Body and DINOv3 are optional sparse video-pose guidance for Kimodo, not
part of any level by default. SAMURAI is an explicit alternative tracker. The
[full video-to-scene route](REAL2SIM_PIPELINE.md) describes stage order and
validation. GVHMR keeps a separate environment because its validated PyAV pin
conflicts with Kimodo's shared core.

## Install the selected level

All components install on one Linux workstation (Ubuntu or WSL2) with one NVIDIA
GPU; see the [runtime policy](../MACHINE.md). On AMD GPUs, install the Static
level and the Human- and Robotics-level runtimes with the [ROCm guide](install/rocm.md) instead. Run the supported installer from
the checkout root:

```bash
bash tools/setup.sh
```

It **asks which Level (1/2/3) to install before running any component setup**.
For scripts or CI without a terminal, answer explicitly with `--level static`,
`--level human` or `--level robotics`; omitting the level fails before changes.
Use `--plan` to review the selected route without installing. No level is
silently selected.

The installer prepares Python code. For Static, also install [Blender](install/blender.md)
and Pi3X/SAM3 checkpoints. For Human, the selector prepares Static and SAM3
video decoding, then directs you through the separate
[PMPose/GVHMR environments](install/human-motion.md), Blender SMPL-X extension,
body assets and weights; it does not claim Level 2 is complete before those
steps. For Robotics, it adds Kimodo to Static's Pi3X environment; follow
[its guide](install/kimodo.md) for native G1/text weights and Blender. Each
guide has smoke checks.

Use an isolated environment; do not upgrade another project's Torch or replace
an existing body add-on to make a new setup fit.

For Robotics motion recipes, register the installed executables and G1 model
with `tools/configure_runtime.py` as shown in
[runtime setup](SETUP.md#5-configure-the-runtime), source `kimodo_blender/env.sh`,
then run `python kimodo_blender/check_runtime.py --model g1`. For Human, follow
the motion stack's own verification. Static reference adapters take explicit
runtime paths and checkpoints. Run the unit tests for the selected checkout.
The tracked runtime JSON is a template, not an installed environment.

## What the checks establish

Environment/version and CPU import checks verify setup and adapter interfaces.
Loading a checkpoint or running a GPU smoke example provides separate evidence;
neither establishes reference fidelity, motion quality or whole-clip consistency.
The guides distinguish historical local validation from newly documented install
steps. Writing this guide did not run the new installer or download weights. The
bundle's executed checks are listed in [validation](VALIDATION.md).

Model/body assets are obtained through the recipient's authorized upstream access.
Store authentication outside the checkout and logs. See
[distribution status](../DISTRIBUTION.md) and [external dependencies](../THIRD_PARTY.md).
