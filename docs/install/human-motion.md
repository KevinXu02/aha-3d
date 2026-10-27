# Install the complete source-motion stack

For a new Level 2 installation, run `bash tools/setup.sh` and choose Human
(or pass `--level human` in a script) before following this guide. The selector
prepares Static and SAM3 video decoding; the GVHMR, PMPose, Blender and model
steps below complete the Human installation.

Use this with [GVHMR installation](gvhmr.md) and the
[world pipeline commands](../WORLD_POSTOPT.md). The tracked backend under
`tools/gvhmr/world_backend/` includes the owner's tracking, dense-camera,
adaptation, v2 optimization and evaluation helpers. No PromptHMR checkout,
PromptHMR weights or PromptHMR environment is required. Selected official
PyTorch3D v0.4.0 rotation utilities are bundled with their BSD license to preserve
the historical numerical behavior; this does not change the installed PyTorch3D
version needed by GVHMR. The optional
`--prompt-hmr` argument overrides this backend for legacy experiments only.

## Evidence and installation boundary

This guide transcribes the owner's 2026-09-17 `world-postopt/INSTALL.md` and
inspection of the installed runtimes. Its measured baseline was Python 3.11.14,
Torch 2.7.1+cu128, torchvision 0.22.1+cu128, one 96 GB RTX PRO 6000 Blackwell,
driver 580.159, CUDA toolkit 12.8 and GCC 12. Use your hardware's architecture
when building extensions: `12.0` below is Blackwell; use `8.9` for Ada
(RTX 4090 / RTX 6000 Ada). 24 GB cards are the target but untested.
This recipe has not been rerun as a clean installation during packaging.

There are two additional Human environments: GVHMR/dense Pi3X/v2 and PMPose.
The Static SAM3 environment provides the default tracker after adding PyAV.
The Robotics core is separate and needed only when generating new motion;
do not downgrade its PyAV. Existing environments and models can stay where
installed; pass their paths explicitly or use ignored local configuration.
Install, build, test and run inference on the local workstation with one GPU,
following [machine policy](../../MACHINE.md).

## Code locations and revisions

Start from this repository's root:

```bash
export R2S="$PWD"
export GVHMR_ROOT="$R2S/.runtime/gvhmr-bedlam2"
export GVHMR_PYTHON="$GVHMR_ROOT/venv/bin/python"
export BMP_ROOT="$R2S/.runtime/bmp/src"
export PMPOSE_PYTHON="$R2S/.runtime/bmp/venv/bin/python"
export PI3X_UPSTREAM="$R2S/external/Pi3"
```

| Library | Source and recorded identity |
| --- | --- |
| GVHMR BEDLAM2 | [upstream](https://github.com/mkocabas/GVHMR_BEDLAM2), `cac2d9dacc6b4b6f145ca02c2e6e616719fff916` |
| Pi3/Pi3X | [upstream](https://github.com/yyfz/Pi3), `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b` |
| BBoxMaskPose/PMPose | [upstream](https://github.com/MiraPurkrabek/BBoxMaskPose), installed package 2.0.0 with bundled MMPose 1.3.1, taken 2026-09-17; original commit not recorded; [smoke-tested revision](https://github.com/MiraPurkrabek/BBoxMaskPose/commit/49a070a6f8396147323b1c4959474077dbe2ce8e) `49a070a6f8396147323b1c4959474077dbe2ce8e` |

The original deployed BBoxMaskPose commit is a reproducibility limit. For a new
installation, the following uses the recorded PMPose smoke-test revision; it
does not establish that the original deployment used the same source. Keep
working installations intact and record the revision used in your validation.

```bash
# Only in a new, unused destination.
export BMP_REVISION="${BMP_REVISION:-49a070a6f8396147323b1c4959474077dbe2ce8e}"
git clone https://github.com/MiraPurkrabek/BBoxMaskPose "$BMP_ROOT"
git -C "$BMP_ROOT" checkout --detach "$BMP_REVISION"
```

## GVHMR and dense cameras

First install the dedicated environment using [GVHMR's recipe](gvhmr.md),
including SMPL-X and PyTorch3D. It keeps NumPy 1.26.4 and PyAV 12.3.0.
The author's deployed CUDA builds used toolkit 12.8.

Add the small tracking/camera dependencies to that environment:

```bash
"$GVHMR_PYTHON" -m pip install iopath portalocker loguru safetensors
```

The default tracker uses Static's SAM3 runtime with PyAV, as configured below.

Install Pi3 at the revision above following [Pi3X installation](pi3x.md).
The dense-camera helper runs in the GVHMR environment, adds `--pi3` to its import
path and loads an explicit safetensors file. Do not apply Pi3's historical Torch
2.5.1 requirements over this environment. The normal room-reference workflow
can continue using its existing shared-core interpreter.

## Separate PMPose environment

The installed PMPose stack uses the same Python/Torch pair but adds OpenMMLab.
Use a new environment; the commands below are not an upgrade recipe for an
existing one. Compiled MMCV is required; `mmcv-lite` does not supply its kernels.

```bash
python3.11 -m venv "$R2S/.runtime/bmp/venv"
"$PMPOSE_PYTHON" -m pip install --upgrade pip setuptools wheel ninja
"$PMPOSE_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$PMPOSE_PYTHON" -m pip install mmengine==0.10.7 numpy==1.26.4 opencv-python==4.10.0.84
export CUDA_HOME=/usr/local/cuda-12.8
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12
MAX_JOBS=6 FORCE_CUDA=1 MMCV_WITH_OPS=1 TORCH_CUDA_ARCH_LIST=12.0 \
  "$PMPOSE_PYTHON" -m pip install --no-build-isolation --no-binary=mmcv mmcv==2.2.0
"$PMPOSE_PYTHON" -m pip install mmdet==3.3.0 mmpretrain==1.2.0 \
  xtcocotools pycocotools hydra-core einops mat4py importlib_metadata \
  json_tricks munkres sparsemax==0.1.9 transformers==4.35.2 tokenizers==0.15.2
"$PMPOSE_PYTHON" -m pip install --no-deps -e "$BMP_ROOT"
```

The installed runtime has a local MMDetection compatibility edit: its
`mmdet/__init__.py` declares `mmcv_maximum_version = '2.3.0'`, allowing the
installed MMCV 2.2.0. This detail was absent from the original install record.
For that exact 3.3.0/2.2.0 combination in your **new PMPose environment**, the
following guarded edit reproduces the recorded bound; it does not prove general
OpenMMLab compatibility. Validate actual inference after applying it.

```bash
"$PMPOSE_PYTHON" - <<'PY'
from importlib.metadata import distribution, version
assert version('mmdet') == '3.3.0' and version('mmcv') == '2.2.0'
p = distribution('mmdet').locate_file('mmdet/__init__.py')
s = p.read_text()
old = "mmcv_maximum_version = '2.2.0'"
new = "mmcv_maximum_version = '2.3.0'"
assert s.count(old) == 1 or s.count(new) == 1, 'Unexpected mmdet source; inspect manually'
if old in s:
    p.write_text(s.replace(old, new))
PY
```

The installed BBoxMaskPose copy's `mmpose/__init__.py` allows MMCV through 2.3.0;
verify this in the revision you choose. Do not separately pip-install MMPose
over that bundled package. The project PMPose worker sets its working directory
to `$BMP_ROOT/mmpose` for relative metainfo paths, registers `mmpretrain`, permits
the trusted checkpoint's NumPy data during loading, and exports the first
17 of PMPose's 23 joints as COCO-17. It requires actual reviewed tracker masks.

If the interpreter inherits an older Conda `libstdc++` and MMCV reports missing
`GLIBCXX_3.4.32`, configure `PMPOSE_LD_PRELOAD` with a compatible system library
path for your machine. The recorded machine used
`/usr/lib/x86_64-linux-gnu/libstdc++.so.6`. The worker applies this only to the
PMPose subprocess; do not hardcode it into portable source.

## External weights and body assets

Download using your own authorized access. No weights or body models are added
to Git. The installed names expected by the wrappers are:

| Location relative to repository | Source |
| --- | --- |
| `.runtime/gvhmr-bedlam2/inputs/checkpoints/gvhmr/gvhmr_b1b2.ckpt` | BEDLAM2, as described in [GVHMR installation](gvhmr.md) |
| `.runtime/gvhmr-bedlam2/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt` | GVHMR image-feature weights |
| `.runtime/gvhmr-bedlam2/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz` | Authorized [SMPL-X](https://smpl-x.is.tue.mpg.de/) assets |
| `.runtime/gvhmr-bedlam2/inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl` | Authorized [SMPL](https://smpl.is.tue.mpg.de/) assets for GVHMR's renderer |
| `.runtime/pi3x/checkpoint/model.safetensors` | [Pi3X](https://huggingface.co/yyfz233/Pi3X), model revision `bb1deea4d7423de5b30691739cb451a3f57dc1d5` |
| `.runtime/bmp/checkpoints/PMPose-h-1.0.0.pth` | [BBoxMaskPose model repository](https://huggingface.co/vrg-prague/BBoxMaskPose/tree/main/PMPose) |

ViTPose weights are needed for the explicit `--pose-detector vitpose` route.
YOLO and DPVO are not the tracking/camera sources in the SAM3/Pi3X default route;
some GVHMR upstream imports still require installed support packages.
The body model uses `use_pca=False`, `flat_hand_mean=True`, `num_betas=10`.

## Verify before a new run

On the workstation, from the repository root:

```bash
"$GVHMR_PYTHON" -c 'import torch, pytorch3d, smplx, av, iopath, portalocker, loguru, safetensors; print(torch.__version__, torch.cuda.get_device_capability())'
PYTHONPATH="$PI3X_UPSTREAM" "$GVHMR_PYTHON" -c 'from pi3.models.pi3x import Pi3X'
(cd "$BMP_ROOT/mmpose" && PYTHONPATH="$BMP_ROOT" \
  "$PMPOSE_PYTHON" -c 'import mmcv, mmpose, mmpretrain; from pmpose import PMPose; print(mmcv.__version__, mmpose.__version__)')
"$GVHMR_PYTHON" tools/gvhmr/world_pipeline.py prepare --help
```

Apply `LD_PRELOAD="$PMPOSE_LD_PRELOAD"` to the PMPose import check only when your
runtime requires that configured workaround. Import checks do not exercise GPU
kernels or establish model quality. Follow the [pipeline](../WORLD_POSTOPT.md)
to allocate a new claimed run, prepare the complete graph, execute it, and inspect
tracking identity, detector consumption, representative source overlays and full
video timing. Existing experiment snapshots retain their original source files.

## SAM3 tracking with PMPose (current default)

Reuse the installed SAM3 checkpoint/runtime through `SAM3_ROOT`, `SAM3_PYTHON`
and `SAM3_CHECKPOINT`. The main graph runs tracking in that interpreter and
PMPose in `PMPOSE_PYTHON`. The SAM3 frontend runs on the local GPU.
The [reference installer](../PI3X_GEOMETRY_REFERENCE.md#runtime) checks SAM3's
image API for `build_reference`; source-person tracking also decodes the full
video with PyAV. Add it to that same standalone SAM3 runtime before running the
motion graph:

```bash
export SAM3_ROOT="$PWD/external/sam3"
export SAM3_PYTHON="$PWD/.runtime/sam3-segmentation/venv/bin/python"
export SAM3_CHECKPOINT="$PWD/.runtime/sam3-segmentation/checkpoints/sam3.pt"
"$SAM3_PYTHON" -m pip install av==12.3.0
"$SAM3_PYTHON" -c 'import av; from sam3.model_builder import build_sam3_video_model; print("SAM3_TRACKING_IMPORT_OK")'
```

Install and check PMPose and GVHMR in their separate environments using the
sections above. Blender and its SMPL-X skinning setup remain required for the
[final authored scene](blender.md).

The original PMPose smoke test used BBoxMaskPose revision
`49a070a6f8396147323b1c4959474077dbe2ce8e`, MMCV 2.2.0 with actual CUDA
NMS, and completed 260-frame PMPose-h inference with the original bbox policy
and SAM3 masks. The additional packages in the PMPose command above were needed
for bundled MMPose/MMPretrain compatibility. Keep these pins out of the Kimodo
runtime. Add the new environment's bin directory to PATH so MMCV finds ninja
for parallel compilation.

Runtime success is separate from target-pose accuracy: the clip32 pilot still
showed distractor contamination under heavy occlusion. No 3D acceptance follows
from these environment or detector checks.
