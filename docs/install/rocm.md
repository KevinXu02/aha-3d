# Static, Human and Robotics levels on AMD GPUs (ROCm)

The Static level (Pi3X, SAM3 and Open3D) runs on AMD Instinct GPUs with the
PyTorch build that ships in the official
[`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch) image. `tools/setup.sh`
installs CUDA wheels, so it is not used on ROCm: Pi3X and SAM3 share the image's
Python, Open3D gets its own Python 3.11 environment, and `build_reference`
selects both through its existing runtime flags. ROCm PyTorch exposes the GPU
through the `torch.cuda` API, so no source changes are needed.

The Human level's runtimes (SAM3 tracking, GVHMR, PMPose) install in the same
image and their GPU kernels match CPU results; a full source-person
reconstruction has not been run on ROCm yet. The Robotics level's Kimodo
generates native G1 motion on the GPU, matching a CPU run started from the same
noise. Blender was not tested at any level.

## Validated configuration

| Item | Value |
| --- | --- |
| GPU | AMD Instinct MI300X, one device (`HIP_VISIBLE_DEVICES=0`) |
| Image | `rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1` (about 70 GB) |
| Pi3X / SAM3 Python | image Python 3.12, `torch 2.7.1+git99ccf24`, `torchvision 0.22.1+59a3e1f`, `triton 3.3.1`, HIP 6.4 |
| Mesh Python | Python 3.11 venv, `open3d 0.18.0` from the `reference` extra |
| Kimodo Python | venv over the image Python 3.12 (`--system-site-packages`), `transformers 5.1.0` |
| Source pins | Pi3 `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`, SAM3 `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7`, Kimodo `1aece8c124d73d255ceff5086d983b844c9f4e94` |

Python 3.12 is a deviation from the documented 3.11 Pi3X/SAM3/Kimodo baseline;
the checks below ran on it. The mesh environment stays on 3.11 because the pinned
Open3D 0.18.0 has no Python 3.12 wheels.

## Start the container

Run from the checkout root. The container keeps running so later steps reuse
the same installed environment:

```bash
docker run -d --name aha3d-rocm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD:/workspace/aha-3d" -w /workspace/aha-3d \
  rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 sleep infinity
docker exec -it aha3d-rocm bash
```

The remaining commands run inside the container. Files it writes into the
checkout are owned by root; `chown -R` them back when done.

## Install Pi3X, SAM3 and Open3D

Clone the pinned sources as the installer does, then install the dependencies
into the image Python with its Torch, torchvision and Triton held fixed:

```bash
git clone https://github.com/yyfz/Pi3.git external/Pi3
git -C external/Pi3 checkout --detach 9fa3ddb3f8d53041f8b2738df404f62223bbaa7b
git clone https://github.com/facebookresearch/sam3.git external/sam3
git -C external/sam3 checkout --detach 660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7

python -m pip list --format=freeze | grep -iE '^(torch|torchvision|triton)==' > /tmp/rocm-torch.txt
python -m pip install -c /tmp/rocm-torch.txt \
  'numpy==1.26.4' 'timm==1.0.20' 'ftfy==6.1.1' 'iopath==0.1.10' 'setuptools<81' \
  'opencv-python-headless==4.11.0.86' pillow plyfile huggingface_hub safetensors pycocotools psutil einops \
  -e external/sam3
python -m pip check
python -c 'import sys; sys.path.insert(0, "external/Pi3"); from pi3.models.pi3x import Pi3X; from sam3.model_builder import build_sam3_image_model; print("PI3X_SAM3_IMPORT_OK")'
```

`pip check` reported no broken requirements, and `torch.__version__` and
`torch.__file__` were unchanged after the install. Do not install `triton` or a
`torch` wheel on top of the image; the constraints file makes pip fail instead.

Create the Open3D environment with [uv](https://docs.astral.sh/uv/), which
fetches Python 3.11:

```bash
python -m pip install uv
uv venv --python 3.11 .runtime/pi3x-mesh/venv
uv pip install --python .runtime/pi3x-mesh/venv/bin/python -e ".[reference]"
```

## Download checkpoints

Download Pi3X as in the [Pi3X guide](pi3x.md#download-authorized-weights),
using `python` for `$PI3X_PY`. For SAM3, pass a Hugging Face token with access
to [facebook/sam3](https://huggingface.co/facebook/sam3) from the host
environment (`docker exec -e HF_TOKEN ...`) and register the runtime:

```bash
mkdir -p .runtime/sam3-segmentation
git -C external/sam3 rev-parse HEAD > .runtime/sam3-segmentation/source_revision.txt
python tools/object_pilot/segmentation/prepare_runtime.py --runtime .runtime/sam3-segmentation
```

## Build the reference

```bash
export PI3X_UPSTREAM="$PWD/external/Pi3" XFORMERS_DISABLED=1 PYTHONDONTWRITEBYTECODE=1
python -m tools.layout_inspection.build_reference --video /absolute/video.mp4 \
  --upstream "$PI3X_UPSTREAM" --checkpoint "$PWD/.runtime/pi3x/checkpoint/model.safetensors" \
  --semantic-python "$(command -v python)" --mesh-python "$PWD/.runtime/pi3x-mesh/venv/bin/python" \
  --num-frames 32 --reference-views 5 --out /absolute/new-reference
```

The [geometry guide](../PI3X_GEOMETRY_REFERENCE.md) describes the outputs and
review.

## Human-level runtimes

These steps replace the CUDA-specific parts of the
[Human installation](human-motion.md) and [GVHMR installation](gvhmr.md); body
models, weights and the pipeline commands are as described there. Each runtime
reuses the image's PyTorch through the constraints file from the Static install.

Add PyAV to the SAM3 runtime for video tracking:

```bash
python -m pip install -c /tmp/rocm-torch.txt av==12.3.0
python -c 'import av; from sam3.model_builder import build_sam3_video_model; print("SAM3_TRACKING_IMPORT_OK")'
```

GVHMR and PMPose get their own venvs created from the image Python with
`--system-site-packages`, so they see the image's PyTorch. Check that before
installing anything; otherwise pip would pull a CUDA build from PyPI.

GVHMR uses the prebuilt
[PyTorch3D ROCm wheel](https://github.com/ZJLi2013/pytorch3d/releases/tag/rocm6.4-py3.12);
a source build of PyTorch3D compiles but has no GPU kernels. DPVO and
torch-scatter are not installed, so select Pi3X camera poses with
`--camera-estimator pi3x --pi3x-bundle` (see [GVHMR](gvhmr.md#dpvo-versus-pi3x-camera-input));
that route does not import DPVO.

```bash
export GVHMR_ROOT="$PWD/.runtime/gvhmr-bedlam2"
export GVHMR_PYTHON="$GVHMR_ROOT/venv/bin/python"
git clone --recursive https://github.com/mkocabas/GVHMR_BEDLAM2.git "$GVHMR_ROOT"
git -C "$GVHMR_ROOT" checkout cac2d9dacc6b4b6f145ca02c2e6e616719fff916
git -C "$GVHMR_ROOT" submodule update --init --recursive
python -m venv --system-site-packages "$GVHMR_ROOT/venv"
"$GVHMR_PYTHON" -c 'import torch; print(torch.__file__)'   # must be the image's torch
"$GVHMR_PYTHON" -m pip install -c /tmp/rocm-torch.txt numpy==1.26.4 av==12.3.0 imageio==2.37.2 \
  imageio-ffmpeg timm==1.0.20 pytorch-lightning==2.5.5 smplx==0.1.28 \
  hydra-core hydra-zen hydra-colorlog omegaconf rich yacs einops \
  opencv-python==4.10.0.84 scipy pillow matplotlib ffmpeg-python scikit-image \
  termcolor joblib trimesh wis3d tensorboardX ipdb \
  ultralytics==8.2.42 cython_bbox lapx fvcore iopath numba pypose \
  portalocker loguru safetensors
"$GVHMR_PYTHON" -m pip install -c /tmp/rocm-torch.txt --no-build-isolation chumpy==0.70
"$GVHMR_PYTHON" -m pip install --no-deps -e "$GVHMR_ROOT"
"$GVHMR_PYTHON" -m pip install --no-deps \
  https://github.com/ZJLi2013/pytorch3d/releases/download/rocm6.4-py3.12/pytorch3d-0.7.9-cp312-cp312-linux_x86_64.whl
```

MMCV 2.2.0 builds its ops for ROCm from source when it detects a HIP PyTorch:

```bash
export BMP_ROOT="$PWD/.runtime/bmp/src"
export PMPOSE_PYTHON="$PWD/.runtime/bmp/venv/bin/python"
git clone https://github.com/MiraPurkrabek/BBoxMaskPose "$BMP_ROOT"
git -C "$BMP_ROOT" checkout --detach 49a070a6f8396147323b1c4959474077dbe2ce8e
python -m venv --system-site-packages "$PWD/.runtime/bmp/venv"
export PATH="$PWD/.runtime/bmp/venv/bin:$PATH"
"$PMPOSE_PYTHON" -m pip install -c /tmp/rocm-torch.txt setuptools wheel ninja \
  mmengine==0.10.7 numpy==1.26.4 opencv-python==4.10.0.84
MAX_JOBS=32 FORCE_CUDA=1 MMCV_WITH_OPS=1 PYTORCH_ROCM_ARCH=gfx942 \
  "$PMPOSE_PYTHON" -m pip install -c /tmp/rocm-torch.txt --no-build-isolation --no-binary=mmcv mmcv==2.2.0
"$PMPOSE_PYTHON" -m pip install -c /tmp/rocm-torch.txt mmdet==3.3.0 mmpretrain==1.2.0 \
  xtcocotools pycocotools hydra-core einops mat4py importlib_metadata \
  json_tricks munkres sparsemax==0.1.9 transformers==4.35.2 tokenizers==0.15.2
"$PMPOSE_PYTHON" -m pip install --no-deps -e "$BMP_ROOT"
```

Then apply the guarded `mmcv_maximum_version` edit from the
[Human installation](human-motion.md#separate-pmpose-environment) and run its
[verification block](human-motion.md#verify-before-a-new-run).
`PYTORCH_ROCM_ARCH=gfx942` targets MI300-series GPUs; set your GPU's
architecture. MMCV compiled in a few minutes with 32 jobs.

## Robotics-level runtime

These steps replace `tools/setup_core.sh`, which requires Python 3.11 and
installs CUDA wheels; weights, the runtime check and the pipeline are as in the
[Kimodo guide](kimodo.md). Kimodo is PyTorch plus a CPU-only C++ extension
(`motion_correction`), so nothing GPU-specific is compiled.

Kimodo gets a `--system-site-packages` venv instead of sharing the image Python
with Pi3X: it requires `av>=16.1.0`, while the SAM3 runtime there pins
`av==12.3.0`. The venv still sees the image's PyTorch and the Pi3X
dependencies. The `torch==2.7.1` pin in `constraints-core.txt` accepts the
image's `2.7.1+git99ccf24`.

```bash
git clone https://github.com/nv-tlabs/kimodo.git kimodo_blender/upstream
git -C kimodo_blender/upstream checkout --detach 1aece8c124d73d255ceff5086d983b844c9f4e94
python -m venv --system-site-packages .runtime/kimodo/venv
export KIMODO_ENV="$PWD/.runtime/kimodo/venv"
"$KIMODO_ENV/bin/python" -c 'import torch; print(torch.__file__)'   # must be the image's torch
source kimodo_blender/env.sh   # puts $KIMODO_ENV/bin first on PATH
python -m pip install -c /tmp/rocm-torch.txt -r requirements-core.txt
export PATH="$(python -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH" CMAKE_GENERATOR=Ninja
python -m pip install -c /tmp/rocm-torch.txt -c constraints-core.txt \
  -e kimodo_blender/upstream -e ".[test]"
python -m pip check
MOMENTUM_ENABLED=0 XFORMERS_DISABLED=1 PYTHONPATH="$PWD/external/Pi3:$PWD/src" \
  python -c 'import kimodo, motion_correction; from pi3.models.pi3x import Pi3X; print("SHARED_CORE_IMPORT_OK")'
```

Then download the G1 and text weights and run `check_runtime.py --model g1`
as in the [Kimodo guide](kimodo.md#authorized-g1-and-text-weights), passing the
Hugging Face token from the host (`docker exec -e HF_TOKEN ...`). Use this
`KIMODO_ENV` in every shell and in the runtime profile.

## Checks run

Input: the 237-frame `example/loop` office walk-through from
[lingbot-map](https://github.com/Robbyant/lingbot-map) encoded at 10 fps.

| Check | Result |
| --- | --- |
| Pi3X smoke (`reconstruct.py`, 32 frames) | finite predictions, rotation orthonormality error 1.3e-6, 59% of points retained; 28 s inference, 14.4 GiB peak GPU memory |
| SAM3 image smoke (`segmentation/run.py`, label "trash can" with a selection box) | text-prompt mask selected, score 0.96, box IoU 0.89 |
| `build_reference` (32 frames, 5 views) | masks for every frame, structural alignment, TSDF mesh and 5 source-view overlays; overlays inspected by eye |
| Unit tests in the mesh environment | 907 run, 30 errors, 37 skipped; identical on an unmodified `main` checkout in the same environment |

Pi3X prints `cannot find cuda-compiled version of RoPE2D, using a slow pytorch
version instead`; the PyTorch fallback was used for the timings above.

Human-level runtimes, on the same machine and image:

| Check | Result |
| --- | --- |
| PyTorch in the SAM3, GVHMR and PMPose runtimes | the image's build and path in all three, GPU visible |
| `pip check` | clean for SAM3 and GVHMR; PMPose reports `bbox-mask-pose` requires `chumpy`, which the Human guide's PMPose recipe does not install either |
| [Human verification block](human-motion.md#verify-before-a-new-run) | all imports and `world_pipeline.py prepare --help` pass |
| PyTorch3D on GPU vs CPU | `MeshRasterizer` (naive and binned) and `knn_points` give identical results |
| MMCV ops on GPU vs CPU | `nms` keeps the same 354 boxes; `roi_align` max difference 1.2e-7 |
| PyAV | decodes all 237 frames of the input video |

These checks do not load model weights or body models and do not establish
reconstruction quality.

Robotics-level runtime, on the same machine and image:

| Check | Result |
| --- | --- |
| PyTorch in the Kimodo venv | the image's build and path, GPU visible; `pip check` clean |
| `check_runtime.py --model g1` | GPU matmul finite; G1 and Llama 3 access pass; Llama 3 and both LLM2Vec snapshots at the revisions listed in the Kimodo guide, G1 at `3020ad8` |
| G1 generation with the pipeline's motion command ("A robot walks forward and waves with the right hand.", 5 s, seed 42, 100 diffusion steps) | 150 frames × 34 joints, all finite, and a 150-row MuJoCo qpos CSV; 46 s including model loading, 15.3 GiB peak GPU memory |
| Same generation on GPU and CPU from the same noise | joint positions differ by 0.51 cm on average and 1.08 cm at most |
| Kimodo-related unit tests (`test_kimodo_frame`, `test_g1_interaction`, `test_interaction_sequence`, `test_reference_motion`, `test_metric_legs`) | 56 run, OK |

Kimodo draws its diffusion noise on the device, so for the GPU/CPU comparison a
small wrapper drew it on the CPU for both runs; the timing is from that wrapped
GPU run. The G1 interaction batch (it needs RoomKit scenes), SMPL-X generation
and Blender rendering and skinning were not run.
