# Static level on AMD GPUs (ROCm)

The Static level (Pi3X, SAM3 and Open3D) runs on AMD Instinct GPUs with the
PyTorch build that ships in the official
[`rocm/pytorch`](https://hub.docker.com/r/rocm/pytorch) image. `tools/setup.sh`
installs CUDA wheels, so it is not used on ROCm: Pi3X and SAM3 share the image's
Python, Open3D gets its own Python 3.11 environment, and `build_reference`
selects both through its existing runtime flags. ROCm PyTorch exposes the GPU
through the `torch.cuda` API, so no source changes are needed.

Human and Robotics levels (PMPose/MMCV, GVHMR/PyTorch3D, Kimodo) and Blender
were not tested on ROCm.

## Validated configuration

| Item | Value |
| --- | --- |
| GPU | AMD Instinct MI300X, one device (`HIP_VISIBLE_DEVICES=0`) |
| Image | `rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1` (about 70 GB) |
| Pi3X / SAM3 Python | image Python 3.12, `torch 2.7.1+git99ccf24`, `torchvision 0.22.1+59a3e1f`, `triton 3.3.1`, HIP 6.4 |
| Mesh Python | Python 3.11 venv, `open3d 0.18.0` from the `reference` extra |
| Source pins | Pi3 `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`, SAM3 `660a5e9e1b8b4c02c0ad97229b88a09a6e4ff5b7` |

Python 3.12 is a deviation from the documented 3.11 Pi3X/SAM3 baseline; the
checks below ran on it. The mesh environment stays on 3.11 because the pinned
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
