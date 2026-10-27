# Install GVHMR BEDLAM2

For a new Level 2 installation, first run `bash tools/setup.sh` and choose
Human (or pass `--level human` in a script). Then follow these GVHMR steps and
the [full Human guide](human-motion.md).

Use [the GVHMR skill](../../.agents/skills/gvhmr-body-reconstruction/SKILL.md) for
source-video human reconstruction. Room authoring still uses [Blender](blender.md)
with [Pi3X](pi3x.md) references. Kimodo is for new/changed actions or fallback.

For default SAM3 tracking, PMPose, dense Pi3X and packaged v2 helpers, continue
with [full human-motion installation](human-motion.md).

## Runtime boundary

Use a separate environment. The validated local stack is Python 3.11, Torch
2.7.1+cu128, torchvision 0.22.1+cu128, NumPy 1.26.4, PyTorch3D 0.7.9, PyAV 12.3.0,
imageio 2.37.2, timm 1.0.20 and pytorch-lightning 2.5.5. Kimodo requires newer
PyAV; do not install this pin into the shared core environment. `setup_core.sh`
does not install GVHMR. A standalone environment also avoids inheriting local
SAM runtime paths. Model weights and licensed SMPL/SMPL-X assets stay external.

The original pilot used an RTX PRO 6000 Blackwell GPU, GCC 12 and a CUDA 12.x
toolkit with cu128 Torch. Build CUDA extensions for your GPU architecture: `8.9`
for Ada (RTX 4090 / RTX 6000 Ada), `12.0` for Blackwell. Install, build and run
everything on the local workstation following [MACHINE.md](../../MACHINE.md).
The launcher and instrumentation have offline contract tests; a new machine
still requires pretrained runtime and result validation.

These commands document a portable reconstruction of the validated stack. The
original environment was layered over existing installations; a clean install
of this complete recipe has not been executed as part of this publication.
A 2026-09-13 workstation experiment installed an isolated runtime with CUDA 12.8
and GCC 12, exercised actual CUDA raster/KNN/scatter kernels, and completed
DPVO and Pi3X/GVHMR video inference. It does not validate every machine.

## Code and Python packages

From the repository root, with Python 3.11, git, FFmpeg, unzip, a compatible C++
compiler and CUDA toolkit available:

```bash
export PROJECT_ROOT="$PWD"
export GVHMR_ROOT="$PROJECT_ROOT/.runtime/gvhmr-bedlam2"
git clone --recursive https://github.com/mkocabas/GVHMR_BEDLAM2.git "$GVHMR_ROOT"
git -C "$GVHMR_ROOT" checkout cac2d9dacc6b4b6f145ca02c2e6e616719fff916
git -C "$GVHMR_ROOT" submodule update --init --recursive
python3.11 -m venv "$GVHMR_ROOT/venv"
export GVHMR_PYTHON="$GVHMR_ROOT/venv/bin/python"
export PATH="$GVHMR_ROOT/venv/bin:$PATH"
"$GVHMR_PYTHON" -m pip install --upgrade pip setuptools wheel ninja
"$GVHMR_PYTHON" -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
"$GVHMR_PYTHON" -m pip install numpy==1.26.4 av==12.3.0 imageio==2.37.2 \
  imageio-ffmpeg timm==1.0.20 pytorch-lightning==2.5.5 smplx==0.1.28 \
  hydra-core hydra-zen hydra-colorlog omegaconf rich yacs einops \
  opencv-python==4.10.0.84 scipy pillow matplotlib ffmpeg-python scikit-image \
  termcolor joblib trimesh wis3d tensorboardX ipdb \
  ultralytics==8.2.42 cython_bbox lapx fvcore iopath numba pypose
"$GVHMR_PYTHON" -m pip install --no-build-isolation chumpy==0.70
"$GVHMR_PYTHON" -m pip install --no-deps -e "$GVHMR_ROOT"
```

Do not install upstream `requirements.txt` over this stack: it pins the older
Torch 2.3 / CUDA 12.1 / Python 3.10 PyTorch3D wheel. See the
[pinned upstream installation notes](https://github.com/mkocabas/GVHMR_BEDLAM2/blob/cac2d9dacc6b4b6f145ca02c2e6e616719fff916/docs/INSTALL.md)
for that alternative; it is not the tested Blackwell stack.

## Native extensions and moving-camera odometry

Set `CUDA_HOME` to your real toolkit location (for example `/usr/local/cuda-12.8`)
and put its `bin` on `PATH`; select GCC 12 with `CC`/`CXX` if it is not the default.
Keep build directories on a local filesystem (inside WSL2, under the Linux home
directory rather than `/mnt/c`).

```bash
export TORCH_CUDA_ARCH_LIST=12.0   # 8.9 for Ada
export MAX_JOBS=6
export FORCE_CUDA=1
"$GVHMR_PYTHON" -m pip install --no-build-isolation --no-deps \
  'git+https://github.com/facebookresearch/pytorch3d.git@0a7d4c1a171e8b768c63f15b17564f9ad495f49b'
"$GVHMR_PYTHON" -m pip install --no-build-isolation --no-deps \
  --no-binary=torch-scatter torch-scatter==2.1.2

git -C "$GVHMR_ROOT/third-party/DPVO" checkout c0c5a104c9c58663aa9be62c3f125d5b52874f3e
git -C "$GVHMR_ROOT/third-party/DPVO" submodule update --init --recursive
curl -L --fail https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
  -o "$GVHMR_ROOT/third-party/DPVO/eigen-3.4.0.zip"
unzip "$GVHMR_ROOT/third-party/DPVO/eigen-3.4.0.zip" \
  -d "$GVHMR_ROOT/third-party/DPVO/thirdparty"
"$GVHMR_PYTHON" tools/gvhmr/patch_dpvo.py "$GVHMR_ROOT/third-party/DPVO"
"$GVHMR_PYTHON" -m pip install --no-build-isolation --no-deps \
  "$GVHMR_ROOT/third-party/DPVO"
export PYTHONPATH="$GVHMR_ROOT/third-party/DPVO:$GVHMR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

The DPVO patch changes scalar-dispatch API calls for Torch 2.7, not the algorithm.
`FORCE_CUDA=1` is necessary when GPUs are hidden during compilation: without it,
PyTorch3D and torch-scatter can successfully build CPU-only wheels. Exercise a
CUDA rasterization/scatter operation after installation; successful imports alone
do not establish that those kernels exist.
Keep its source directory on `PYTHONPATH`: its built wheel omits the
`dpvo.loop_closure` namespace. The launcher supplies this automatically to inference.
The original Rocky 8 runtime required a source-built torch-scatter because the
available binary wheel required a newer glibc; on current Ubuntu the source build
is still the tested route. PyAV 18 failed imageio video writing;
the validated 12.3.0 pin avoids that failure without changing Kimodo's environment.

## Body assets and weights

Obtain raw body models through your authorized [SMPL](https://smpl.is.tue.mpg.de/)
and [SMPL-X](https://smpl-x.is.tue.mpg.de/) access. An installed Blender add-on
`.blend` is not a substitute for the raw files used by GVHMR. Do not commit these
assets, checkpoint files or credentials. Keep upstream terms with external assets.

Place files under `$GVHMR_ROOT/inputs/checkpoints/`:

```text
body_models/smpl/SMPL_NEUTRAL.pkl
body_models/smplx/SMPLX_NEUTRAL.npz
gvhmr/gvhmr_b1b2.ckpt
hmr2/epoch=10-step=25000.ckpt
vitpose/vitpose-h-multi-coco.pth
yolo/yolov8x.pt
dpvo/dpvo.pth
```

Get preprocessing weights from the links in the
[pinned upstream setup](https://github.com/mkocabas/GVHMR_BEDLAM2/blob/cac2d9dacc6b4b6f145ca02c2e6e616719fff916/docs/INSTALL.md).
Download the BEDLAM1+2 checkpoint from the official BEDLAM2 host:

```bash
mkdir -p "$GVHMR_ROOT/inputs/checkpoints/gvhmr"
curl -L --fail https://download.is.tue.mpg.de/bedlam2/ml/videos/gvhmr_b1b2.ckpt \
  -o "$GVHMR_ROOT/inputs/checkpoints/gvhmr/gvhmr_b1b2.ckpt"
```

Use only trusted model files: legacy SMPL and some checkpoints contain Python
pickle data. `demo_entry.py` confines NumPy/Chumpy compatibility shims to the child
process; the launcher enables trusted legacy checkpoint loading there.

## Check, run, inspect

Use a fresh claimed output directory, and inspect the source shot before choosing
static versus moving camera. On the workstation:

```bash
bash tools/gvhmr/run.sh check --camera moving
"$GVHMR_PYTHON" -c 'import torch,pytorch3d,dpvo,smplx,av; print(torch.__version__,torch.cuda.is_available(),av.__version__)'
bash tools/gvhmr/run.sh run --camera moving \
  --video references/unassigned/my_clip.mp4 \
  --output runs/my-scene/gvhmr-first
"$GVHMR_PYTHON" tools/gvhmr/inspect_result.py --help
```

`check` verifies expected paths/modules/revision, not weight loading or result
quality. Inference normalizes to 30 fps, preserves original source timing metadata,
and checks complete output decoding. The final acceptance requires source overlays,
whole-clip identity/action review, room alignment and saved-scene validation.
Follow the [workflow guide](../GVHMR_BEDLAM2.md) and skill for those stages.
Contact refinement is low priority unless the task explicitly requires it.

### GPU selection

The launcher runs on the local GPU. Optional `--gpu N` selects a physical device:

```bash
bash tools/gvhmr/run.sh check --gpu 0 --camera moving
bash tools/gvhmr/run.sh run --gpu 0 --camera moving \
  --video /absolute/source-video.mp4 --output runs/my-scene/gvhmr-first-gpu0
```

The launcher sets `CUDA_VISIBLE_DEVICES` before preflight/model imports and
records hostname, run ID and device selection. Without `--gpu` it keeps an
existing `CUDA_VISIBLE_DEVICES`, otherwise uses device 0 with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`. With an existing numeric mask, the requested
index must occur in that mask and its original device ordering is preserved.
Empty and UUID/MIG masks are rejected with `--gpu` because a physical numeric
index cannot be safely mapped through them. Use one GPU for this workflow.

Source revision checks use a command-scoped
`git -c safe.directory=/absolute/selected/repo -C /absolute/selected/repo ...`.
No global Git trust configuration or wildcard is installed.

### Track selection, original observations and postprocessing ablations

Every new launcher run records `raw_tracking.json`, preserving all original
tracker IDs/boxes, the chosen ID, detected-frame mask and gap-fill status before
the dense interpolation/smoothing step. Stock tracking does not expose box
confidence in that history; its absence remains explicit. `normalized_probe.json`
and `launcher_options.json` preserve exact decoded 30 Hz presentation timestamps.
These are the actual model-input timestamps, not a relabelled original video.

Add `--track-id INTEGER --actor-id LABEL` after reviewing tracker/source overlays
to select a specific person in a multi-person shot. Without an explicit ID, the
original upstream summed-box-area selection is retained. The actor label is
recorded metadata, not an appearance-based identity selector. A missing requested
ID stops with the raw history retained for review. Track fragments are not
silently joined. New output directories are required; reused dense caches cannot
produce original observation support.

For explicitly reviewed fragments of the same actor, use the paired
`--cached-raw-tracking PATH --actor-review PATH` arguments instead of `--track-id`.
The review identifies usable original tracker IDs and retains unusable head-only
fragments separately. Source SHA, raster, PTS, cached raw-history SHA and actor
selection must match before inference. No automatic re-identification or invented
full-body box is performed. This retains detection evidence; it does not prevent
the model from reading an occluder inside an interpolated target crop.

For a reviewed contamination experiment, add
`--unsupported-body-inputs confidence-only` to mark keypoints missing on frames
without a usable selected target body box, or `confidence-and-features` to also
zero the image condition using the model's trained null representation. The
default is `keep`. Source hashes, exact PTS, raster, actor and raw/cache history
must validate before this hook runs. Only cloned model inputs change; original
ViTPose arrays and tracking history remain the evaluation evidence. The sidecar
`body_input_gating.json` and its NPZ record both original and applied conditions.
Box/CLIFF conditioning and bbox-derived camera depth remain active. A missing
body box does not prove total invisibility, and supported-frame outputs can still
change through temporal context. Clip32 tests reduced foreground contamination
but worsened reappearance residuals; this is an optional comparison, not a new
default or source-motion acceptance certificate.

For a controlled ablation, add `--no-postprocess` and use another output path.
This temporarily forces `Pipeline.forward(postproc=False)` inside the child
because `DemoPL.predict` hardcodes `postproc=True`. It disables the upstream
stationary-joint translation correction and limb IK while preserving the network
prediction and camera-mode choice. The temporary patch is restored after the
demo. `provenance.json` records the selected variant and implementation/options
hashes. The separate global display still recenters its visualization; it is not
evidence that the native body matches the authored floor.

Export the native-camera observation diagnostic in the same configured runtime:

```bash
"$GVHMR_PYTHON" tools/gvhmr/export_motion_quality.py \
  --repo "$GVHMR_ROOT" --prediction /absolute/run/hmr4d_results.pt \
  --vitpose /absolute/run/preprocess/vitpose.pt \
  --video /absolute/run/0_input_video.mp4 \
  --tracking-evidence /absolute/run/raw_tracking.json \
  --output /absolute/run/new-quality-export --device cuda
```

The exporter uses the pinned mesh-derived COCO17 regressor, unchanged ViTPose
XY/scores, exact native intrinsics and timestamps. It binds tracking evidence
to the source and actor's saved bounding boxes. Historical caches without raw
evidence remain unverified for tracking support. These diagnostics test consistency
with observations that also feed GVHMR; they do not establish independent 3D,
room-trajectory or object-contact accuracy.

### DPVO versus Pi3X camera input

Moving-camera runs default to DPVO with `--dpvo-timing source-pts`. This corrects
the pinned demo's use of processing wall-clock timestamps in DPVO's extrapolator.
`dpvo_timing.json` records each original and supplied timestamp, processed-frame
index, source hash and completion status. To reproduce the upstream control,
select `--dpvo-timing upstream-wallclock` explicitly in a separate fresh run.

A reviewed same-shot Pi3X bundle can supply camera poses without a DPVO runtime:

```bash
bash tools/gvhmr/run.sh run --camera moving \
  --camera-estimator pi3x --pi3x-bundle /absolute/pi3x-bundle \
  --track-id 3 --actor-id reviewed-person \
  --video /absolute/original-video.mp4 --output runs/my-scene/gvhmr-pi3x
```

The bundle must refer to the exact normalized video bytes/raster/PTS, have a
continuous-shot review and include both sequence endpoints. Rotations use SLERP
without extrapolation. Pi3X OpenCV camera-to-world matrices are converted to
the upstream cache's translation plus XYZW quaternion convention. GVHMR then
uses relative world-to-camera rotations; translations and Pi3X intrinsics are
unused in this initial adapter. A constant world-axis rotation cancels from this
relative input. Do not apply a Blender camera-axis flip inside this interface.
This contract is tested; physical camera accuracy, metric scale, scene placement
and support geometry still require independent checks.

Normalization preserves duration to within one 30 Hz frame and records the
original PTS origin. The bounded EOF policy retains the last source frame when
appropriate; a nonzero original first PTS must not silently truncate the clip.
Full output decoding and representative source/body inspection remain required.
