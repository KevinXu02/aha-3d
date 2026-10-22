# Runtime policy

The pipeline runs on one Linux workstation (Ubuntu or Windows WSL2) with one
NVIDIA GPU. There is no scheduler: every stage runs directly on this machine.
See [setup](docs/SETUP.md) for installation and runtime configuration.

## Hardware

Target a single 24 GB NVIDIA GPU (RTX 4090 / RTX 6000 Ada class) with a current
driver, plus enough RAM and disk for model weights, caches and renders.

The pipeline was originally validated only on one 96 GB NVIDIA RTX PRO 6000.
Smaller cards are the target but untested: heavy stages (Pi3X 64-frame
inference, SAM3 tracking, Kimodo with its Llama text encoder, Cycles renders)
may need the 32-frame tier, lower resolution or pixel limits, shorter clips or
smaller batches. Record any such reduction with the result.

The Static level also runs on AMD Instinct GPUs with ROCm, and the Human-level
runtimes install there; see [docs/install/rocm.md](docs/install/rocm.md).

## Environment

Use an isolated Python environment and externally supplied Blender and model
assets. Local executable paths and caches are configuration
(`configs/runtimes/local.json`, `kimodo_blender/env.local.sh`); the source
contains no personal deployment configuration. Do not modify another project's
or a shared environment to make this one fit.

Source `kimodo_blender/env.sh` in each shell. CPU threads come from
`INDOOR_THREADS`, then the runtime profile's `threads`, then all cores. The GPU
is the profile's `gpu` (default 0) or an existing `CUDA_VISIBLE_DEVICES`.

Run the unit tests locally:

```bash
source kimodo_blender/env.sh
python -m unittest discover -s tests -v
```

## Execution

- Run one GPU job at a time. Batches run their tasks sequentially in a detached
  local background worker; do not start another GPU stage beside it.
- Use new output directories; never overwrite earlier results.
- A finished process is not evidence of completion: inspect exit codes
  (`*.log.exit`), logs and validation artifacts.
- Stop only processes your task started.
- Do not include credentials or tokens in logs, configs or commands.
