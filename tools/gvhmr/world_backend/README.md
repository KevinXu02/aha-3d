# Packaged world-motion backend

These are the owner's custom motion helpers, migrated on 2026-09-17 from local,
untracked files alongside PromptHMR revision
`3b566b7dbb28ce506c7ea972c18693f4c705ce8c`. The owner confirmed authorship and
authorized publication in this project. These files were not part of that
upstream revision. They retain this project's [distribution status](../../../DISTRIBUTION.md);
no new code license is assigned.

| Source-relative file | Purpose |
| --- | --- |
| `experiments/postopt_ab/run_pi3x_dense.py` | Dense same-shot Pi3X camera track |
| `experiments/postopt_ab/build_results.py` | GVHMR/camera data adaptation |
| `experiments/postopt_ab/build_gvhmr_global.py` | Native global-motion comparator |
| `experiments/postopt_ab/run_ab.py` | Standalone v2 runner and import bridge |
| `experiments/postopt_ab/metrics.py` | Contact, projection and motion diagnostics |
| `experiments/postopt_ab/export_viewer_data.py` | Mesh/joint viewer arrays |
| `pipeline/postprocessing_v2.py` | Optional per-person world placement optimizer |

The layout is retained so existing immutable run snapshots and optional
`--prompt-hmr` overrides keep their interfaces. Normal commands in
[the pipeline guide](../../../docs/WORLD_POSTOPT.md) use this directory by
default. No PromptHMR installation, model, checkpoint or neighboring source
checkout is needed. The `prompt_hmr/utils/rotation_conversions.py` file here is
a selected subset fetched directly from official [PyTorch3D v0.4.0](https://github.com/facebookresearch/pytorch3d/blob/v0.4.0/pytorch3d/transforms/rotation_conversions.py),
with its full BSD license retained in the file. Only unused functions/imports
were removed. This preserves the legacy numerical behavior: substituting the
newer installed transforms changed optimized axis-angle output in regression
testing. No PromptHMR implementation is bundled.

Packaging changes: the tracker accepts configured model/dependency paths and
does not search historical runs; the standalone runner exposes only v2; rotation
utilities retain the attributed PyTorch3D v0.4.0 implementation. Model behavior, optimization
weights and scene acceptance remain governed by the project wrappers. In
particular, standalone adaptation's historical body-derived floor is not the
accepted scene-floor workflow. Use `world_pipeline.py` with the reviewed prior.

See [human-motion installation](../../../docs/install/human-motion.md).
Model libraries, environments, checkpoints and body assets are installed
separately. This migration does not rerun a clean installation or establish motion
quality on new footage.

## Packaging validation (2026-09-17)

The two new portability tests and 13 existing world-pipeline tests passed in the
installed GVHMR environment. Both preparation CLIs ran without `--prompt-hmr`,
and a copied backend imported outside the checkout. On the installed 96 GB RTX PRO 6000,
the original and packaged v2 optimizers produced exactly equal output arrays for
one existing 260-frame adapted clip with 1,000 iterations (19 arrays compared).
This checks migration behavior, not new source-motion quality or a clean install.

Catalog and asset checks passed.
