# Resumable human experiment runner

Run a structured stage graph without copying a scene-specific shell script. This
stdlib runner is separate from the existing `tools/indoor` scene recipe pipeline.
It launches commands in the current environment in the foreground, does not
reserve a GPU, and does not install anything. Run one GPU job at a time; do not
execute GPU stages while a background batch is using the GPU.

```bash
bash tools/human_experiments/run.sh plan configs/experiment.json
bash tools/human_experiments/run.sh execute configs/experiment.json --run-dir runs/experiment
bash tools/human_experiments/run.sh status runs/experiment
# Same command resumes; optional stop after one stage and its preceding graph nodes:
bash tools/human_experiments/run.sh execute configs/experiment.json --run-dir runs/experiment --until inspect
```

Set `HUMAN_RUNNER_PYTHON=/absolute/python` to choose the runner interpreter. Each
stage may select a different executable in `argv`; the runner never activates,
mutates or guesses a model environment. Input paths are relative to the config's
`root`, and `root` itself is relative to the config file. Explicit absolute paths
are also accepted. The run directory is relative to the caller's working directory.

## JSON interface

This schematic example assumes `tools/example_solve.py` and
`tools/example_inspect.py` implement the depicted arguments. Substitute the actual
shared solver, inspector and reporting commands in a case config; these names are
not additional installed tools.

```json
{
  "schema_version": 1,
  "id": "clip32_contact_comparison",
  "root": "..",
  "stages": [
    {
      "id": "solve",
      "needs": [],
      "argv": ["{python}", "{root}/tools/example_solve.py", "--body", "{input:body}", "--config", "{params_file}", "--output", "{stage_dir}/solve"],
      "inputs": {"body": {"path": "runs/frozen/body_room.npz", "sha256": "REPLACE_WITH_ACTUAL_64_LOWERCASE_HEX_CHARACTERS"}},
      "code": [{"path": "tools/example_solve.py"}, {"path": "src/aha3d/motion/trajectory.py"}],
      "outputs": {"body": "solve/scene/body_room.npz", "report": "solve/scene/report.json"},
      "params": {"method": "smooth_scene_contact", "body_scale": 1},
      "env": {"CUDA_VISIBLE_DEVICES": ""}
    },
    {
      "id": "inspect",
      "needs": ["solve"],
      "argv": ["{python}", "{root}/tools/example_inspect.py", "--body", "{input:body}", "--output", "{stage_dir}/inspection"],
      "inputs": {"body": {"stage": "solve", "output": "body"}},
      "code": [{"path": "tools/example_inspect.py"}],
      "outputs": {"decision": "inspection/decision.json", "evidence": "inspection/evidence"}
    },
    {
      "id": "render",
      "needs": ["solve", "inspect"],
      "gates": [{"stage": "inspect", "output": "decision", "pointer": "/allow_render", "equals": true}],
      "argv": ["/absolute/blender", "--background", "--python", "{root}/tools/example_render.py", "--", "--body", "{input:body}", "--output", "{output:frames}"],
      "inputs": {"body": {"stage": "solve", "output": "body"}},
      "code": [{"path": "tools/example_render.py"}],
      "outputs": {"frames": "frames"}
    },
    {
      "id": "summary",
      "needs": ["solve", "inspect", "render"],
      "needs_policy": "settled",
      "argv": ["{python}", "{root}/tools/example_report.py", "--stages", "{dependency_status_file}", "--output", "{output:report}"],
      "inputs": {},
      "code": [{"path": "tools/example_report.py"}],
      "outputs": {"report": "report.json"}
    }
  ]
}
```

Every command is a JSON string array passed directly to `subprocess.Popen`, with
`shell=False`. `$()`, backticks, spaces and semicolons inside an argument are literal.
The config remains executable code: explicitly invoking a shell is possible and
does not gain sandboxing. Prefer a declared Python entry point for preparation.

Supported string placeholders are `{root}`, `{run_dir}`, `{python}`,
`{stage_dir}`, `{params_file}`, `{dependency_status_file}`, `{input:NAME}` and
`{output:NAME}`. They may occur inside a string, such as `{stage_dir}/solve`.
`_runner/params.json` contains exactly this stage's `params` object. The attempt
directory and its `_runner` metadata exist before launch; declared output paths
and their parents **are not created**, so solvers that require a fresh output
directory work. The child creates its own output directories.

`needs` declares direct dependencies. Artifact references and gates must name a
declared direct dependency and one of its named outputs. IDs, cycles, unknown
fields, duplicate/overlapping outputs, escaping paths and nonfinite params fail
validation before launch. Outputs must be files or nonempty file-containing
directories within the fresh attempt; symlink outputs are rejected.

## Hashes and reuse

Inputs and code entries accept `{"path": "...", "sha256": "..."}`. The optional
expected hash is strongly recommended for frozen regression videos, actor/mask
reviews, Pi3X/GVHMR caches, camera/basis/scene geometry and model weights. A mismatch
fails that stage before launch; it never silently accepts changed data. Omitting
the expected hash means "hash the current artifact and invalidate on change", not
"do not validate". A directory hash is the shared `signature` of its sorted
relative-file-name → SHA256 mapping; code directories ignore Python bytecode.

Declare the exact Python entry points **and imported implementation dependencies**
that affect the stage. The runner hashes its own code, shared I/O helpers and the
resolved executable automatically, but cannot infer arbitrary imports, dynamic
file reads, plugins, model checkpoints or network state. Record those as code/input
artifacts or explicit params. Installed package versions can be bound through an
environment-lock file input. Params are canonical JSON in the stage fingerprint,
not only an opaque whole-run config hash.

Each stage fingerprint includes its own command, params, outputs, declared
inputs/code, direct-dependency identities, executable and relevant environment.
There is no global config fingerprint in stage reuse. Changing only contact
params or contact code therefore reuses intact Pi3X/GVHMR ancestor stages.
Completed matching attempts are reused only after hashing all declared output
files; corrupted, missing or extra directory files invalidate that attempt and
are retained as evidence. A new numbered attempt runs in a fresh directory.
Restoring an earlier config may reuse an older intact matching attempt.

The environment is inherited, with literal `env` overrides. PATH, PYTHONPATH,
VIRTUAL_ENV, CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER, LD_PRELOAD, LD_LIBRARY_PATH,
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are hashed. Add other
behavior-affecting names to `inherit_env`. Do not put credentials in tracked env
or params: these are written to provenance. Runtime/clock/network nondeterminism
is not made reproducible by hashing. Declare random seeds explicitly.

Reuse is within one run directory. For existing expensive outputs from other
runs, declare them as pinned external inputs and use a clearly named adoption/
verification stage if needed. Its timing is verification/copy time, **not** a new
Pi3X, SAM3 or GVHMR inference measurement. No cross-run shared output is mutated.

## Decisions, failure and interruption

The inspector should write a hash-bound JSON decision (for example `allow_render`)
and exit zero for a scientifically rejected but successfully inspected candidate.
A typed `false`, missing field, invalid JSON or missing decision blocks that gate.
The runner records `skipped_gate` and launches no render process. It never changes
the decision into acceptance or drops rejected candidates' evidence.

By default, a failed/skipped dependency blocks its dependent stage. Set
`needs_policy: "settled"` for an inspection/report node that can consume dependency
statuses after failure. `{dependency_status_file}` contains those receipts,
including log and evidence paths. Referencing a failed stage's completed output
still fails; reports must handle unavailable artifacts explicitly. Independent
branches and settled-policy reports continue after ordinary process failure.

Optional `timeout_seconds` is finite and positive. Each stage runs in its own
process group. Failure, timeout, SIGINT and SIGTERM preserve logs/partial output;
interruption stops and waits for the stage process group. A detached background
writer makes the stage fail and is terminated. SIGKILL/power loss cannot be cleaned
up by Python: the retained `.execute.lock/owner.json` prevents an automatic retry.
Inspect the recorded host/PID and associated child process group/job, stop remaining
writers and archive the stale lock before explicit recovery. Locks never expire or
get removed on age alone. The runner is cooperative orchestration, not filesystem
access control; commands must not write into input caches or another attempt.

`run.json` retains config revisions, current stage states and invocation history.
Each `stages/ID/attempt-NNNN/_runner/` contains receipt, params, dependencies and
combined stdout/stderr log. Preflight failures also have a log. Failed partial
artifacts remain in that attempt. A matching cache must pass current validation
before use; a corrupt run manifest is a hard error rather than a fresh run.
The running receipt records the child `process_id`, `process_group_id` and host
before waiting, so hard-kill recovery can identify the exact remaining group.

Exit status: 0 means all requested engineering stages completed, possibly with
explicit gate skips; 1 means failed/blocked stages; 2 means config/run-lock error;
130 means interrupted. Inspect `status`, not just exit zero, when gates skip work.
`completed` does not establish scientific, contact, layout or visual acceptance.

Tests: `python -m unittest discover -s tests -p test_human_runner.py` exercises real
tiny CPU subprocesses, failures, corruption, interruption, locking and gated reports.
