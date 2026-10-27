# Scripted human-scene experiments and inspection

## Scope: debug experiments, not normal-use prerequisites

The September 16, 2026 [working/debug contract](HUMAN_PIPELINE_GUIDE.md) supersedes
the earlier workflow presentation. The actor selection, lifecycle/visible-interval
reviews, raw diagnostics and alternative-method decisions below describe debug
experiments. Working mode reconstructs all people, uses tracker-driven visibility
and a ViTPose confidence threshold, and keeps only contact-position correction
after rigid placement. Those manual debug checkpoints must not be inserted into
normal use. Existing single-actor/schema dependencies are listed as integration
pending in the contract; this runner does not supply that missing automation.


For a plain-language stage-by-stage explanation of the current room/human route
and the decisions that need GPT-6, read [the responsibility guide](HUMAN_PIPELINE_GUIDE.md).
The human-facing comparison now defaults to synchronized source, retained-control
and actual new-result videos. History and numerical packages are secondary.
When the user explicitly requests full videos of rejected experiments, render
them as diagnostics and retain their rejection status; do not treat this as motion
acceptance or silently enable a frozen automatic full-render policy.

The computational stages run from explicit configuration. Agent inspection supplies
actor identity, source-supported visibility/contact events, room geometry decisions
and the final action/style review. Store those decisions with their source hashes;
ordinary solver comparisons reuse them without repeating visual interpretation.

Use the [human runner](../tools/human_experiments/RUNNER.md) for a resumable graph of
existing commands. It complements the [scene recipe pipeline](PIPELINE.md): it can
coordinate Pi3X, SAM3/GVHMR, cached observations, correction, inspection and
delivery commands without forcing every experiment through Kimodo generation.
It runs in the configured local environment and does not reserve GPUs; run one GPU stage at a time.

## Inspection checkpoints

1. **Source and reconstruction inputs.** Identify the actor and continuous shot;
   review the floor and contact-sensitive furniture. Preserve static scene evidence
   independently of human placement. A new scene still requires room authoring.
2. **Initial scene and motion.** Review identity switches, actual exits versus
   occlusion, useful depth intervals, contact/release events and candidate action
   prompts. Write source/actor/scene-bound configuration. Unknown contact remains
   unknown; an empty tracker mask alone does not establish exit.
3. **Corrected result.** Inspect automatically selected normal, contact-boundary,
   exit-boundary and worst-error frames. Check action/style and geometry alongside
   numerical gates. Passing an engineering stage does not certify reconstruction.

These are agent technical checks within the requested task, not repeated user
approval requests. Revisit a decision when its source changes or new evidence
contradicts it. Do not invoke an LLM for every frame or optimization iteration.

## Repeatable execution

```bash
bash tools/human_experiments/run.sh plan /path/to/experiment.json
bash tools/human_experiments/run.sh execute /path/to/experiment.json --run-dir runs/my_experiment
bash tools/human_experiments/run.sh status runs/my_experiment
```

Run the same execute command to resume. Commands use structured argument arrays,
not generated shell strings. Each stage records parameters, declared input/code
hashes, executable, relevant environment, output inventory, timings and logs.
Changing contact parameters invalidates correction and dependent checks; it does
not invalidate unchanged reconstruction ancestors. Corrupt outputs are retained
as failed cache evidence and recomputed in a new attempt directory.

Declare imported implementation files, geometry/model assets and runtime locks
that affect results. Hashing cannot discover undeclared inputs or make a
nondeterministic model deterministic. Keep seeds explicit. Existing cache-adoption
stages must say that they validate previous outputs; their timings are not fresh
Pi3X/GVHMR inference measurements.

## Numerical gates and compact evidence

The inspector writes `decision.json` separately from its source/scene review page.
A candidate that was successfully measured but fails a numerical gate still gets
an inspection package. A malformed or mismatched source/actor/scene input is an
execution failure. Gate a full-render stage on the typed `/allow_render` value;
keep reporting stages runnable after rejection.

Check every active frame, including source-occluded or cropped intervals. Exclude
terminal inactive padding. Full-active floor and unwanted-object checks supplement
contact gap, penetration and planted-patch sliding. They do not establish contact
forces, friction, balance, exact finger contact area or correct source style.

Also inspect positive minimum-body support clearance. Penetration can improve
simply because the whole body floats higher. The inspector samples the largest
clearance as a diagnostic; a jump or elevated support can make it legitimate, so
it is not a universal hard gate.32's strong joint candidate reaches21.15cm and
fails the intended grounded-walking behavior despite lower penetration.

The compact package retains source-frame IDs and selection reasons, so worst-frame
sampling can be audited. Normal frames and event/exit boundaries accompany maxima;
an error-only montage can miss identity or action changes. Cached historical room
images must be labeled historical and must not stand in for a new candidate.

`tools/human_experiments/render_inspection.py` produces sparse **actual candidate**
room views from a frozen room and body cache. It saves and reopens the animated
scene, verifies all active body meshes and visibility, checks static mesh geometry,
and renders the selected source-frame indices using an explicit camera/backend.
This diagnostic path can run for rejected candidates without rendering a full
animation. It does not supersede final full-video decoding/timing checks.

Offline inspection packages contain `bundle_manifest.json` with an entry HTML and
an exact SHA256 file inventory. The unified portal links them through a report
plus bundle reference; its server exposes those explicit verified files, not an
entire run directory. Regenerate and revalidate the package before relinking if
any artifact changes.

## Contact experiment policy

Keep body dimensions, actor identity, scene geometry, source events and evaluation
cohorts fixed while comparing solver settings. The new global guards for this
regression are 20 mm floor penetration and 2 mm containment in explicitly declared
closed-convex object components. They are new declared guards, not a retroactive
change to the old local contact limits. Concave geometry and triangle crossings
require separately reported checks; convex containment is not general collision
certification.

For translation-feasible placement, jointly penalize full-active ground/object
errors and the reviewed hand/foot targets, with temporal regularization. Do not
apply a separate unreported grounding shift after the joint solve. Limit comparisons
to the declared candidate set and preserve failed results.

When fixed articulation cannot satisfy simultaneous targets, a root-only solver
should report the conflict. A subsequent Kimodo pose-change experiment requires
an explicit action prompt, preservation constraints throughout visible intervals,
and final contact/action/transition checks. Never generate the departed tail or
claim that conditioning alone guarantees exact contact.

Plant optimization uses centroid speed; version2 acceptance also checks the maximum
speed of corresponding selected patch points. The latter catches rotation around
a stationary centroid. Preserve `adjacent_event_pair_ending_frame` semantics when
reading derivative cohorts: acquisition frames are already excluded. Contact and
plant training masks are distinct; their union is not an independent heldout set.
