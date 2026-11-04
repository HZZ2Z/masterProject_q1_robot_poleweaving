# Q1 pole weaving

The pole task is an independent SuperMimic profile that reproduces source task
7.7's three-pole S-course on this repository's soccer-field scene. It has its
own source-compatible observation, reward, reset, termination, actor,
world-model, and critic contract, its own training run directories under
`logs/q1-pole-weaving/`, and its own project-trained checkpoint format. The ball
and soccer goals remain visible scenery and never enter pole observations,
rewards, resets, or termination.

## Required source artifacts

Pole training initializes only from the registered task 7.7 warmup checkpoint.
`pole_task/source_manifest.json` records the source revision and the SHA-256 of
every required file under the sibling `../../SuperMimic` checkout:

| Artifact | Registered path | Use |
| --- | --- | --- |
| `warmup_checkpoint` | `runs/q1-pole-weaving/stage-7-7/course-warmup-v10/checkpoint_2000.pt` | strict training initialization |
| `passed_final_checkpoint_reference` | `runs/q1-pole-weaving/stage-7-7/actor-block7/checkpoint_2500.pt` | provenance hash only, never loaded |
| `warmup_config`, `actor_config` | `configs/q1-pole-weaving-*-v10*.toml` | training recipe cross-check |
| `robot_xml` | `assets/q1_v3/xml/q1_v3_mc.xml` | source robot physics identity |
| source model/course/training files | `src/supermimic/*.py` | architecture and course semantics |

The registered warmup is hash-checked, then every actor, world-model, critic,
and robot tensor is loaded with `strict=True` and the source fixture parity is
checked. Missing, corrupt, or unrecognized artifacts stop preflight with the
mismatched field; no run directory or substitute checkpoint is created. The
source final actor and the soccer expert checkpoint are never fallbacks.

## Data split

`pole_task/data/manifest.json` pairs `nominal-01` (675 control steps) and the
held-out `eval-01` (671 control steps) with their ordered pole layouts. The
held-out reference is excluded from training batches and used only for
evaluation. `straight`, `left_turn`, and `right_turn` are training and fixed
gait-retention references. Every reference and course layout is hash-paired.

## Preflight and training

The preflight is read-only: it verifies data pairing and hashes, the source
manifest, the strict warmup load, the resolved 2,000 + 500 update recipe, output
isolation, and hardware. It opens no viewer and creates no run directory.

```bash
python -m pole_task.training --preflight-only --config pole_task/config.json
```

Training writes only under the pole profile's output root. A run records its
source revision, warmup path/hash, resolved stage/optimizer/learning-rate
recipe, data split, seed, environment-step budget, metrics, stop reason, and
selected checkpoints:

```bash
python -m pole_task.training --config pole_task/config.json
```

The source recipe is fixed: 2,000 warmup updates with the actor frozen and
deterministic physics, then 500 actor updates with randomization enabled and
the six pole-input columns of each history frame frozen. Every update fits the
world model with the registered batch/horizon settings, then fits the critic
and (in the actor stage) the actor through source-aligned imagination. Real pole
contact terminates an episode as a replay boundary and a failed rollout.

Resume restores the project actor, world model, critic, optimizers, update
index, stage, and best score after validating the checkpoint format, source
warmup hash, robot/data identity, exact config, and recipe:

```bash
python -m pole_task.training --config pole_task/config.json \
  --resume-checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt
```

For a bounded smoke run, add `--max-updates N` (it cannot exceed the registered
2,500-update budget). Checkpoint selection uses `best.pt` when it exists and
otherwise `latest.pt`; selection never searches for a soccer or source-final
checkpoint.

## Headless evaluation and playback

Both commands are headless by default and accept only project-trained
pole-profile checkpoints. A missing, soccer-format, wrong-profile, or source
final checkpoint is rejected without fallback.

Evaluate the registered nominal, held-out, and retention cases with fixed seeds:

```bash
python -m pole_task.evaluate \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
```

Run the full headless acceptance: every registered case plus 20 fixed-seed
nominal trials. Nominal/held-out cases must complete within 0.5 m of the
endpoint with root XY RMSE <= 0.5 m and no pole contacts or falls; retention
cases must have no falls and RMSE <= 0.3 m; every nominal trial must pass.

```bash
python -m pole_task.acceptance \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
```

Play one registered case without opening a display:

```bash
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --case nominal-01
```

Add `--gui` only when a display is available. Results, including route
completion, endpoint distance, played steps, pole contacts, falls, root XY
RMSE, and success, are written next to the checkpoint's run output.
