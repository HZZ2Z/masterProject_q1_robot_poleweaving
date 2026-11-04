# Q1 pole weaving

This profile reproduces source task 7.7's three-pole S-course on the repository's
soccer-field scene. It uses the source-compatible Q1 robot/control contract, a
78-value per-frame observation, 10-frame history, a 149-value course context,
robot-anchored goals, and the source actor/world-model/critic architecture.

Training initializes strictly from the registered task 7.7 warmup checkpoint in
`pole_task/source_manifest.json` and never falls back to the soccer expert or
the source final actor. See [../docs/pole-weaving.md](../docs/pole-weaving.md)
for required source artifacts, the data split, and full commands.

## Quick start

```bash
# Read-only: hashes data and source artifacts, strict-loads the warmup,
# resolves the source recipe, and checks hardware. Creates no run directory.
python -m pole_task.training --preflight-only --config pole_task/config.json

# Train the project pole profile (2,000 warmup + 500 actor updates).
python -m pole_task.training --config pole_task/config.json

# Resume a project-trained run.
python -m pole_task.training --config pole_task/config.json \
  --resume-checkpoint logs/q1-pole-weaving/run-<timestamp>/latest.pt

# Headless evaluation, full acceptance, and playback (project pole checkpoints only).
python -m pole_task.evaluate \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
python -m pole_task.acceptance \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --device cpu
python -m pole_task.play \
  --checkpoint logs/q1-pole-weaving/run-<timestamp>/best.pt --case nominal-01
```

The default output root is `logs/q1-pole-weaving/`, separate from the soccer
profile's `logs/auto`. Pole playback and training do not load soccer policy or
checkpoint code.
