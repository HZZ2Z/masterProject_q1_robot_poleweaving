# Football Field Pole Weaving Demo

## Intent

Bring the passed SuperMimic task 7.7 pole-weaving capability into this repository as a standalone MuJoCo demonstration on the existing Q1 soccer field. The robot only follows the three-pole S-shaped route; soccer kicking behavior is not part of this demo.

The source task 7.7 actor checkpoint completed the 675-step nominal course without pole contact or falling, with 0.0312 m root tracking RMSE. Its held-out course completed 671 steps with the same failure-free result. Straight, left-turn, and right-turn retention clips also passed. The source checkpoint is about 212 MB and depends on the source project's policy, robot, control, scene, and reference-motion semantics.

## Goals

- Show the source task 7.7 policy weaving around three collidable poles on the soccer field already represented by `expert/resources/q1_description/q1_soccer.xml`.
- Keep this as a separate locomotion demo and entry point; do not load or modify the soccer student's actor or training flow.
- Preserve the source actor's joint/action meaning, control contract, and reference motion. Adapt scene appearance and placement without silently changing those semantics.
- Provide a reproducible launch command and report course completion, pole collision, fall, and route tracking outcome.

## Non-goals

- Integrating pole weaving into soccer perception, ball pursuit, kicking, or the current `primebot_supermimic_soccer_v1` student checkpoint.
- Training a new policy or expanding beyond the fixed nominal three-pole layout and matching reference.
- Changing the soccer training scene or default soccer playback behavior.

## Constraints

- The existing soccer scene is Q1 V3 with a 22-DoF robot, free ball, field markings, and a 2 ms MuJoCo timestep. The source checkpoint has a different checkpoint format and reference-conditioned observation contract from the soccer student.
- Build a dedicated pole-demo scene derived from the soccer scene so the field, markings, and ball remain visible, while preserving the robot physics and action/control semantics required by the source checkpoint. If the robot models differ, use the source-compatible robot definition in the derived scene and carry over the soccer field and ball assets around it.
- Keep source and target scenes intact. The derived scene, launcher, and assets belong to a dedicated pole-demo namespace.
- Avoid a runtime dependency on files outside this repository. Bring over the minimum policy inference code, checkpoint, reference motion, and scene/config data needed for playback; retain their provenance and hashes.
- The source nominal course positions poles at approximately (0.0, 0.018), (2.024, 0.146), and (3.839, 0.117) metres in course coordinates. Place the course in a clear area of the soccer field and preserve its ordered route/reference pairing.
- Use course +x along the soccer field's long axis and +y toward the field's positive lateral side, with zero yaw. Translate all reference root positions and pole positions together by (-2.0, +2.4) m so the course sits in the upper half of the field, away from the static soccer ball at (1, 0); do not rotate or reshape the passed route.
- Import `runs/q1-pole-weaving/stage-7-7/actor-block7/checkpoint_2500.pt` (SHA-256 `c42237dff52a5664d83af37fa7afed00670d09955f562855161dc59f7b73f8d6`) and its matching `runs/q1-pole-weaving/motions-v7/nominal-01.npz` (SHA-256 `6c49b6df27172d3daf00348849418592e893797908a07543d14b0b6f6f3d51a3`). Record source paths and hashes in the target asset manifest.
- Use the target project's documented runtime baseline (Python 3.11, PyTorch 2.9, MuJoCo 3.17). The demo opens a GUI viewer by default; a headless full-course run is available for deterministic acceptance.
- Match source poles exactly: static cylinders, radius 0.04 m, height 1.2 m, zero density, `contype=1`, `conaffinity=1`, `condim=3`, and friction `[0.8, 0.005, 0.0001]`.
- Use source outcome semantics: any robot-pole contact is a collision; falling uses the source robot's existing height/tilt termination; route completion means the final root position is within 0.5 m of the reference endpoint; tracking RMSE is the stepwise Euclidean root XY error against the matched reference.

## Approaches

1. **Compose a dedicated scene from the soccer field and source-compatible locomotion model (chosen).** Keep the existing soccer scene as the source for field and ball presentation, derive an isolated pole-demo scene, and use the passed source actor and its matching runtime assets. This preserves the proven behavior while avoiding changes to soccer playback. It requires checking model/control compatibility and validating the composed scene.
2. **Modify the current soccer scene and use its existing playback entry point.** This puts poles directly into the default soccer environment, but couples an unrelated locomotion demo to soccer resets, policy loading, and future scene changes. It also risks checkpoint behavior if the source robot/control semantics do not match exactly.
3. **Retrain a pole policy using this repository's soccer student stack.** This gives one checkpoint format but requires adding a reference-conditioned locomotion task to a vision-based soccer policy and repeats the source training/acceptance effort without helping the standalone demo goal.

Approach 1 is selected because it reuses the requested soccer-field presentation and the already-passed weaving behavior while isolating the demo from soccer training and playback.

## Behavior and implementation shape

- Add a dedicated derived MJCF scene with the soccer pitch, its visible ball, and three physical poles at the fixed nominal layout. Keep non-demo soccer flows on the original scene.
- Add a standalone launcher that loads only the pole-weaving checkpoint and its matched reference, initializes the source-compatible controller, and runs a full episode in the derived scene.
- Keep runtime imports and assets self-contained within this repository. Use an isolated namespace so the existing soccer policy, reset logic, and checkpoint validation remain unaffected.
- Show a full-field camera view suitable for observing the robot and all three poles. The ball is scenery and does not affect the route or reset/termination logic.
- At completion, report steps, route completion, collisions, falls, and tracking RMSE. Preserve the source's successful/failed outcome semantics.
- Document the command, required runtime environment, asset provenance, and the distinction between the pole demo and soccer student playback.

## Acceptance

- The existing soccer scene and soccer student playback continue to load without changes in behavior.
- The new launcher loads the in-repository checkpoint/reference/scene without looking outside this repository.
- A deterministic full nominal playback completes the 675-step reference with all three poles present, no pole collision, and no fall; it reports route completion and tracking RMSE.
- The playback reaches the endpoint within 0.5 m and root XY tracking RMSE is at most 0.5 m, matching the task 7.7 gate.
- The composed soccer-field scene visually retains the existing field, markings, goals, and ball.
- The demo remains independent of soccer kick and ball-goal success criteria.

## Calls left to the builder

- Choose exact file names and the smallest maintainable boundary for source inference code.
- Choose pole visual materials and camera defaults consistent with the existing soccer viewer.
- If exact physics parity cannot be maintained in the composed scene, identify the smallest scene adjustment that preserves the soccer-field appearance and validate the source policy against it before finalizing the composition.
