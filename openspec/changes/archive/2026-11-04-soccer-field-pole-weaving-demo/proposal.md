# Proposal

## Why

The current change built a pole learner around the soccer expert and this repository's older SuperMimic model contract. That differs from the successful source task 7.7 setup. The project needs independently configurable SuperMimic training profiles so pole weaving can reproduce the source task's model, inputs, robot and training semantics while soccer keeps its current network and checkpoint behavior.

## What Changes

- Add independent `soccer` and `pole-weaving` SuperMimic profiles selected by configuration, with separate model contracts, run outputs, and checkpoint identities.
- Rework the pole profile to migrate the source task 7.7 robot, observation/history, model architecture, course conditioning, reward, and training configuration into this project.
- Initialize pole training from the source task 7.7 warmup checkpoint and train a new project checkpoint; use the task 7.7 final actor only as a reference result, never as initialization or runtime policy.
- Reuse the project's soccer-field scene for presentation while preserving source-compatible Q1 robot and pole physics; the ball and goals do not participate in pole training or evaluation.
- Keep the existing soccer profile's network, expert/student checkpoint format, commands, and training behavior compatible.
- Replace local-expert-based acceptance with strict source-checkpoint compatibility, profile-isolation checks, and headless task 7.7 course/replay gates.

## Capabilities

### New Capabilities

- `pole-weaving-demo`: Train and evaluate a source-compatible pole-weaving profile on the soccer-field scene, independently from soccer play.

### Modified Capabilities

None.

## Impact

Reworks `pole_task/` and the SuperMimic integration/configuration boundary, adds a source-compatible pole profile and checkpoint provenance, and updates training/evaluation documentation. The shared vendor pin may remain fixed, but the required source task behavior must be migrated behind opt-in profile settings. Existing `q1_soccer` defaults and checkpoint contracts remain unchanged. The work requires the source project and its task 7.7 warmup checkpoint as migration inputs; no UI is required for implementation or acceptance.
