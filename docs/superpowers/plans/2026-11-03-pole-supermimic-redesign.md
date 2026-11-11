# Pole SuperMimic Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement project-owned Q1 pole weaving training on the vendored SuperMimic core while keeping soccer training and playback independent.

**Architecture:** Keep pole-specific scene, course pairing, policy features, collection, rewards, evaluation, and run identity under `pole_task/`. Adapt the vendor's model, critic, replay, and update primitives through pole-only classes without changing soccer defaults.

**Tech Stack:** Python 3.11, PyTorch, MuJoCo, vendored SuperMimic, NumPy, YAML/TOML.

**Spec:** `docs/superpowers/specs/2026-11-03-pole-supermimic-redesign-design.md`; OpenSpec change `openspec/changes/soccer-field-pole-weaving-demo/`.

## Global Constraints

- Preserve expert input width 79, history length 50, and action width 22.
- Initialize the pole feature residual path to zero and check action parity at tolerance `1e-6` before training.
- Keep the vendored SuperMimic commit and soccer pipeline behavior unchanged.
- Never load the task 7.7 final actor or soccer student as pole initialization.
- Keep the soccer ball visible as scene dressing but exclude it from pole observation, reward, reset, and termination.
- Do not open a UI during implementation or headless acceptance.
- Use the locally recorded `expert/checkpoints/model_17800.pth` as the expert checkpoint; verify its hash against `expert/SOURCE.json`. Do not require a `best.pth` alias or substitute a soccer student/task 7.7 actor.

## Review Focus

- Missing or incompatible expert checkpoint must fail before run output creation; preflight can cover this without the binary.
- Paired course/reference identity must survive collection, replay, and imagined updates.
- New pole features must not alter the expert action path before training.
- Pole and fall terminals must remain distinguishable from route completion.
- Soccer imports, checkpoint schema, and scene hash must remain independent.

### Task 1: Align OpenSpec with the approved vendor-adapter architecture

**Files:** Modify `openspec/changes/soccer-field-pole-weaving-demo/design.md`, `tasks.md`.

- [ ] Update design decisions and task acceptance to require vendor model/replay/update primitives, keeping source task actor weights excluded.
- [ ] Reconcile task descriptions with 79x50 expert history and the 1e-6 parity gate.

### Task 2: Complete isolated scene, transform, and transition collection

**Files:** Modify `pole_task/scene.py`, `environment.py`, `data.py`; add or modify focused checks under `tests/`.

- [ ] Ensure derived soccer scene contains the three source-compatible pole geoms and unchanged original soccer XML.
- [ ] Apply the shared `(-2.0, +2.4)` transform to every route pole and reference root.
- [ ] Preserve course/reference identity and contacts/actions/terminal causes in transitions.

### Task 3: Replace custom learner models with SuperMimic adapters

**Files:** Modify `pole_task/models.py`, `pole_task/training.py`; use `3rdparty/SuperMimic/src/supermimic/{models.py,algorithm.py,replay.py}` without changing vendor files.

- [ ] Keep an expert-compatible `PolePolicy` and zero-initialized route/pole residual; validate exact input and output slicing.
- [ ] Configure or subclass vendor `WorldModel` and `Critic`; remove the project-owned independent MLP/optimizer implementations.
- [ ] Use vendor replay and SuperMimic model, critic, imagination, and actor update code; retain only pole-specific orchestration and gates.

### Task 4: Add reproducible pretraining and staged actor gates

**Files:** Modify `pole_task/training.py`, `pole_task/config.json`, `pole_task/preflight.py`.

- [ ] Train on the nominal S route plus the straight/left/right retention references, and reserve the held-out S route for validation/evaluation; persist every pair identity in replay and run metadata.
- [ ] Gate actor updates on held-out model/termination and real-return critic thresholds.
- [ ] Apply actor trust bounds and periodic real rollout rollback; persist checkpoint lineage, budgets, seeds, best checkpoint, and stop reason.

### Task 5: Finish headless playback, metrics, and operator documentation

**Files:** Modify `pole_task/play.py`, `pole_task/evaluate.py`, project docs.

- [ ] Load only project pole checkpoints and report all registered route, collision, fall, and RMSE fields.
- [ ] Document preflight, train/resume, evaluate, and optional GUI commands without requiring a display to parse or run headlessly.

### Task 6: Verify compatibility and close OpenSpec tasks

**Files:** `openspec/changes/soccer-field-pole-weaving-demo/tasks.md` and affected implementation files.

- [ ] Run required preflight and focused headless checks; run soccer smoke checks without opening UI.
- [ ] Run nominal/held-out/retention training acceptance from the local expert checkpoint; do not claim completion without recorded passing metrics.
- [ ] Mark only fully implemented and verified OpenSpec tasks complete; record the missing-checkpoint blocker for checkpoint-dependent acceptance.
