# Design

## Context

See `proposal.md` for the change motivation and `specs/pole-weaving-demo/spec.md` for externally visible requirements. The current local pole implementation uses the soccer expert initialization and an older SuperMimic network contract; source task 7.7 uses a course-conditioned actor/world model/critic contract that cannot be loaded by merely changing a run config. The source project's task 7.7 training begins from `course-warmup-v10/checkpoint_2000.pt` (SHA-256 `b27dabb8c1f1698b669f7dcb59b5b08f1655a7a43f85b57b2686f76096a224f5`) and produces a distinct final actor. This project pins an older SuperMimic revision than the source task.

## Goals / Non-Goals

**Goals:**

- Make the SuperMimic integration resolve profile-specific model, history, observation, robot, and training configuration.
- Migrate source task 7.7 pole behavior into an opt-in pole profile that strictly loads the source warmup checkpoint and trains a new policy here.
- Preserve soccer's existing model, checkpoint format, default configuration, and entry points.
- Use the soccer-field presentation without changing the source-compatible robot and pole task physics.

**Non-Goals:**

- Loading the source task 7.7 final actor as a project policy or training initialization.
- Using the soccer expert `expert/checkpoints/model_17800.pth` to initialize pole training.
- Replacing the source task with the prior adapter-only learner or another independently implemented SuperMimic algorithm.
- Upgrading the shared vendor dependency in a way that changes the soccer profile's default behavior.

## Decisions

### Profile-based architecture selection

Add explicit `soccer` and `pole-weaving` profiles at the SuperMimic construction boundary. Resolve architecture and task dimensions from the selected profile rather than global constants. The soccer profile continues constructing the current 79-value, 50-frame expert-based soccer policy and its existing 22-action/checkpoint contract. The pole profile defines the source-compatible 78-value-per-frame pole actor input, source history length, source actor/world-model/critic module layout, and course conditioning. Save the fully resolved profile and architecture identity in every run/checkpoint manifest; reject absent or conflicting profile metadata rather than guessing.

This is a configurable shared framework with distinct profile contracts, not one universal network shape. The selection point and defaults must keep existing soccer calls equivalent when no pole profile is selected.

### Migrate source task behavior behind the pole profile

Keep the shared `3rdparty/SuperMimic` pin unchanged. Port or adapt the source task 7.7 behavior represented by its registered checkpoints, including model modules and state-dict names/shapes, robot/control setup, 78-value course-conditioned observations, task reward/termination, training stages, and evaluation rules. Keep these pieces in an explicit source-compatible profile/task integration so the soccer profile does not import or execute pole-specific behavior. Do not retain parallel pole MLPs or optimizer/update loops where the source SuperMimic core already provides the behavior; profile-specific code supplies only task and input differences.

The captured successful warmup and final checkpoints contain only the base reference encoder/decoder actor weights; they do not contain the later `speed_encoder`, `route_encoder`, or `route_adapter` keys present in the current source checkout. The source task's six pole coordinates are already included in each 78-value observation. Therefore this checkpoint profile MUST construct the exact checkpoint architecture and MUST NOT add those later modules: adding them would require partial loading or fresh weights and would no longer reproduce the registered successful checkpoint contract.

The source checkpoint's full required model state must load strictly. Preflight checks keys and tensor shapes and reports differences before any run directory is created. No `strict=False`, silently ignored weights, projection shim, unregistered extension modules, or unreported remapping is allowed. Any intentional key mapping must be deterministic, tested against the registered source checkpoint, and recorded.

### Warmup checkpoint and training provenance

Resolve the task 7.7 warmup checkpoint and its source config/revision from an explicit migration manifest rather than treating the soccer expert path as a generic initializer. Verify the recorded SHA-256, model contract, and source task identity before training. Continue training from the warmup state under task 7.7's configured stage semantics, learning rates, horizon, and update budget, producing a new project-owned checkpoint. The task 7.7 final actor (SHA-256 `c42237dff52a5664d83af37fa7afed00670d09955f562855161dc59f7b73f8d6`) is only a provenance and performance reference; it is never loaded for policy execution or initialization.

### Separate physical task from field presentation

Use the project's soccer field, markings, and goals as scene presentation. Keep the source-compatible Q1 robot/control model and three pole collision geoms authoritative for pole simulation. The visible ball is static and excluded from pole observations, rewards, reset/termination, and course logic. Validate source and project scene identities with hashes and named geometry checks so field presentation changes cannot silently substitute a different robot or pole physical contract.

### Acceptance mirrors the validated task

Acceptance runs from a newly trained project checkpoint in headless mode. First require strict warmup-checkpoint loading and profile isolation. Then evaluate the nominal 675-step route, held-out 671-step route, straight 198-step reference, and left/right 298-step references with registered task 7.7 source metrics. Require nominal and held-out endpoint distance <= 0.5 m, root XY RMSE <= 0.5 m, and no pole contacts or falls; require each retention RMSE <= 0.3 m and no falls. Run 20 fixed-seed nominal trials and require all to pass. Record per-case outcomes and resolved profile identity. A source final actor can be run only as a comparison fixture during development, not counted as the project's trained result.

Separately run the existing soccer preflight and headless checkpoint playback smoke against its existing soccer artifact format. Do not require or open a display; GUI playback remains available for the user after this work.

## Risks / Trade-offs

- **Source and pinned vendor APIs differ** -> Port the minimum source task changes behind profile-specific selection; keep the shared dependency pin and prove soccer construction/output parity.
- **Checkpoint keys match but semantics differ** -> Validate tensor keys/shapes plus observation ordering, normalization, history, action scaling, and robot joint identity; perform deterministic source policy parity checks before training.
- **Profile defaults leak pole changes into soccer** -> Add explicit profile selection and regression checks for soccer config resolution, state-dict shapes, and checkpoint loading.
- **Source warmup artifact is unavailable in a clean environment** -> Make its path configurable, record the expected hash and source provenance, and fail preflight clearly without substituting `model_17800.pth` or the final actor.
- **Field composition changes contact behavior** -> Compare robot/control and pole physical identities separately from presentation assets and run headless contact/clearance checks.

## Migration Plan

1. Record source task 7.7 config, source revision, warmup checkpoint hash, state-dict contract, observation/action semantics, robot identity, and acceptance references in a migration manifest.
2. Add profile selection while preserving the current soccer profile as the default; verify current soccer config and checkpoint loading before adding pole profile behavior.
3. Port source-compatible pole model/task behavior behind the pole profile and prove strict warmup loading plus source policy output parity.
4. Train the new pole checkpoint from the warmup state and run all source-aligned headless acceptance cases.
5. Run soccer compatibility checks and document pole commands, required source artifact setup, and results.

Rollback consists of selecting the unchanged soccer profile and its existing entry points; pole-specific config, code paths, and checkpoints remain isolated.
