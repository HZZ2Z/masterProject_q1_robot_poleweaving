# pole-weaving-demo Specification

## Purpose

Provides an independently configurable SuperMimic training and evaluation profile that reproduces the source project's successful three-pole S-course task in this project's soccer-field scene.

## Requirements

### Requirement: SuperMimic profiles are independently selectable
The system MUST select soccer or pole-weaving training behavior through an explicit profile, with profile-specific model, configuration, run output, and checkpoint identity.

#### Scenario: Start soccer training
- **WHEN** training is launched with the soccer profile or its existing default command
- **THEN** it uses the existing soccer network and checkpoint contract without requiring pole-task settings or artifacts

#### Scenario: Start pole-weaving training
- **WHEN** training is launched with the pole-weaving profile
- **THEN** it loads that profile's source-compatible model and task settings and writes outputs separate from soccer runs

### Requirement: Pole profile matches source task 7.7 contracts
The pole profile MUST reproduce the source task's Q1 robot/control, observation and history, actor/world-model/critic architecture, course conditioning, and training semantics, with source settings recorded in the run identity.

#### Scenario: Load pole profile
- **WHEN** pole training or evaluation starts
- **THEN** the resolved profile reports the expected source task identity and compatible tensor names and shapes, and rejects a mismatched contract before creating a run

#### Scenario: Load soccer profile after pole configuration is added
- **WHEN** the existing soccer training or playback path is run without pole settings
- **THEN** its model dimensions, action interface, and checkpoint serialization remain compatible with existing soccer artifacts

### Requirement: Pole training starts from source warmup weights
The pole profile MUST strictly load the designated source task 7.7 warmup checkpoint and MUST train a new project-owned pole checkpoint.

#### Scenario: Warmup checkpoint is valid
- **WHEN** pole training starts with the registered task 7.7 warmup checkpoint
- **THEN** every required source model weight is loaded and the new training run records checkpoint path, content hash, config, and source revision

#### Scenario: Warmup checkpoint is missing or incompatible
- **WHEN** the warmup checkpoint is missing, corrupt, or has an unrecognized model contract
- **THEN** preflight fails with the mismatched fields and creates no training run or substitute checkpoint

#### Scenario: Select pole policy for playback
- **WHEN** pole playback loads a policy
- **THEN** it loads a project-trained pole checkpoint and never loads the source task 7.7 final actor or a soccer checkpoint as fallback

### Requirement: Pole course semantics remain paired and source-compatible
Pole training and evaluation MUST preserve the source task's course layouts, matching references, ordered pole geometry, reset rules, reward, and termination semantics across collection, replay, imagination, and playback.

#### Scenario: Sample course data
- **WHEN** a training batch is assembled
- **THEN** each transition retains its matching reference and course identity, and the held-out S-course pair is excluded from training

#### Scenario: Simulate pole contact
- **WHEN** any robot collision geometry contacts a course pole
- **THEN** the episode records pole contact and terminates as unsuccessful in real simulation and evaluation

### Requirement: Pole task uses soccer-field presentation without soccer behavior
The pole scene MUST reuse the project's soccer-field presentation while preserving source-compatible Q1 robot and pole physical behavior, and MUST exclude soccer ball and goal behavior from pole observations, rewards, resets, and termination.

#### Scenario: Run pole task in the field scene
- **WHEN** the pole scene is loaded
- **THEN** field markings and goals are visible, poles have the source task collision geometry, and the ball is static scenery without task interaction

#### Scenario: Verify physical and soccer isolation
- **WHEN** pole and soccer scenes/configurations are loaded independently
- **THEN** the pole task uses its source-compatible robot/control contract while soccer scene physics and soccer training behavior retain their existing identities

### Requirement: Trained pole checkpoint passes source-aligned headless acceptance
The system MUST evaluate a project-trained pole checkpoint headlessly on the registered nominal, held-out, and gait-retention cases and report route completion, endpoint distance, pole contacts, falls, and tracking RMSE.

#### Scenario: Pass nominal and held-out routes
- **WHEN** the checkpoint is evaluated on the nominal 675-step and held-out 671-step S-course references
- **THEN** each completes within 0.5 m of its endpoint without pole contact or fall and has root XY RMSE at most 0.5 m

#### Scenario: Pass gait-retention cases
- **WHEN** the checkpoint is evaluated on straight, left-turn, and right-turn retention references
- **THEN** each reaches its registered 198-, 298-, and 298-step duration without fall and has root XY RMSE at most 0.3 m

#### Scenario: Repeat nominal evaluation
- **WHEN** the final checkpoint undergoes 20 fixed-seed nominal playback trials
- **THEN** all trials complete without pole contact or fall and each satisfies the nominal endpoint and RMSE limits

#### Scenario: Run acceptance without a display
- **WHEN** acceptance is run in a headless environment
- **THEN** all required metrics and per-case outcomes are emitted without opening a UI
