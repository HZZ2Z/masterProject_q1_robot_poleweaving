# Tasks

## 1. Register source task and introduce independent profiles

- [x] 1.1 Import the task 7.7 nominal, held-out, and gait-retention reference/course data; verify the files and frame counts against the registered source task artifacts.
- [x] 1.2 Add a migration manifest for source revision, warmup/final checkpoint identities, hashes, source config, robot contract, observation/action semantics, and expected tensor contract; verify all registered files and hashes before any run can be created.
- [x] 1.3 Add explicit `soccer` and `pole-weaving` profile resolution with separate config/output identities; verify default soccer config resolution is unchanged and a pole run cannot reuse soccer output paths.

## 2. Migrate the source-compatible pole model contract

- [x] 2.1 Implement the source task 7.7 actor, world-model, and critic architecture behind the pole profile, including module names and dimensions; verify every required state-dict key and shape against the source warmup checkpoint.
- [x] 2.2 Migrate source robot/control, 78-value per-frame observation, 10-frame history, normalization, action scaling, and course-conditioning semantics; verify dimensions/order and known source input-output fixtures. Match the registered successful checkpoint exactly; do not add speed/route encoders or a route adapter because that checkpoint contains no weights for them and its observation already includes pole coordinates.
- [x] 2.3 Strictly load the registered source warmup checkpoint and reproduce source actor outputs on fixed inputs within the registered tolerance; verify incompatible weights fail before creating a run and record all deterministic mappings in provenance.
- [x] 2.4 Keep the existing soccer architecture/checkpoint path behind the soccer profile; verify existing soccer state-dict shapes, serialization, and a pre-refactor checkpoint still load without pole-only settings.

## 3. Integrate source course behavior with the soccer-field scene

- [x] 3.1 Reconcile the dedicated pole scene with the soccer-field presentation while preserving source-compatible Q1 and pole collision physics; verify field/goal/ball/pole geometry identities, source robot dimensions, and headless pole contact behavior.
- [x] 3.2 Verify course layout/reference pairing, ordering, route transforms, and nominal/held-out split against source task 7.7; verify held-out data cannot enter training batches.
- [x] 3.3 Route pole collection, temporal replay, imagination, and evaluation through the selected SuperMimic profile while retaining course/reference identity; verify reset and sampled batches preserve that identity end to end.
- [x] 3.4 Keep soccer ball and goal behavior out of pole inputs, reward, resets, and termination; verify the ball is static presentation and soccer logic is not invoked during pole rollouts.

## 4. Train with the source task 7.7 recipe

- [x] 4.1 Encode the source warmup-to-actor stage transition, optimizer settings, learning rate, imagination horizon, update budget, and configured frozen pole-input columns; verify the resolved run config matches the registered task 7.7 recipe.
- [x] 4.2 Use the SuperMimic profile's model fitting, replay, critic, imagination, and actor update flow with source pole reward/termination behavior; verify updates use paired course context and real pole contact terminates an episode.
- [x] 4.3 Add training preflight, resume, checkpoint selection, bounded stopping, and provenance reporting for the source warmup initialization; verify missing or corrupt source artifacts fail clearly and never fall back to `model_17800.pth` or the source final actor.
- [x] 4.4 Document headless pole training/evaluation commands and required source artifacts; verify the documented preflight resolves config/data without opening a UI or starting an unintended run.

## 5. Implement source-aligned evaluation and playback

- [x] 5.1 Make pole playback load only a project-trained pole-profile checkpoint; verify missing, soccer-format, source-final, and wrong-profile checkpoints fail clearly.
- [x] 5.2 Report route completion, endpoint distance, played steps, pole contacts, falls, root XY RMSE, and combined success; verify successful, collision, fall, and early-stop cases report consistent fields.
- [x] 5.3 Add fixed-seed headless evaluation for nominal 675-step, held-out 671-step, and straight/left/right 198/298/298-step references; verify each case uses its registered course/reference pairing and emits the required metrics.

## 6. Run final acceptance

- [x] 6.1 Train a new pole-profile checkpoint from the registered source warmup checkpoint and record the exact config, source hash, seed, updates, and output hash.
- [x] 6.2 Require nominal and held-out completion within 0.5 m of endpoint, root XY RMSE <= 0.5 m, and no pole contacts/falls; require all three retention references to have no falls and RMSE <= 0.3 m; verify 20 fixed-seed nominal trials all pass without opening a UI.
- [x] 6.3 Run soccer-profile configuration, checkpoint preflight, and headless playback smoke after the profile migration; verify the existing soccer scene and checkpoint/training entry points remain independent from pole-task behavior.
