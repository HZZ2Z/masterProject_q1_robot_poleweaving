"""Bounded physical-model fit check. No actor updates or W&B training run.

Two independently seeded collections separate train and validation trajectories.
Only near-neutral action excitation is covered; this is not a ball-contact test.
"""
import json
from pathlib import Path
import torch
from .physics import WarpPhysics
from .robot import Q1Control
from .replay import Replay
from .dynamics import Dynamics


@torch.no_grad()
def collect(physics, control, seed):
    generator = torch.Generator(device=physics.device).manual_seed(seed)
    count = physics.qpos.shape[0]
    q = torch.tensor(physics.native_model.qpos0, device=physics.device, dtype=torch.float32).repeat(count, 1)
    q[:, control.qpos_indices] = control.neutral
    physics.reset(q, torch.zeros_like(physics.qvel))
    replay = Replay(160)
    for t in range(160):
        state = torch.cat((physics.qpos, physics.qvel), -1).clone()
        action = .05 * torch.randn(count, 22, device=physics.device, generator=generator)
        for _ in range(control.substeps):
            physics.step(control(physics.qpos, physics.qvel, action))
        following = torch.cat((physics.qpos, physics.qvel), -1).clone()
        fell = following[:, 2] < .3
        replay.append(state=state, next_state=following, action=action,
                      boundary=fell | (t == 159), terminated=fell,
                      context=torch.zeros(count, 0, device=physics.device))
    physics.check()
    return replay


def main():
    torch.manual_seed(37)
    torch.cuda.set_per_process_memory_fraction(.4)
    physics = WarpPhysics(16)
    control = Q1Control(physics.native_model).to(physics.device)
    training = collect(physics, control, 731)
    validation = collect(physics, control, 983)
    world = Dynamics(physics.native_model, widths=(256, 256, 256)).to(physics.device)
    world.set_scales(training.storage['state'])
    optimizer = torch.optim.Adam(world.parameters(), lr=2e-4)
    held_out = validation.sample(256, length=8, generator=torch.Generator(device=physics.device).manual_seed(991))
    with torch.no_grad():
        before = world.window_loss(held_out).item()
        persistence = world.error(held_out['state'][0].expand_as(held_out['next_state']), held_out['next_state']).item()
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = world.window_loss(training.sample(128, length=8))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(world.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
    with torch.no_grad():
        after = world.window_loss(held_out).item()
    result = dict(scope='near-neutral excitation only; no policy or kicking validation',
                  training_seed=731, validation_seed=983, steps_per_world=160,
                  worlds=16, window=8, batches=200, widths=[256,256,256],
                  held_out_loss_before=before, held_out_loss_after=after,
                  persistence_loss=persistence, beats_persistence=after < persistence)
    destination = Path('logs/rebuild_dynamics_check.json')
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
