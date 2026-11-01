"""Opt-in hardware tests: Q1_TEST_WARP=1 ./scripts/python.sh -m pytest -q tests/test_warp.py."""
import os
import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get('Q1_TEST_WARP') != '1', reason='Explicit CUDA test opt-in required')


@pytest.fixture(scope='module')
def env():
    from q1_4v4.warp_env import WarpSoccerEnv
    return WarpSoccerEnv(count=4)


def test_warp_six_seconds_and_partial_reset(env):
    import warp as wp
    env.stage = 'balance'
    env.episode_steps = 1000
    env.reset()
    a = torch.zeros((4, 8, 22), device=env.device)
    for _ in range(300):
        o, s, r, d, t, info = env.step(a)
        assert not d.any().item()
    env.check()
    assert info['stable'].all().item()
    assert o.shape == (4, 8, 121) and s.shape == (4, 8, 972)
    q, v = env.qpos.clone(), env.qvel.clone()
    env.reset(torch.tensor([1, 3], device=env.device))
    assert torch.equal(q[[0, 2]], env.qpos[[0, 2]])
    assert torch.equal(v[[0, 2]], env.qvel[[0, 2]])
    assert env.steps.tolist() == [300, 0, 300, 0]
    np.testing.assert_allclose(wp.to_torch(env.data.time).cpu(), [6., 0., 6., 0.], atol=2e-4)


@pytest.mark.parametrize('stage', ['balance', 'move', 'approach', 'kick', 'teamplay', 'match'])
def test_cpu_gpu_observation_and_step_parity(env, stage):
    cpu = env.reference
    cpu.stage = env.stage = stage
    cpu.episode_steps = env.episode_steps = 1000
    cpu.reset()
    env.reset()
    env.qpos[0] = torch.as_tensor(cpu.data.qpos, device=env.device)
    env.qvel[0] = 0
    env.practice[0] = cpu.practice
    env.targets[0] = torch.as_tensor(cpu.targets, device=env.device)
    env.start_positions[0] = torch.as_tensor(cpu.start_positions, device=env.device)
    env.start_ball[0] = torch.as_tensor(cpu.start_ball, device=env.device)
    env.forward()
    np.testing.assert_allclose(env.observe()[0].cpu(), cpu.observe(), atol=2e-5)
    np.testing.assert_allclose(env.state()[0].cpu(), cpu.state(), atol=2e-5)
    action = np.zeros((8, 22), np.float32)
    co, cs, cr, cd, ct, ci = cpu.step(action)
    go, gs, gr, gd, gt, gi = env.step(torch.zeros((4, 8, 22), device=env.device))
    env.check()
    np.testing.assert_allclose(gr[0].cpu(), cr, atol=2e-3)
    # CPU/Warp contact solvers differ in transient ankle velocities. Check poses,
    # then compare observation code at IDENTICAL post-step physical states.
    np.testing.assert_allclose(env.qpos[0].cpu(), cpu.data.qpos, atol=.003)
    cpu.data.qpos[:] = env.qpos[0].cpu().numpy()
    cpu.data.qvel[:] = env.qvel[0].cpu().numpy()
    import mujoco
    mujoco.mj_forward(cpu.model, cpu.data)
    np.testing.assert_allclose(go[0].cpu(), cpu.observe(), atol=2e-5)
    assert gd[0].item() == cd and gt[0].item() == ct


def test_gpu_goal_out_timeout_and_touch(env):
    env.stage = 'match'
    env.episode_steps = 1
    env.reset()
    env.qpos[0, env.ballq:env.ballq+3] = env.tensor([7.05, 0, .11])
    env.qvel[0, env.ballv] = 10
    env.qpos[1, env.ballq:env.ballq+3] = env.tensor([-7.05, 0, .11])
    env.qvel[1, env.ballv] = -10
    env.qpos[2, env.ballq:env.ballq+3] = env.tensor([7.05, 2, .11])
    env.qvel[2, env.ballv] = 10
    foot = env.model.body('red2_left_ankle_roll_link').id
    env.qpos[3, env.ballq:env.ballq+3] = env.xpos[3, foot]+env.tensor([.12, 0, .05])
    env.forward()
    o, s, r, d, t, i = env.step(torch.zeros((4, 8, 22), device=env.device))
    env.check()
    assert i['goal'].tolist() == [1, -1, 0, 0]
    assert i['out'][2].item()
    assert d[:3].all().item() and t[3].item()
    assert env.touches[3, 2].item() == 1
    assert (r[0, :4] > 19).all().item() and (r[1, 4:] > 19).all().item()


def test_gpu_overflow_latches_through_reset(env):
    import warp as wp
    env.reset()
    wp.to_torch(env.failure)[0] = 1
    env.reset()
    with pytest.raises(RuntimeError, match='latched'):
        env.check()
    env.failure.zero_()


def test_expanded_epa_capacity_updates_all_collider_workspaces(env):
    from mujoco_warp._src import types,collision_convex,collision_flex
    assert env.physics_config['epa_horizon']>=128
    assert env.physics_config['epa_horizon']>=10+2*env.model.opt.ccd_iterations
    assert types.MJ_MAX_EPAHORIZON==collision_convex.MJ_MAX_EPAHORIZON==collision_flex.MJ_MAX_EPAHORIZON


def test_gpu_gae_and_normalizer_roundtrip():
    from q1_4v4.gpu_learning import GpuRollout, DeviceValueNorm
    b = GpuRollout(2, 1, 'cuda:0')
    b.rewards.fill_(1)
    b.values.zero_()
    b.masks[0] = 0
    b.compute_returns(gamma=.9, lam=1.)
    assert torch.equal(b.returns, torch.ones_like(b.returns))
    n = DeviceValueNorm('cuda:0')
    x = torch.arange(8, device='cuda:0', dtype=torch.float32)[:, None]
    n.update(x)
    restored = DeviceValueNorm('cuda:0')
    restored.load_state_dict(n.state_dict())
    assert len(n.state_dict()) == 3
    torch.testing.assert_close(restored.denormalize(restored.normalize(x)), x)


def test_gpu_fall_ends_world(env):
    env.stage = 'match'
    env.episode_steps = 1000
    env.reset()
    env.qpos[0, env.rootq[0]+2] = .1
    env.forward()
    _, _, reward, done, truncated, info = env.step(torch.zeros((4, 8, 22), device=env.device))
    assert done[0].item() and info['fallen'][0].item()
    assert not truncated[0].item() and reward[0, 0, 0].item() < -4
    env.check()


def test_gpu_inter_robot_contacts(env):
    import warp as wp
    env.reset()
    # Overlap two opponents only for collision detection; do not integrate this fixture.
    env.qpos[0, env.rootq[7]:env.rootq[7]+3] = env.qpos[0, env.rootq[2]:env.rootq[2]+3]
    env.forward()
    count = int(wp.to_torch(env.data.nacon).item())
    geoms = wp.to_torch(env.data.contact.geom)[:count].cpu().numpy()
    worlds = wp.to_torch(env.data.contact.worldid)[:count].cpu().numpy()
    owners = env.reference.geom_owner
    assert any(world == 0 and {owners.get(int(a), -1), owners.get(int(b), -1)} == {2, 7}
               for (a, b), world in zip(geoms, worlds))
    env.check()
    env.reset()
