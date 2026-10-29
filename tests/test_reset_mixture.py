"""Fixed environment quotas, spatial sampling, and collision-free near-ball starts."""
import mujoco
import pytest
import torch
from q1_supermimic.environment import load_environment
from utils.torch_math import quat_rotate_inverse


def assert_mixture(env):
    e = env.env
    assert torch.bincount(e.reset_group, minlength=3).tolist() == [16, 16, 32]
    field, disk, near = (e.reset_group == i for i in range(3))
    for coordinates in (e.base_pos[:, :2], e.ball_pos[:, :2]):
        assert (coordinates[field, 0].abs() <= 6.5).all()
        assert (coordinates[field, 1].abs() <= 4).all()
        assert (coordinates[disk].norm(dim=-1) <= 3.5 + 1e-5).all()
    gap = e._foot_ball_gap()[near]
    assert (gap >= .099).all() and (gap <= .251).all()
    direction = quat_rotate_inverse(e.base_quat[near], e.ball_pos[near] - e.base_pos[near])
    assert (direction[:, 0] > 0).all()
    assert (torch.atan2(direction[:, 1], direction[:, 0]).abs() < 1e-4).all()
    torch.testing.assert_close(e.dof_pos[near], e.default_dof_pos[near])
    assert e.qvel[near].eq(0).all()
    assert torch.equal(e.near_start_episode, near)


def test_fixed_quotas_domains_and_partial_reset_preserve_group():
    env = load_environment('cpu', 64)
    assert_mixture(env)  # Applies from the first reset, including standing phase.
    e = env.env
    groups = e.reset_group.clone()
    before = e.qpos.clone()
    ids = torch.arange(0, 64, 2)
    e._reset_idx(ids)
    assert_mixture(env)
    torch.testing.assert_close(e.reset_group, groups)
    torch.testing.assert_close(e.qpos[1::2], before[1::2], rtol=0, atol=0)
    e.near_start_success_rate = 1.
    e.near_start_trials = 1000
    e.survival_rate_5s = 1.
    e.set_training_iteration(10000)
    assert e.near_start_fraction == .5 and e.near_start_progress == 0
    e.reset()
    assert_mixture(env)


def test_circle_sampler_is_uniform_in_area():
    env = load_environment('cpu', 64)
    e = env.env
    ids = (e.reset_group == 1).nonzero().flatten().repeat(1000)
    xy = e._sample_spawn_xy(ids)
    assert (xy.norm(dim=-1) <= 3.5).all()
    assert abs(xy.square().sum(-1).mean().item() / 3.5**2 - .5) < .02
    assert xy.mean(0).abs().max() < .08


def test_safe_near_starts_have_no_robot_ball_penetration():
    env = load_environment('cpu', 64)
    e = env.env
    data = mujoco.MjData(e.model)
    ball = e.model.geom('soccer_ball').id
    floor = e.model.geom('floor').id
    for row in e.qpos[e.reset_group == 2]:
        data.qpos[:] = row.numpy()
        mujoco.mj_forward(e.model, data)
        assert not any(ball in c.geom and floor not in c.geom and c.dist < 0 for c in data.contact)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_gpu_mixture_geometry():
    env = load_environment('cuda:0', 64)
    assert_mixture(env)
    env.step(torch.zeros(64, 22, device='cuda:0'))
    env.env.batch.check()
