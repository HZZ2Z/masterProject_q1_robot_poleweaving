"""Learning semantics: episode boundaries, gradients, and real terminal states."""
import torch
from q1_supermimic.learning import SoccerAgent, TemporalReplay
from q1_supermimic.environment import load_environment


def test_replay_wrap_preserves_time_and_target():
    replay = TemporalReplay(5, 1, 3, 'cpu')
    for t in range(9):
        replay.add(torch.full((1, 3), float(t)), torch.zeros(1, 22),
                   torch.full((1, 3), float(t + 1)), torch.zeros(1),
                   torch.zeros(1), torch.zeros(1))
    state, _, following, *_ = replay.sample(100, 4)
    torch.testing.assert_close(state[:, 1:] - state[:, :-1], torch.ones_like(state[:, 1:]))
    torch.testing.assert_close(following - state, torch.ones_like(state))
    assert state.min() >= 4 and state.max() <= 8


def test_real_actor_update_preserves_model_weights():
    torch.manual_seed(3)
    agent = SoccerAgent(12, obs_dim=5, width=16, horizon=3)
    agent.make_optimizers()
    states = torch.randn(8, 12)
    before_actor = [p.clone() for p in agent.actor.parameters()]
    before_model = [p.clone() for p in agent.dynamics.parameters()]
    latent = torch.randn(1, 8, 22) * .1
    terminal = torch.ones(1, 8)
    rollout = (states[None], .1*latent.tanh(), states[None], latent[..., 0], terminal, terminal, latent)
    losses = agent.learn_real_rollout(rollout, .01, critic_epochs=1)
    assert all(torch.isfinite(torch.tensor(losses)))
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, agent.actor.parameters()))
    for a, b in zip(before_model, agent.dynamics.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_autoregressive_loss_never_crosses_reset():
    torch.manual_seed(4)
    agent = SoccerAgent(12, obs_dim=5, width=16, horizon=3, lr=0)
    agent.make_optimizers()
    states = torch.randn(8, 3, 12)
    actions = torch.randn(8, 3, 22)
    targets = torch.randn(8, 3, 12)
    reward = torch.randn(8, 3)
    dones = torch.ones(8, 3)  # Every first transition ends the episode.
    fall = torch.ones(8, 3)
    loss = agent.fit_model((states, actions, targets, reward, dones, fall))
    targets[:, 1:] = 999
    reward[:, 1:] = 999
    after = agent.fit_model((states, actions, targets, reward, dones, fall))
    assert abs(loss - after) < 1e-6


def test_terminal_target_is_captured_before_timeout_reset():
    env = load_environment('cpu', 2)
    env.env.episode_length_buf[0] = 100000
    next_state, terminal, _, info = env.step(torch.zeros(2, 22))
    assert info['done'][0] and info['timeout'][0] and not info['fall'][0]
    assert not info['terminated'][0]
    # Qpos follows 79 actor and 91 critic coordinates in model state.
    qpos_start = 170
    assert not torch.allclose(next_state[0, qpos_start:qpos_start+3], terminal[0, qpos_start:qpos_start+3])
    assert torch.isfinite(terminal).all()


def test_soccer_v4_feedback_in_model_state_and_reward_terms():
    env = load_environment('cpu', 4)
    e = env.env
    e.balance_release_iteration = 0
    e.soccer_progress = 1.
    e.cfg['curriculum']['near_start']['initial_fraction'] = 1.
    e.set_training_iteration(3000)
    e.reset()
    assert e.near_start_episode.sum() == 2
    before = env.state()
    e.near_ball_idle_s += .7
    assert not torch.equal(before, env.state())
    _, terminal, reward, info = env.step(torch.zeros(4, 22))
    assert torch.isfinite(terminal).all()
    terms = sum(v for k, v in info['metrics'].items() if k.startswith('reward_terms/'))
    torch.testing.assert_close(reward.mean(), terms)
    assert 'reward_terms/near_ball_stall' in info['metrics']
    assert e.cfg['rewards']['version'] == 4
    assert e.cfg['rewards']['scales']['kick_ball'] == 2
    assert e.cfg['rewards']['scales']['side_kick_ball'] == 8


def test_visual_history_advances_once_with_terminal_capture():
    env = load_environment('cpu', 2)
    for reset_first in (False, True):
        env.env.ball_obs_buffer[:] = torch.arange(60).reshape(1, 20, 3)
        before = env.env.ball_obs_buffer.clone()
        if reset_first:
            env.env.episode_length_buf[0] = 100000
        env.step(torch.zeros(2, 22))
        torch.testing.assert_close(env.env.ball_obs_buffer[1, 1:], before[1, :-1])


def test_real_goal_is_terminal_for_value_bootstrap():
    wrapper = load_environment('cpu', 2)
    e = wrapper.env
    e.qpos[0, e.ball_addr:e.ball_addr+2] = torch.tensor([7.1, 0.])
    e.qvel[0, e.ball_vel_addr:e.ball_vel_addr+3] = 0
    e.goal_cnt[0] = 49
    _, _, _, info = wrapper.step(torch.zeros(2, 22))
    assert info['success'][0] and info['done'][0]
    assert not info['fall'][0] and info['terminated'][0]
