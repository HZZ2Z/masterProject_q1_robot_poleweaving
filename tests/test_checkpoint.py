import torch
from q1_supermimic.environment import load_environment
from q1_supermimic.learning import SoccerAgent
from train import save_checkpoint, validate_checkpoint


def test_checkpoint_preserves_policy_optimizer_and_curriculum(tmp_path):
    env = load_environment('cpu', 2)
    agent = SoccerAgent(env.state_dim, width=16, horizon=2)
    agent.make_optimizers()
    state = env.state()
    agent.calibrate(state)
    agent.optimize(agent.actor(state[:, :79]).square().sum() + agent.actor(state[:, :79]).sum(),
                   agent.actor_optimizer, agent.actor.parameters())
    env.env.soccer_progress = .13
    env.env.near_start_progress = .2
    env.env.near_start_trials = 321
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, agent, env, {'width': 16}, 42)
    saved = torch.load(path, weights_only=False)
    other = SoccerAgent(env.state_dim, width=16, horizon=2)
    other.make_optimizers()
    other.load_state_dict(saved['agent'])
    other.actor_optimizer.load_state_dict(saved['optimizers'][1])
    torch.testing.assert_close(agent.act(state), other.act(state), rtol=0, atol=0)
    validate_checkpoint(saved, env)
    assert saved['near_start']['progress'] == .2
    assert saved['near_start']['trials'] == 321
    env.env.near_start_progress = 0
    env.env.restore_near_start_state(saved['near_start'])
    assert env.env.near_start_progress == .2
    assert saved['iteration'] == 42
    assert saved['curriculum']['soccer_progress'] == .13
    assert other.actor_optimizer.state_dict()['state']


def test_old_reward_checkpoint_rejected():
    import pytest
    env = load_environment('cpu', 2)
    with pytest.raises(ValueError, match='Incompatible task'):
        validate_checkpoint({'version': 1, 'state_dim': env.state_dim}, env)
    with pytest.raises(ValueError, match='Incompatible task'):
        validate_checkpoint({'version': 2, 'state_dim': env.state_dim,
                             'task_signature': 'different-reward'}, env)


def test_curriculum_only_migration_rejects_reward_changes():
    import copy
    import pytest
    env = load_environment('cpu', 4)
    saved = {'version': 5, 'state_dim': env.state_dim, 'task_signature': 'old',
             'environment': copy.deepcopy(env.cfg)}
    saved['environment']['curriculum']['release_active_5s_fraction'] = .8
    validate_checkpoint(saved, env, migrate_curriculum=True)
    saved['environment']['rewards']['version'] = -1
    with pytest.raises(ValueError, match='cannot change'):
        validate_checkpoint(saved, env, migrate_curriculum=True)


def test_near_group_gets_soccer_before_full_groups():
    env = load_environment('cpu', 8).env
    env.episode_length_buf[:] = 250
    env.survival_rate_5s = .73
    env.set_training_iteration(3206)
    assert not env.balance_phase and env.soccer_progress > 0
    assert (env.task_progress[env.reset_group == 2] > 0).all()
    assert (env.task_progress[env.reset_group != 2] == 0).all()
    assert env.reference_init_probability == 0
    env.balance_release_iteration = 0
    env.soccer_progress = 1.
    env.near_start_success_rate = .1
    env.set_training_iteration(3206)
    assert (env.task_progress == 1).all()


def test_old_policy_versions_cannot_enter_fresh_model_route():
    import pytest
    env = load_environment('cpu', 2)
    old = {'version': 3, 'state_dim': env.state_dim, 'task_signature': env.task_signature}
    with pytest.raises(ValueError, match='Incompatible'):
        validate_checkpoint(old, env)
    old['version'] = 4
    with pytest.raises(ValueError, match='Incompatible'):
        validate_checkpoint(old, env)
