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


def test_history_advances_once_and_does_not_cross_resets():
    wrapper = load_environment('cpu', 2)
    before = wrapper.history.clone()
    wrapper.env.episode_length_buf[0] = 100000
    _, terminal, _, info = wrapper.step(torch.zeros(2, 22))
    assert info['done'][0]
    torch.testing.assert_close(wrapper.history[1, :-1], before[1, 1:])
    torch.testing.assert_close(wrapper.history[0], wrapper.env.obs_buf[0].expand(4, -1))
    torch.testing.assert_close(terminal[0, -wrapper.history_dim:].reshape(4, -1)[:-1], before[0, 1:])


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


def test_policy_history_contains_no_privileged_state_and_model_shifts_it():
    wrapper = load_environment('cpu', 2)
    e = wrapper.env
    agent = SoccerAgent(wrapper.state_dim, width=16, history_dim=wrapper.history_dim)
    agent.configure_task({'root_quat': 173+e.root_addr, 'dynamic_ranges': [(0,262), (308,368)]})
    state = wrapper.state()
    corrupt = state.clone(); corrupt[:, 79:-wrapper.history_dim] += 100
    torch.testing.assert_close(agent.policy_observation(agent.normalize(state)),
                               agent.policy_observation(agent.normalize(corrupt)))
    following, _, _ = agent.predict(agent.normalize(state), torch.zeros(2, 22))
    raw = following*agent.scale+agent.mean
    history = raw[:, -wrapper.history_dim:].reshape(2, 4, 79)
    torch.testing.assert_close(history[:, :-1], wrapper.history[:, 1:])
    torch.testing.assert_close(history[:, -1], raw[:, :79])


def test_counterfactual_validation_preserves_training_rng():
    from q1_supermimic.safety import validate_action_response
    wrapper = load_environment('cpu', 4)
    e = wrapper.env
    agent = SoccerAgent(wrapper.state_dim, width=16, history_dim=wrapper.history_dim)
    agent.configure_task({'root_quat':173+e.root_addr, 'physical_start':170,
                          'physical_end':170+e.model.nq+e.model.nv,
                          'dynamic_ranges':[(0,262),(308,368)]})
    before = torch.random.get_rng_state()
    result = validate_action_response(agent, count=4)
    torch.testing.assert_close(torch.random.get_rng_state(), before)
    assert all(torch.isfinite(torch.tensor(v)) for v in result.values())


def test_physical_projection_matches_privileged_observations_and_backpropagates():
    from train import task_layout
    wrapper = load_environment('cpu', 4)
    e = wrapper.env
    agent = SoccerAgent(wrapper.state_dim, width=16, history_dim=wrapper.history_dim)
    agent.configure_task(task_layout(wrapper))
    current = wrapper.state()
    action = torch.zeros(4, 22)
    _, target, _, _ = wrapper.step(action)
    projected = agent.project_observations(current, target, action)
    # Visibility is held over imagination; all physical critic features are exact.
    keep = torch.ones(91, dtype=torch.bool); keep[8] = False
    torch.testing.assert_close(projected[:,79:170][:,keep], target[:,79:170][:,keep], atol=2e-5, rtol=2e-5)
    agent.make_optimizers()
    loss = agent.improve_model_policy(current, 2, .025)
    assert torch.isfinite(torch.tensor(loss))


def test_velocity_integration_tracks_real_control_step_positions():
    from train import task_layout
    wrapper = load_environment('cpu', 4)
    agent = SoccerAgent(wrapper.state_dim, width=16, history_dim=wrapper.history_dim)
    agent.configure_task(task_layout(wrapper))
    torch.manual_seed(104729)
    errors = []
    for _ in range(32):
        current = wrapper.state()
        _, target, _, info = wrapper.step(.025*torch.randn(4, 22))
        integrated = agent.integrate_positions(current, target)
        errors.append((integrated[:,170:206]-target[:,170:206]).abs()[~info['done']])
    # An approximate control-period integrator, checked against real substeps.
    error = torch.cat(errors)
    assert error[:,:3].mean() < .002  # root translation, metres
    assert error[:,7:29].mean() < .005  # joints, radians
    assert error[:,7:29].max() < .04


def test_physical_model_excludes_sensor_noise_and_global_translation():
    from train import task_layout
    w = load_environment('cpu', 4)
    agent = SoccerAgent(w.state_dim, width=16, history_dim=w.history_dim)
    agent.configure_task(task_layout(w))
    state = w.state(); changed = state.clone()
    changed[:,:158] += torch.randn_like(changed[:,:158])
    changed[:,-w.history_dim:] += 5
    for start in (agent.layout['root_xy'], agent.layout['ball_start']):
        changed[:,start:start+2] += torch.tensor([2.,-1.])
    torch.testing.assert_close(agent.physical_features(state), agent.physical_features(changed), atol=1e-6, rtol=1e-6)
    action = torch.zeros(4,22)
    a = agent.predict(agent.normalize(state), action)[0]
    b = agent.predict(agent.normalize(changed), action)[0]
    torch.testing.assert_close(a[:,206:240], b[:,206:240], atol=1e-6, rtol=1e-6)


def test_physical_model_has_all_randomized_inertial_context():
    from train import task_layout
    w = load_environment('cpu', 4)
    agent = SoccerAgent(w.state_dim, width=16, history_dim=w.history_dim)
    agent.configure_task(task_layout(w))
    before = agent.physical_features(w.state())
    w.env.ipos[:,1,0] += .001
    after = agent.physical_features(w.state())
    assert not torch.equal(before, after)
    before = after
    w.env.inertia[:,1,0] *= 1.05
    assert not torch.equal(before, agent.physical_features(w.state()))


def test_balance_dense_reward_matches_real_world_frame_acceleration():
    from train import task_layout
    w = load_environment('cpu', 4)
    e = w.env
    agent = SoccerAgent(w.state_dim, width=16, history_dim=w.history_dim)
    agent.configure_task(task_layout(w))
    for _ in range(12):
        current = w.state()
        action = .025*torch.randn(4,22)
        _, target, _, _ = w.step(action)
        reward, continuation = agent.task_reward(agent.normalize(current), agent.normalize(target), action)
        names = ('survival','upright','base_height','action_rate','root_acc','stance_pose')
        actual = sum(e.extras['rew_terms'][name] for name in names)
        # Compare dense terms before smooth termination shaping (separately approximated).
        dense = reward/.1 + 20*(1-continuation)
        torch.testing.assert_close(dense, actual, atol=2e-5, rtol=2e-4)


def test_balance_value_excludes_absolute_field_location_and_camera_noise():
    from train import task_layout
    w = load_environment('cpu',4)
    agent = SoccerAgent(w.state_dim,width=16,history_dim=w.history_dim)
    agent.configure_task(task_layout(w))
    state = w.state(); changed = state.clone()
    changed[:,:79] += torch.randn_like(changed[:,:79])
    changed[:,-w.history_dim:] += torch.randn_like(changed[:,-w.history_dim:])
    changed[:,88:90] += 3  # true goal vector changes under translation
    for start in (agent.layout['root_xy'],agent.layout['ball_start']):
        changed[:,start:start+2] += 3
    torch.testing.assert_close(agent.value_features(agent.normalize(state)),
                               agent.value_features(agent.normalize(changed)),atol=1e-6,rtol=1e-6)
    state[:,agent.layout['progress_index']] = 1
    changed = state.clone(); changed[:,88:90] += 1
    assert not torch.equal(agent.value_features(agent.normalize(state)),
                           agent.value_features(agent.normalize(changed)))


def test_football_feedback_projection_and_dense_rewards_match_real_transitions():
    from train import task_layout
    torch.manual_seed(1420011)
    w = load_environment('cpu', 8)
    e = w.env
    e.balance_release_iteration = 0
    e.soccer_progress = .1
    e.set_training_iteration(0)
    w.reset()
    agent = SoccerAgent(w.state_dim, width=16, history_dim=w.history_dim)
    agent.configure_task(task_layout(w))
    p = agent.layout['progress_index']
    for _ in range(160):
        current = w.state()
        action = torch.zeros(8,22)
        _, target, _, info = w.step(action)
        physical_oracle = target.clone()
        physical_oracle[:,p+1:p+11] = current[:,p+1:p+11]
        projected = agent.project_observations(current,physical_oracle,action)
        live = ~info['done']
        torch.testing.assert_close(projected[live,p+1:p+11],target[live,p+1:p+11],atol=2e-5,rtol=2e-5)
        reward, continuation = agent.task_reward(agent.normalize(current),agent.normalize(target),action)
        names = ('survival','upright','base_height','action_rate','root_acc','stance_pose',
                 'ball_distance','goal_distance','pos_still','near_ball_stall','face_ball_pitch','face_ball_yaw')
        actual = sum(e.extras['rew_terms'][name] for name in names)
        dense = reward/.1+20*(1-continuation)
        torch.testing.assert_close(dense[live],actual[live],atol=2e-5,rtol=2e-4)
    # The full projected rollout must still support actor gradients through FK.
    agent.make_optimizers()
    loss = agent.improve_model_policy(w.state(),2,.025)
    assert torch.isfinite(torch.tensor(loss))


def test_camera_geometry_matches_physics_and_visibility_changes_in_rollouts():
    from train import task_layout
    from utils.torch_math import quat_mul, quat_rotate, quat_rotate_inverse
    w = load_environment('cpu',4)
    e = w.env
    agent = SoccerAgent(w.state_dim,width=16,history_dim=w.history_dim)
    agent.configure_task(task_layout(w))
    state = w.state()
    optical = quat_mul(e.head_quat,state.new_tensor([.5,.5,.5,.5]).expand(4,-1))
    expected = quat_rotate_inverse(optical,e.ball_pos-e.camera_pos)
    torch.testing.assert_close(agent.camera_ball(state),expected,atol=2e-5,rtol=2e-5)
    b,p = agent.layout['ball_start'],agent.layout['progress_index']
    inside = state.clone()
    outside = state.clone()
    inside[:,b:b+3] = e.camera_pos+quat_rotate(optical,state.new_tensor([0.,0.,1.]).expand(4,-1))
    outside[:,b:b+3] = e.camera_pos+quat_rotate(optical,state.new_tensor([0.,2.,1.]).expand(4,-1))
    assert agent.camera_visibility(inside).all()
    assert not agent.camera_visibility(outside).any()
    for current,target,visible in [(outside,inside,1.),(inside,outside,0.)]:
        current[:,p-2] = e.dt/60  # Next observation is a fresh camera sample.
        current[:,307] = 0       # Zero-delay case isolates geometry from the queue.
        noise = state.new_tensor([0.,0.,.5]).expand(4,-1)
        projected = agent.project_observations(current,target,torch.zeros(4,22),visual_noise=noise)
        torch.testing.assert_close(projected[:,8],torch.full((4,),visible))
        torch.testing.assert_close(projected[:,87],torch.full((4,),visible))
        if not visible:
            torch.testing.assert_close(projected[:,6:8],torch.zeros(4,2))
