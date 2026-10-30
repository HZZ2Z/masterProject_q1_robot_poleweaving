"""Q1 asset, reference, reset, and real batched rollout regression checks."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from mujoco.batch import Batch
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
import yaml

from envs.q1 import Q1, mirror_joints, mirror_observations
from utils.kinematics import ResetKinematics
from utils.motion import Motion
from utils.torch_math import quat_rotate


@pytest.fixture
def cfg():
    result = yaml.safe_load(Path('envs/Q1.yaml').read_text())
    result['basic']['device'] = 'cpu'
    result['env']['num_envs'] = 4
    result['physics']['num_threads'] = 2
    result['curriculum']['near_start']['enabled'] = False
    result['curriculum']['reset_mixture']['enabled'] = False
    return result


def test_official_model_and_effort_limits(cfg):
    root = ET.parse(cfg['asset']['file']).getroot()
    joints = [j for j in root.findall('joint') if j.get('type') != 'fixed']
    assert [j.get('name') for j in joints] == cfg['asset']['dof_names']
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    assert model.nu == 22
    assert [model.actuator(i).name for i in range(model.nu)] == cfg['asset']['dof_names']
    np.testing.assert_allclose(model.actuator_ctrlrange[:, 1], [float(j.find('limit').get('effort')) for j in joints])
    mass = sum(float(link.find('inertial/mass').get('value')) for link in root.findall('link') if link.find('inertial/mass') is not None)
    np.testing.assert_allclose(model.body_subtreemass[model.body('pelvis').id], mass, rtol=1e-6)
    for mesh in root.findall('.//mesh'):
        assert (Path(cfg['asset']['file']).parent / mesh.get('filename')).is_file()


def test_reset_fk_and_camera_match_mujoco(cfg):
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    data = mujoco.MjData(model)
    names = cfg['asset']['dof_names']
    frames = cfg['asset']['foot_names'] + [cfg['asset']['head_name']]
    fk = ResetKinematics(cfg['asset']['file'], names, frames, 'cpu')
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(-.2, .2, 22).astype(np.float32)
        result = fk(torch.from_numpy(q).unsqueeze(0))
        data.qpos[:7] = [0, 0, .417, 1, 0, 0, 0]
        for i, name in enumerate(names):
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = q[i]
        mujoco.mj_forward(model, data)
        for name in frames:
            np.testing.assert_allclose(result[name][0][0].numpy() + [0, 0, .417], data.body(name).xpos, atol=2e-6)
            quat = data.body(name).xquat[[1, 2, 3, 0]]
            assert abs(np.dot(quat, result[name][1][0].numpy())) == pytest.approx(1, abs=1e-6)
        head_pos, head_quat = result[cfg['asset']['head_name']]
        expected = head_pos[0].numpy() + Rotation.from_quat(head_quat[0].numpy()).apply(cfg['asset']['camera_pos']) + [0, 0, .417]
        np.testing.assert_allclose(data.sensor('camera_position').data, expected, atol=2e-6)


def test_mirror_involution_and_foot_reflection(cfg):
    rng = np.random.default_rng(2)
    obs = rng.standard_normal((2, 50, 79))
    np.testing.assert_array_equal(mirror_observations(mirror_observations(obs)), obs)
    q = torch.from_numpy(rng.uniform(-.2, .2, (20, 22)).astype(np.float32))
    torch.testing.assert_close(mirror_joints(mirror_joints(q)), q)
    scale = torch.tensor(cfg['control']['action_scale'])
    torch.testing.assert_close(mirror_joints(q * scale), mirror_joints(q) * scale)
    fk = ResetKinematics(cfg['asset']['file'], cfg['asset']['dof_names'], cfg['asset']['foot_names'], 'cpu')
    original, reflected = fk(q), fk(mirror_joints(q))
    for left, right in [('left_ankle_roll_link', 'right_ankle_roll_link'), ('right_ankle_roll_link', 'left_ankle_roll_link')]:
        torch.testing.assert_close(reflected[left][0], original[right][0] * torch.tensor([1, -1, 1]), atol=4e-5, rtol=0)


def test_motion_limits_schema_and_amp(cfg):
    manifest = json.loads(Path('data/q1/retarget_manifest.json').read_text())
    files = sorted(Path('data/q1').glob('*.csv'))
    assert len(files) == sum(v['accepted'] for v in manifest['files'].values())
    assert len(files) >= 20
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    ids = [model.joint(n).id for n in cfg['asset']['dof_names']]
    limits = model.jnt_range[ids]
    for path in files:
        with path.open() as f:
            header = next(csv.reader(f))
        assert header[13:35] == ['q_' + n for n in cfg['asset']['dof_names']]
        a = np.loadtxt(path, delimiter=',', skiprows=1)
        assert a.shape[1] == 57 and np.isfinite(a).all()
        assert np.all(a[:, 13:35] >= limits[:, 0] - 1e-6)
        assert np.all(a[:, 13:35] <= limits[:, 1] + 1e-6)
        np.testing.assert_allclose(np.linalg.norm(a[:, 3:7], axis=1), 1, atol=1e-6)
        np.testing.assert_allclose(a[:, 35:], np.gradient(a[:, 13:35], .02, axis=0), atol=5e-6)
        assert a[:, 2].min() > cfg['rewards']['terminate_height']
    motion = Motion(cfg, 'cpu')
    assert motion.sample_batch(8)['dof_pos'].shape == (8, 22)
    a, b = next(motion.feed_forward_generator(1, 8))
    assert a.shape == b.shape == (8, 59)
    assert torch.isfinite(a).all()


def test_batch_step_matches_serial_physics(cfg):
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    batch = Batch(model, 2, nthread=2)
    qpos, qvel, ctrl = batch.bind('qpos'), batch.bind('qvel'), batch.bind('ctrl')
    serial = [mujoco.MjData(model) for _ in range(2)]
    for i in range(2):
        qpos[i] = model.qpos0
        serial[i].qpos[:] = qpos[i]
        ctrl[i] = i * .1
        serial[i].ctrl[:] = ctrl[i]
    batch.forward()
    for _ in range(10):
        batch.step()
        for data in serial:
            mujoco.mj_step(model, data)
    for i, data in enumerate(serial):
        np.testing.assert_allclose(qpos[i], data.qpos, atol=1e-12)
        np.testing.assert_allclose(qvel[i], data.qvel, atol=1e-12)


def test_partial_reset_and_training_observation_contract(cfg):
    torch.set_num_threads(2)
    env = Q1(cfg)
    obs, info = env.reset()
    assert obs.shape == (4, 79)
    assert info['critic_obs'].shape == (4, 91)
    assert info['privileged_obs'].shape == (4, 14)
    assert info['amp_obs'].shape == (4, 59)
    before = env.qpos[1:].clone()
    env._reset_idx(torch.tensor([0]))
    np.testing.assert_array_equal(env.qpos[1:], before)
    for _ in range(15):
        obs, reward, done, info = env.step(torch.randn(4, 22) * .1)
        for tensor in (obs, reward, info['critic_obs'], info['amp_obs'], info['last_amp_obs']):
            assert torch.isfinite(tensor).all()
        assert done.dtype == torch.bool
        assert bool((abs(env.ctrl) <= torch.tensor(env.model.actuator_ctrlrange[:, 1]) + 1e-9).all())
    # A forced fall terminates, returns the terminal AMP frame, and resets.
    env.qpos[0, env.root_addr + 2] = .05
    _, _, done, info = env.step(torch.zeros(4, 22))
    assert done[0] and info['fall'][0] == 1
    assert env.qpos[0, env.root_addr + 2] > .2


def test_timeout_reset_keeps_truncation_flag(cfg):
    env = Q1(cfg)
    env.set_training_iteration(0)  # neutral stance: isolate timeout from random reference falls
    env.reset()
    env.episode_length_buf[:] = int(cfg['rewards']['episode_length_s'] / env.dt) + 1
    _, _, done, info = env.step(torch.zeros(4, 22))
    assert done.all() and info['time_outs'].all()


def test_amp_reference_layout_matches_batched_simulation(cfg):
    env = Q1(cfg)
    env.reset()
    row = np.loadtxt('data/q1/left_kick_1.csv', delimiter=',', skiprows=1)[0]
    quat = torch.tensor(row[3:7], dtype=torch.float32).expand(4, -1)
    a, v = env.root_addr, env.root_vel_addr
    env.qpos[:, a:a + 3] = torch.as_tensor(row[:3], dtype=torch.float32)
    env.qpos[:, [a + 3, a + 4, a + 5, a + 6]] = torch.as_tensor(row[[6, 3, 4, 5]], dtype=torch.float32)
    env.qpos[:, env.pos_addrs] = torch.as_tensor(row[13:35], dtype=torch.float32)
    env.qvel[:, v:v + 3] = torch.as_tensor(quat_rotate(quat, torch.tensor(row[7:10], dtype=torch.float32).expand(4, -1)).numpy(), dtype=torch.float32)
    env.qvel[:, v + 3:v + 6] = torch.as_tensor(row[10:13], dtype=torch.float32)
    env.qvel[:, env.vel_addrs] = torch.as_tensor(row[35:57], dtype=torch.float32)
    env.batch.forward()
    env._sync_states()
    env._refresh_task_state()
    expected = env.motion_sampler.data[0].clone()
    expected[31:53] *= cfg['normalization']['dof_vel']
    torch.testing.assert_close(env._compute_amp_obs()[0], expected, atol=2e-6, rtol=0)




def test_reset_clears_solver_history_and_separates_ball(cfg):
    cfg['env']['num_envs'] = 256
    env = Q1(cfg)
    env.reset()
    env.warmstart[:] = 123.
    env._reset_idx(torch.tensor([0]))
    np.testing.assert_array_equal(env.warmstart[0], 0.)
    np.testing.assert_array_equal(env.warmstart[1:], 123.)
    # Check the actual collision engine after multiple randomized resets.
    data = mujoco.MjData(env.model)
    feet = {env.model.body(n).id for n in cfg['asset']['foot_names']}
    floor = env.model.geom('floor').id
    ball = env.model.geom('soccer_ball').id
    for _ in range(3):
        env.reset()
        distance = np.linalg.norm(env.qpos[:, env.ball_addr:env.ball_addr + 2] - env.qpos[:, :2], axis=1)
        assert (distance >= .65).all()
        for row in env.qpos:
            data.qpos[:] = row
            mujoco.mj_forward(env.model, data)
            for contact in data.contact:
                assert ball not in contact.geom or floor in contact.geom
                if floor in contact.geom and contact.dist < -.002:
                    assert any(env.model.geom_bodyid[g] in feet for g in contact.geom)


def test_floor_supports_ball_weight(cfg):
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    data = mujoco.MjData(model)
    ball_joint = model.joint('ball_free').id
    a = model.jnt_qposadr[ball_joint]
    data.qpos[:3] = [10, 10, 1.5]
    data.qpos[a:a + 3] = [0, 0, .5]
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert data.qpos[a + 2] == pytest.approx(.1, abs=.001)
    force = 0.
    for i, contact in enumerate(data.contact):
        if model.geom('soccer_ball').id in contact.geom:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, wrench)
            force += wrench[0]
    assert force == pytest.approx(model.body_mass[model.body('ball').id] * 9.81, rel=.01)






def test_reference_positions_are_reachable_and_resets_have_speed_margin(cfg):
    defaults = cfg['init_state']['default_joint_angles']
    neutral = np.array([defaults.get(n, 0.) for n in cfg['asset']['dof_names']])
    scale = np.array(cfg['control']['action_scale'])
    for path in Path(cfg['motion']['motion_path']).glob('*.csv'):
        q = np.loadtxt(path, delimiter=',', skiprows=1)[:, 13:35]
        assert (abs(q - neutral) <= scale + 1e-6).all(), str(path)
    motion = Motion(cfg, 'cpu')
    state = motion.sample_batch(4096)
    squared_speed = state['base_lin_vel'].square().sum(-1) + state['base_ang_vel'].square().sum(-1)
    assert (squared_speed < .5 * cfg['rewards']['terminate_vel']).all()


def test_nominal_pd_stance_remains_supported_for_six_seconds(cfg):
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    data = mujoco.MjData(model)
    names = cfg['asset']['dof_names']
    joint_ids = [model.joint(n).id for n in names]
    pa, va = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    defaults = cfg['init_state']['default_joint_angles']
    neutral = np.array([defaults.get(n, 0.) for n in names])
    kp = np.array([next(v for k, v in cfg['control']['stiffness'].items() if k in n) for n in names])
    kd = np.array([next(v for k, v in cfg['control']['damping'].items() if k in n) for n in names])
    data.qpos[pa] = neutral
    data.qpos[2] = .427
    ball = model.joint('ball_free').qposadr[0]
    data.qpos[ball:ball + 3] = [2, 0, .1]
    for _ in range(3000):
        torque = kp * (neutral - data.qpos[pa]) - kd * data.qvel[va]
        assert np.all(abs(torque) <= model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)
        assert data.qpos[2] > .35
    assert np.isfinite(data.qpos).all()




def test_curriculum_requires_standing_before_soccer(cfg):
    cfg['curriculum'] = {'balance_until_iteration': 10, 'soccer_transition_iterations': 20,
                         'release_active_5s_fraction': .8}
    env = Q1(cfg)
    assert env.set_training_iteration(0)
    env.reset()
    assert env.balance_phase and not cfg['motion']['reference_state_init']
    assert env.amp_reward_scale == 0 and env.reward_scales['ball_distance'] == 0
    np.testing.assert_array_equal(env.qvel[:, env.vel_addrs], 0.)
    assert not env.set_training_iteration(10)
    assert env.balance_phase  # Iteration count alone cannot release the gate.
    env.episode_length_buf[:] = 250
    assert env.set_training_iteration(10)
    assert not env.balance_phase and cfg['motion']['reference_state_init']
    assert env.amp_reward_scale == pytest.approx(.05)
    env.set_training_iteration(30)
    assert env.amp_reward_scale == 1.




def test_adaptive_curriculum_retreats_and_preserves_live_state(cfg):
    cfg['curriculum'] = {'balance_until_iteration': 0, 'release_active_5s_fraction': .8,
                         'soccer_transition_iterations': 10, 'adaptive': True,
                         'advance_active_5s_fraction': .7, 'retreat_active_5s_fraction': .4}
    env = Q1(cfg)
    env.reset()
    env.episode_length_buf[:] = 250
    before = env.qpos.clone()
    env.set_training_iteration(0)
    assert env.amp_reward_scale == pytest.approx(.1)
    assert env.reference_init_probability == pytest.approx(.1)
    np.testing.assert_array_equal(env.qpos, before)
    env.episode_length_buf[:] = 0
    env.set_training_iteration(1)
    assert env.amp_reward_scale == 0
    assert env.reference_init_probability == 0
    assert env.reward_scales['goal'] == 0


def test_reference_resets_are_mixed_instead_of_replacing_every_environment(cfg):
    cfg['env']['num_envs'] = 256
    env = Q1(cfg)
    env.reference_init_probability = .2
    env.reset()
    moving = np.linalg.norm(env.qvel, axis=1) > 1e-5
    assert .1 < moving.mean() < .3


def test_no_robot_ball_contact_means_no_random_goals(cfg):
    assert cfg['randomization']['kick_ball_prob'] == 0
    assert cfg['randomization']['reset_ball_prob'] == 0
    cfg['env']['num_envs'] = 16
    cfg['motion']['reference_state_init'] = False
    env = Q1(cfg)
    env.reset()
    contype = env.batch.expand('geom_contype')
    contype[:, env.model.geom('soccer_ball').id] = 4
    # Put the ball near the goal, but disable robot-ball collisions entirely.
    env.qpos[:, env.ball_addr:env.ball_addr + 3] = torch.as_tensor([6.45, 0., .11], dtype=torch.float32)
    success = 0
    for _ in range(500):
        _, _, _, info = env.step(torch.zeros(16, 22))
        success += int(info['success'].sum())
    assert success == 0


def test_simultaneous_goal_and_fall_is_not_a_success_or_timeout(cfg):
    env = Q1(cfg)
    env.reset()
    env.base_pos[:, 2] = .1
    env.ball_pos[:, :2] = torch.tensor([7.2, 0.])
    env.goal_cnt[:] = 49
    env._check_termination()
    assert env.extras['fall'].all()
    assert not env.extras['success'].any()
    assert not env.time_out_buf.any()


def test_five_second_cohort_counts_failures_and_does_not_double_count(cfg):
    env = Q1(cfg)
    env.reset()
    env.reset_buf[:] = False
    env.reset_buf[0] = True
    env.episode_length_buf[:] = torch.tensor([10, 250, 20, 250])
    env.extras['fall'] = torch.tensor([1., 0., 0., 0.])
    env.extras['success'] = torch.zeros(4)
    env._record_survival_trials()
    assert env.survival_rate_5s == pytest.approx(2 / 3)
    before = env._survival_trial_mass.clone()
    env._record_survival_trials()
    torch.testing.assert_close(env._survival_trial_mass, before)
    env.cfg['curriculum'] = {'balance_until_iteration': 0, 'release_active_5s_fraction': .8,
                             'use_cohort_gate': True}
    env.episode_length_buf[:] = 300
    env.set_training_iteration(1)
    assert env.balance_phase  # Long-lived survivors cannot hide failed reset trials.


def test_curriculum_observes_a_window_before_changing_difficulty(cfg):
    cfg['curriculum'] = {'balance_until_iteration': 0, 'release_active_5s_fraction': .8,
                         'soccer_transition_iterations': 10, 'adaptive': True,
                         'adjust_interval_iterations': 3, 'max_progress_step': .1}
    env = Q1(cfg)
    env.reset()
    env.episode_length_buf[:] = 300
    progress = []
    for iteration in range(6):
        env.set_training_iteration(iteration)
        progress.append(env.amp_reward_scale)
    assert progress == pytest.approx([0., 0., .1, .1, .1, .2])
    env.episode_length_buf[:] = 0
    for iteration in range(6, 9):
        env.set_training_iteration(iteration)
    assert env.amp_reward_scale == 0


def test_idle_penalties_do_not_overwhelm_upright_survival_reward(cfg):
    weights = cfg['rewards']['scales']
    # Worst camera-direction penalties plus idling, before any AMP bonus.
    stationary_rate = (weights['survival'] + weights['pos_still'] +
                       np.pi ** 2 * (weights['face_ball_pitch'] + weights['face_ball_yaw']))
    assert stationary_rate > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for Warp')
def test_warp_gpu_contact_reset_and_model_constants(cfg):
    cfg['basic']['device'] = 'cuda:0'
    cfg['motion']['reference_state_init'] = False
    env = Q1(cfg)
    env.set_training_iteration(0)
    env.reset()
    assert env.qpos.is_cuda and env.ctrl.is_cuda
    # Randomized model constants and forward kinematics match native MuJoCo.
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    for name in ('body_mass', 'body_inertia', 'body_ipos', 'geom_friction'):
        getattr(model, name)[:] = env.batch.expand(name)[0].cpu().numpy()
    mujoco.mj_setConst(model, mujoco.MjData(model))
    np.testing.assert_allclose(env.batch.expand('body_subtreemass')[0].cpu(), model.body_subtreemass, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(env.batch.expand('dof_invweight0')[0].cpu(), model.dof_invweight0, rtol=2e-4, atol=1e-5)
    data = mujoco.MjData(model)
    data.qpos[:] = env.qpos[0].cpu()
    data.qvel[:] = env.qvel[0].cpu()
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(env.outputs['xpos'][0].cpu(), data.xpos, atol=3e-6)
    before = env.qpos[1:].clone()
    env._reset_idx(torch.tensor([0], device='cuda:0'))
    torch.testing.assert_close(env.qpos[1:], before, rtol=0, atol=0)
    # Check nominal support separately: zero-action PD is not a balance
    # controller for every randomized COM and joint calibration offset.
    for key in ('base_com', 'other_com'):
        cfg['randomization'][key]['range'] = [0., 0.]
    for key in ('base_mass', 'other_mass', 'dof_stiffness', 'dof_damping'):
        cfg['randomization'][key]['range'] = [1., 1.]
    defaults = cfg['init_state']['default_joint_angles']
    env.default_dof_pos[:] = torch.tensor([defaults.get(n, defaults['default']) for n in env.dof_names], device='cuda:0')
    env.reset()
    for _ in range(300):
        obs, reward, done, info = env.step(torch.zeros(4, 22, device='cuda:0'))
        assert not done.any(), 'Nominal PD stance fell in Warp'
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    env.batch.check()
    assert (env.base_pos[:, 2] > .39).all()
    assert torch.allclose(env.ball_pos[:, 2], torch.full((4,), .1, device='cuda:0'), atol=.002)
    # The resting ball's contact force supports its randomized weight.
    torch.testing.assert_close(env.contact_forces[:, env.ball_indice, 2],
                               env.mass[:, env.ball_indice] * 9.81, rtol=.03, atol=.03)


def test_q1_foot_box_matches_collision_mesh(cfg):
    model = mujoco.MjModel.from_xml_path(cfg['asset']['mujoco_file'])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bounds = np.array(cfg['asset']['foot_collision_bounds'])
    for name in cfg['asset']['foot_names']:
        body = model.body(name).id
        for geom in np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype != 0)):
            mesh = model.geom_dataid[geom]
            vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh] + model.mesh_vertnum[mesh]]
            world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
            local = (world - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
            np.testing.assert_allclose([local.min(0), local.max(0)], bounds, atol=2e-6)
    assert cfg['rewards']['ball_radius'] == pytest.approx(model.geom_size[model.geom('soccer_ball').id, 0])


def _place_test_ball(env, x):
    env.feet_pos[:] = torch.tensor([0., 0., .04])
    env.feet_quat[:] = torch.tensor([0., 0., 0., 1.])
    env.feet_yaw[:] = 0
    env.feet_vel[:] = torch.tensor([.5, 0.])
    env.ball_pos[:] = torch.tensor([x, 0., .1])
    env.last_ball_pos[:] = env.ball_pos[:, :2]
    env.base_pos[:, :2] = 0
    env.last_base_pos[:] = 0
    env.projected_gravity[:] = torch.tensor([0., 0., -1.])
    env.extras['fall'] = torch.zeros(env.num_envs)


def test_near_ball_idle_cannot_collect_amp_or_reset_timer_by_wiggling(cfg):
    env = Q1(cfg)
    env.reset()
    _place_test_ball(env, .30)
    for _ in range(200):
        # Foot movement without ball progress must not restart preparation time.
        env.feet_pos[:, :, 0] = .01 if _ % 2 else -.01
        env._update_soccer_feedback()
    assert (env.near_ball_stall == 1).all()
    assert (env.amp_reward_multiplier == 0).all()
    assert env._reward_kick_ball().eq(0).all()
    # Making real forward ball progress permits preparation/style reward again.
    env.last_ball_pos[:, 0] -= .01
    env._update_soccer_feedback()
    assert env.near_ball_idle_s.eq(0).all()
    assert env.amp_reward_multiplier.eq(1).all()
    env._reset_idx(torch.tensor([0]))
    assert env.near_ball_idle_s[0] == env.far_ball_idle_s[0] == 0


def test_ball_at_one_meter_is_not_in_kicking_range_or_idle_exemption(cfg):
    env = Q1(cfg)
    env.reset()
    _place_test_ball(env, .8)
    for _ in range(200):
        env._update_soccer_feedback()
    assert (env.foot_ball_gap > cfg['rewards']['near_ball_gap']).all()
    assert env.near_ball_idle_s.eq(0).all()
    assert env._reward_pos_still().eq(1).all()


def test_kick_reward_requires_nearby_goal_directed_ball_motion(cfg):
    env = Q1(cfg)
    env.reset()
    _place_test_ball(env, .22)
    env.last_ball_pos[:, 0] -= .01
    env._update_soccer_feedback()
    assert (env._reward_kick_ball() > 0).all()
    assert cfg['rewards']['scales']['kick_ball'] > 0
    env.last_ball_pos[:, 0] += .02
    env._update_soccer_feedback()
    assert env._reward_kick_ball().eq(0).all()
    env.last_ball_pos[:, 0] -= .02
    env.feet_pos[:, :, 0] = -2
    env._update_soccer_feedback()
    assert env._reward_kick_ball().eq(0).all()
    env.feet_pos[:, :, 0] = 0
    env._update_soccer_feedback()
    env.extras['fall'][:] = 1
    assert env._reward_kick_ball().eq(0).all()


def test_far_ball_rocking_does_not_restart_idle_window_or_keep_amp(cfg):
    env = Q1(cfg)
    env.reset()
    _place_test_ball(env, .8)
    for i in range(250):
        env.base_pos[:, 0] = .01 if i % 2 else -.01
        env._update_soccer_feedback()
    assert env.far_ball_stall.eq(1).all()
    assert env.amp_reward_multiplier.eq(0).all()
    assert env.extras['soccer_episode']['far_ball_stall_s'].gt(0).all()
    env._reset_idx(torch.tensor([0]))
    assert env.approach_window_steps[0] == env.far_ball_stall[0] == 0
    assert env.amp_reward_multiplier[0] == 1


def test_sustained_approach_keeps_amp_but_single_step_cannot_clear_stall(cfg):
    env = Q1(cfg)
    env.reset()
    _place_test_ball(env, 5.)
    for i in range(250):
        env.base_pos[:, 0] = i * .002  # net 0.1 m/s, comfortably above threshold
        env._update_soccer_feedback()
    assert env.far_ball_stall.eq(0).all()
    assert env.amp_reward_multiplier.eq(1).all()
    for _ in range(250):
        env._update_soccer_feedback()
    assert env.far_ball_stall.eq(1).all()
    env.base_pos[:, 0] += .001  # one forward twitch is not a completed window
    env._update_soccer_feedback()
    assert env.far_ball_stall.eq(1).all()








def test_near_ball_boundary_jitter_does_not_restart_preparation(cfg):
    env = Q1(cfg)
    env.reset()
    _place_test_ball(env, .30)
    env._update_soccer_feedback()
    assert env.near_ball_active.all()
    for i in range(200):
        # Stay on either side of the entry threshold, inside the exit threshold.
        env.ball_pos[:, 0] = .33 if i % 2 else .35
        env.last_ball_pos[:] = env.ball_pos[:, :2]
        env._update_soccer_feedback()
    assert env.near_ball_active.all()
    assert env.near_ball_stall.eq(1).all()
    assert env.amp_reward_multiplier.eq(0).all()


def test_forward_and_side_kick_weights_and_no_double_reward(cfg):
    env = Q1(cfg)
    env.reset()
    _place_test_ball(env, .22)
    env.last_ball_pos[:, 0] -= .01
    env._update_soccer_feedback()
    forward = env._reward_kick_ball().clone()
    assert forward.gt(0).all() and env._reward_side_kick_ball().eq(0).all()
    env.feet_vel[:] = torch.tensor([0., .5])
    side = env._reward_side_kick_ball()
    torch.testing.assert_close(side, forward)
    assert env._reward_kick_ball().eq(0).all()
    assert cfg['rewards']['scales']['kick_ball'] == 2
    assert cfg['rewards']['scales']['side_kick_ball'] == 8
    env.feet_vel[:] = torch.tensor([.5, .5])
    torch.testing.assert_close(env._reward_kick_ball() + env._reward_side_kick_ball(), forward)
    env.last_ball_pos[:] = env.ball_pos[:, :2]
    env._update_soccer_feedback()
    assert env._reward_kick_ball().eq(0).all()
    assert env._reward_side_kick_ball().eq(0).all()


def test_ball_spawn_is_uniform_in_center_disk_and_excludes_robot(cfg):
    env = Q1(cfg)
    ids = torch.arange(env.num_envs)
    env.qpos[:, env.root_addr:env.root_addr + 2] = torch.tensor([6., 0.])
    samples = []
    for _ in range(1000):
        env._reset_ball_states(ids)
        samples.append(env.qpos[:, env.ball_addr:env.ball_addr + 2].clone())
    xy = torch.cat(samples)
    radius = cfg['init_state']['ball_spawn_radius']
    assert radius == 3.5
    assert (xy.norm(dim=-1) <= radius + 1e-6).all()
    assert abs(xy.square().sum(-1).mean().item() / radius**2 - .5) < .03
    assert xy.mean(0).abs().max() < .12
    env.qpos[:, env.root_addr:env.root_addr + 2] = 0
    for _ in range(50):
        env._reset_ball_states(ids)
        xy = env.qpos[:, env.ball_addr:env.ball_addr + 2]
        assert (xy.norm(dim=-1) >= .65).all()
        assert (xy.norm(dim=-1) <= radius + 1e-6).all()




def test_near_start_curriculum_is_safe_and_preserves_ball_disk(cfg):
    cfg['curriculum']['near_start']['enabled'] = True
    cfg['curriculum']['near_start']['initial_fraction'] = 1.
    cfg['env']['num_envs'] = 128
    env = Q1(cfg)
    env.reset()
    assert env.near_start_episode.all()
    gap = env._foot_ball_gap()
    assert gap.min() > .075
    assert gap.max() < .28
    assert (env.ball_pos[:, :2].norm(dim=-1) <= 3.5 + 1e-5).all()
    torch.testing.assert_close(env.dof_pos, env.default_dof_pos)
    assert env.qvel.eq(0).all()
    # Native collision engine must not report a robot-ball overlap at spawn.
    data = mujoco.MjData(env.model)
    ball_geom = env.model.geom('soccer_ball').id
    floor = env.model.geom('floor').id
    for row in env.qpos:
        data.qpos[:] = row.numpy()
        mujoco.mj_forward(env.model, data)
        assert not any(ball_geom in c.geom and floor not in c.geom and c.dist < 0 for c in data.contact)


def test_near_start_mixture_advances_only_on_completed_near_trials(cfg):
    # Legacy adaptive mode remains available when the fixed mixture is disabled.
    cfg['curriculum']['near_start']['final_fraction'] = .15
    cfg['curriculum']['near_start']['enabled'] = True
    cfg['env']['num_envs'] = 512
    env = Q1(cfg)
    env.reset()
    assert .4 < env.near_start_episode.float().mean() < .6
    env.balance_release_iteration = 0
    env.survival_rate_5s = .9
    env.near_start_success_rate = .9
    env.set_training_iteration(7000)
    assert env.near_start_progress == 0  # no completed trials yet
    env.near_start_trials = 200
    env.set_training_iteration(7001)
    assert env.near_start_progress == pytest.approx(.1)
    assert env.near_start_fraction < .5
    state = env.near_start_state()
    env.near_start_progress = 0
    env.restore_near_start_state(state)
    assert env.near_start_progress == pytest.approx(.1)
    env.near_start_success_rate = .1
    env.near_start_trials += 128
    env.set_training_iteration(7101)
    assert env.near_start_progress == 0






@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for Warp')
def test_near_start_gpu_reset_geometry(cfg):
    cfg['curriculum']['near_start'].update(enabled=True, initial_fraction=1.)
    cfg['basic']['device'] = 'cuda:0'
    cfg['env']['num_envs'] = 32
    env = Q1(cfg)
    env.reset()
    assert env.near_start_episode.all()
    assert env._foot_ball_gap().min() > .075
    assert env._foot_ball_gap().max() < .28
    for _ in range(20):
        obs, reward, _, _ = env.step(torch.zeros(32, 22, device='cuda:0'))
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    env.batch.check()




def test_survival_curriculum_does_not_overweight_repeatedly_failing_worlds(cfg):
    env = Q1(cfg); env.reset()
    env.episode_length_buf[:] = torch.tensor([250,250,10,10])
    env.reset_buf[:] = torch.tensor([False,False,True,True])
    env.extras['fall'] = torch.tensor([0.,0.,1.,1.])
    env.extras['success'] = torch.zeros(4)
    env._record_survival_trials()
    assert env.survival_rate_5s == pytest.approx(.5)
    for _ in range(20):
        env._survival_checked[2:] = False
        env._record_survival_trials()
    assert env.survival_rate_5s == pytest.approx(.5)
