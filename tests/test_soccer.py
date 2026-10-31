import numpy as np
import mujoco
import pytest
from q1_4v4.env import SoccerEnv
from q1_4v4.curriculum import Curriculum, STAGES


@pytest.fixture(scope='module')
def env():
    return SoccerEnv()


def test_scene_and_collision_masks(env):
    m = env.model
    assert m.nu == 176
    assert m.nq == 8*29+7
    assert m.nv == 8*28+6
    for i, team in enumerate(('red', 'blue')):
        for n in range(4):
            cap = m.geom(f'{team}{n}_cap')
            assert cap.contype == cap.conaffinity == 0
            assert cap.rgba[i*2] > .9
    # All inter-robot collision pairs enabled, same-robot mesh pairs filtered.
    for a in range(8):
        ga = next(g for g, owner in env.geom_owner.items() if owner == a and m.geom_contype[g])
        assert not (m.geom_contype[ga] & m.geom_conaffinity[ga])
        for b in range(8):
            gb = next(g for g, owner in env.geom_owner.items() if owner == b and m.geom_contype[g])
            assert bool(m.geom_contype[ga] & m.geom_conaffinity[gb]) == (a != b)


def test_neutral_six_seconds(env):
    env.stage = 'balance'
    env.reset()
    for _ in range(300):
        o, s, r, done, _, info = env.step(np.zeros((8, 22)))
        assert not done
        assert np.isfinite(o).all() and np.isfinite(s).all() and np.isfinite(r).all()
    assert info['stable'] and info['success']
    assert (env.data.xpos[env.bodies, 2] > .3).all()


def test_neutral_stance_geometry(env):
    env.stage='balance';env.reset()
    for prefix in ('red0_','blue0_'):
        root=env.data.body(prefix+'pelvis')
        R=root.xmat.reshape(3,3)
        feet=[env.data.body(prefix+s+'_ankle_roll_link') for s in ('left','right')]
        separation=(feet[0].xpos-feet[1].xpos)@R
        assert separation[1]==pytest.approx(.1753964,abs=1e-6)
        for f in feet:
            local=R.T@f.xmat.reshape(3,3)
            assert local[2,2]>np.cos(np.deg2rad(1))
            assert abs(np.arctan2(local[1,0],local[0,0]))<np.deg2rad(5)


@pytest.mark.parametrize('stage', STAGES)
def test_stage_spawn_and_step(stage, env):
    env.stage = stage
    o, s = env.reset()
    assert o.shape == (8, 121)
    assert s.shape == (8, 972)
    assert np.array_equal(o[:, -6:-2], env.roles)
    for _ in range(10):
        o, s, r, done, truncated, info = env.step(np.zeros((8, 22)))
        assert not done
    assert not info['success']  # Ten standing steps cannot pass a five-second gate.


def test_goals():
    goal = SoccerEnv.goal_crossing
    assert goal(np.array([7., 0, .1]), np.array([7.2, 0, .1])) == 1
    assert goal(np.array([-7., 0, .1]), np.array([-7.2, 0, .1])) == -1
    assert goal(np.array([7., 2, .1]), np.array([7.2, 2, .1])) == 0
    assert goal(np.array([7.2, 2, .1]), np.array([7.3, 0, .1])) == 0
    assert goal(np.array([7., 0, 1.2]), np.array([7.2, 0, 1.2])) == 0


def test_fall_and_timeout(env):
    env.stage = 'match'
    env.reset()
    env.data.qpos[env.rootq[0]+2] = .1
    mujoco.mj_forward(env.model, env.data)
    _, _, r, done, truncated, info = env.step(np.zeros((8, 22)))
    assert done and not truncated and info['fallen'][0] and not info['success']
    assert r[0, 0] < -4
    env.reset()
    env.episode_steps = 1
    _, _, _, done, truncated, _ = env.step(np.zeros((8, 22)))
    assert truncated and not done
    env.episode_steps = 1500


def test_real_ball_contact(env):
    env.stage = 'match'
    env.reset()
    # Place ball overlapping an actual foot mesh, then check MuJoCo contact reporting.
    foot = env.model.body('red2_left_ankle_roll_link').id
    env.data.qpos[env.ballq:env.ballq+3] = env.data.xpos[foot] + [.12, 0, .05]
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, _, info = env.step(np.zeros((8, 22)))
    assert 2 in info['touches']


def test_curriculum_gate_and_restore():
    c = Curriculum()
    for _ in range(40):
        c.record(dict(stage='balance', stable=False, success=False))
        assert not c.advance()
    for _ in range(64):
        c.record(dict(stage='balance', stable=True, success=True))
    assert c.advance() and c.stage == 'move'
    assert not c.history
    d = Curriculum()
    d.load_state_dict(c.state_dict())
    assert d.state_dict() == c.state_dict()
    for _ in range(64):
        d.record(dict(stage='move', stable=True, success=False))
        assert not d.advance()


def test_invalid_actions(env):
    with pytest.raises(ValueError):
        env.step(np.full((8, 22), np.nan))


@pytest.mark.parametrize('sign', [1, -1])
def test_physical_goal_reward_and_reset(env, sign):
    env.stage = 'match'
    env.reset()
    env.data.qpos[env.ballq:env.ballq+3] = [sign*7.05, 0, .11]
    env.data.qvel[env.ballv:env.ballv+3] = [sign*10., 0, 0]
    mujoco.mj_forward(env.model, env.data)
    _, _, reward, done, truncated, info = env.step(np.zeros((8, 22)))
    assert done and not truncated and info['goal'] == sign
    assert (reward[:4, 0]*sign > 19).all()
    assert (reward[4:, 0]*sign < -19).all()
    env.reset()
    assert np.all(env.data.qvel == 0) and env.steps == 0
    assert abs(env.ball[0]) < .3


def test_outside_mouth_is_not_goal(env):
    env.stage = 'match'
    env.reset()
    env.data.qpos[env.ballq:env.ballq+3] = [7.05, 2, .11]
    env.data.qvel[env.ballv] = 10
    mujoco.mj_forward(env.model, env.data)
    _, _, _, done, _, info = env.step(np.zeros((8, 22)))
    assert done and info['out'] and not info['goal']
