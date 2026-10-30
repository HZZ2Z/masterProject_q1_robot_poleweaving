import os
import pytest
import torch
from q1_4v4.course_config import Curriculum, COURSES, LOW_LEVEL, command


def test_disjoint_consecutive_gates_and_legacy_migration():
    c=Curriculum()
    c.load_state_dict({'index':3,'stage_updates':200,'history':[]})
    assert c.stage=='balance' and c.stage_updates==0
    for iteration in range(64):
        for j in range(32): c.record(dict(stage='balance',stable=True,success=True,duration=6,team=j%2))
        advanced=c.advance()
        if iteration<63: assert not advanced
    assert advanced and c.stage=='balance_push'
    d=Curriculum();d.load_state_dict(c.state_dict());assert d.state_dict()==c.state_dict()
    for _ in range(100):
        for _ in range(128): d.record(dict(stage=d.stage,stable=True,success=False,duration=6))
        assert not d.advance()


def test_task_commands_preserve_identity():
    assert len({tuple(command(s)) for s in COURSES})==len(COURSES)
    assert command('balance')==[1,0,0,0,0,0]
    assert all(len(command(s))==6 for s in COURSES)


def test_team_windows_survive_large_ordered_bursts_and_restart():
    c=Curriculum('step_in_place')
    for team in (0,1):
        for _ in range(10000):
            c.record(dict(stage=c.stage,team=team,stable=True,success=True,duration=12))
    m=c.metrics()
    assert m['red_samples']==m['blue_samples']==1024
    assert m['red_stability']==m['blue_stability']==1
    assert m['red_gate_samples']==m['blue_gate_samples']==1024
    restored=Curriculum();restored.load_state_dict(c.state_dict())
    assert restored.state_dict()==c.state_dict()
    restored.stage_updates=31
    assert not restored.advance() and restored.passed_windows==1
    assert len(restored.history)==0
    # A new window needs fresh observations from BOTH teams.
    for _ in range(10000):
        restored.record(dict(stage=restored.stage,team=1,stable=True,success=True))
    restored.stage_updates=47
    assert not restored.advance() and restored.passed_windows==0
    assert restored.last_reason=='samples'


def test_missing_team_is_unknown_and_bad_team_cannot_hide_in_average():
    c=Curriculum()
    for _ in range(256):c.record(dict(stage=c.stage,team=0,stable=False,success=True))
    assert c.metrics()['blue_stability'] is None and c.metrics()['blue_samples']==0
    for _ in range(10000):c.record(dict(stage=c.stage,team=1,stable=True,success=True))
    c.stage_updates=31
    assert not c.advance() and c.last_reason=='stability'
    stage=c.stage
    c.recertify('reward_change_recertification')
    assert c.stage==stage and c.stage_updates==0 and len(c.monitor)==0


@pytest.fixture(scope='module')
def env():
    if os.environ.get('Q1_TEST_WARP')!='1': pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.course_env import CourseWarpEnv
    return CourseWarpEnv(count=4)


@pytest.mark.parametrize('course',COURSES)
def test_each_course_spawn_and_physics(env,course):
    env.configure(course);env.episode_steps=300;env.reset()
    for _ in range(3):
        obs,state,reward,done,trunc,info=env.step(torch.zeros((4,8,22),device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(state).all() and torch.isfinite(reward).all()
    env.check()
    assert obs.shape==(4,8,128) and state.shape==(4,8,1028)
    assert done.shape==((4,8) if course in LOW_LEVEL else (4,))
    assert not info['success'].any()


def test_independent_robot_reset(env):
    env.configure('move');env.reset()
    env.step(torch.zeros((4,8,22),device=env.device))
    q,v,steps=env.qpos.clone(),env.qvel.clone(),env.agent_steps.clone()
    env.reset_agents(torch.tensor([[0,0]],device=env.device))
    # The seven other articulated bodies and shared ball are untouched.
    allowed=torch.ones(env.model.nq,device=env.device,dtype=torch.bool)
    allowed[env.qidx[0]]=False;allowed[env.rootq[0]:env.rootq[0]+7]=False
    assert torch.equal(q[0,allowed],env.qpos[0,allowed])
    assert torch.equal(q[1:],env.qpos[1:]) and torch.equal(v[1:],env.qvel[1:])
    assert env.agent_steps[0,0]==0 and torch.equal(steps[0,1:],env.agent_steps[0,1:])


@pytest.mark.parametrize('side',['left','right'])
def test_both_foot_contacts_and_body_exclusion(env,side):
    env.configure('match');env.reset()
    f=0 if side=='left' else 1
    foot=env.feet[2,f]
    env.qpos[0,env.ballq:env.ballq+3]=env.xpos[0,foot]+env.tensor([.12,0,.05])
    # A ball at torso height may touch the body, but must never count as a foot contact.
    torso=env.model.body('red2_torso_link').id
    env.qpos[1,env.ballq:env.ballq+3]=env.xpos[1,torso]+env.tensor([.05,0,.05])
    env.forward()
    env.step(torch.zeros((4,8,22),device=env.device));env.check()
    assert env.foot_hits[0,2,f]==1
    assert not env.foot_hits[1].any()
    assert env.forward_kick[1].sum()==0 and env.side_kick[1].sum()==0
