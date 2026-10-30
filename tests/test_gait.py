import os
from pathlib import Path
import torch
import pytest
from q1_4v4.learning import _input_map,_expand_input,make_args,make_team,load_team
from q1_4v4.course_config import COURSES,DURATIONS,SUCCESS


def test_course_order_and_thresholds():
    assert len(COURSES)==28 and COURSES[2]=='weight_shift'
    assert DURATIONS['weight_shift']==500 and DURATIONS['start_stop']==900
    assert SUCCESS['balance']==.95 and SUCCESS['slow_walk']==.8


def test_migration_column_and_optimizer_layout():
    index=_input_map(121,128)
    old=torch.arange(121.).expand(5,-1)
    result=_expand_input(old,(5,128),index)
    assert torch.equal(result[:,index],old)
    assert not result[:,115:122].any()
    idx=_input_map(972,1028,central=True)
    assert idx[-4:]==[1024,1025,1026,1027]
    assert idx[121]==128 and idx[115]==122
    with pytest.raises(ValueError):_input_map(100,128)


@pytest.fixture(scope='module')
def env():
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.course_env import CourseWarpEnv
    return CourseWarpEnv(count=2,stage='weight_shift')


def test_ground_contacts_are_separate_from_ball_and_static_stance_cannot_pass(env):
    env.configure('step_in_place');env.reset()
    zero=torch.zeros((2,8,22),device=env.device)
    for _ in range(100):
        obs,_,_,_,_,info=env.step(zero)
    assert env.gait_contact.bool().any()
    assert not env.foot_hits.any()
    assert not env.gait_landings.any()  # neutral PD is not a walking gait
    assert not info['success'].any()
    assert env.gait_metrics['slip_mps']<.05
    env.check()


def test_independent_gait_reset_and_command_observation(env):
    env.configure('slow_walk');env.reset()
    for _ in range(4):env.step(torch.zeros((2,8,22),device=env.device))
    before=env.agent_steps.clone();period=env.gait_period.clone();q=env.qpos.clone()
    env.reset_agents(torch.tensor([[0,1]],device=env.device))
    assert env.agent_steps[0,1]==0 and env.agent_steps[0,0]==before[0,0]
    assert torch.equal(q[1],env.qpos[1]) and torch.equal(period[1],env.gait_period[1])
    obs=env.observe()
    assert obs.shape==(2,8,128)
    assert torch.equal(obs[:,:,-6:-2],env.roles.expand(2,-1,-1))
    assert ((obs[:,:,115]>=.05)&(obs[:,:,115]<=.15)).all()
    torch.testing.assert_close(obs[:,:,118].square()+obs[:,:,119].square(),torch.ones((2,8),device=env.device))


def test_load_existing_weights_and_optimizer_moments(env):
    path=Path('logs/gait_migration/source_last.pt')
    if not path.exists():pytest.skip('Local training checkpoint not distributed')
    old=torch.load(path,map_location=env.device,weights_only=False)
    args=make_args(2,8,1)
    p,t,_=make_team(args,env,env.device,gpu_rollout=True)
    load_team(old['teams'][0],p,t)
    key='base.base.mlp.fc1.0.weight'
    torch.testing.assert_close(p.actor.state_dict()[key][:,_input_map(121,128)],old['teams'][0]['actor'][key])
    for param,state in p.actor_optimizer.state.items():
        if 'exp_avg' in state:assert state['exp_avg'].shape==param.shape
    assert len(t.value_normalizer.state_dict())==3
