import os
from pathlib import Path
import torch
import pytest
from q1_4v4.learning import _input_map,_expand_input,make_args,make_team,load_team
from q1_4v4.course_config import COURSES,DURATIONS,SUCCESS
from q1_4v4.gait import phase_lift_credit,stepping_reward,marching_shaping,marching_success


def test_marching_requires_knee_and_foot_lift_and_penalizes_drift():
    feet=torch.tensor([[[0.,.0877,-.377],[0.,-.0877,-.377]]])
    neutral=torch.tensor([[.02,.09085,-.188],[.02,-.09085,-.188]])
    knees=neutral[None].clone();knees[0,1,0]+=.04;knees[0,1,2]+=.02
    expected=torch.tensor([[True,False]])
    height=torch.full((1,2),.04);clearance=torch.tensor([[0.,.04]])
    zero=torch.zeros(1)
    def score(k=knees,f=feet,cl=clearance,contact=expected,drift=zero,yaw=zero,v=zero):
        return marching_shaping(f,k,neutral,cl,height,contact,expected,drift,yaw,v,zero)
    good,cost,_=score()
    assert good.item()>.99 and cost.item()==0
    assert score(k=neutral[None])[0].item()==0
    crouch=neutral[None].clone();crouch[:,:,0]+=.04;crouch[:,:,2]+=.02
    assert score(k=crouch)[0].item()==0  # common bilateral knee rise cannot earn credit
    assert score(cl=torch.zeros_like(clearance))[0].item()==0
    assert score(contact=~expected)[0].item()==0
    # Raising the pelvis leaves pelvis-relative knee positions unchanged.
    assert score(k=neutral[None])[0].item()==0
    assert score(drift=torch.tensor([.1]))[1].item()>.15
    assert score(yaw=torch.tensor([.3]))[1].item()>.1
    assert score(v=torch.tensor([.1]))[1].item()>.05
    wide=feet.clone();wide[:,:,1]*=2
    assert score(f=wide)[1].item()>.2
    mirrored_k=knees.flip(1).clone();mirrored_k[:,:,1]*=-1
    mirrored_f=feet.flip(1).clone();mirrored_f[:,:,1]*=-1
    mirrored=marching_shaping(mirrored_f,mirrored_k,neutral,clearance.flip(1),height,
        expected.flip(1),expected.flip(1),zero,zero,zero,zero)
    torch.testing.assert_close(good,mirrored[0])


def test_marching_gate_rejects_walking_turning_and_one_leg_shuffling():
    quality=torch.ones(6,dtype=torch.bool)
    steps=torch.full((6,2),4.)
    alternations=torch.full((6,),7.)
    drift=torch.zeros(6);heading=torch.zeros(6);knee=torch.full((6,),.4)
    # A prior excursion must fail even if the robot has returned to the start.
    drift[1]=.11;heading[2]=.3;knee[3]=.1;steps[4,1]=0;alternations[5]=0
    assert marching_success(quality,steps,alternations,drift,heading,knee).tolist()==[True,False,False,False,False,False]


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


def test_correct_phase_lifting_beats_static_stance_symmetrically():
    expected=torch.tensor([[True,False]])
    target=torch.full((1,2),.02)
    healthy=torch.tensor([True]);zero=torch.zeros(1);none=torch.zeros((1,2),dtype=torch.bool)
    def score(clearance,contact,expect=expected,landing=none,alternating=none):
        credit=phase_lift_credit(clearance,contact,expect,target)
        return stepping_reward(.02,credit,contact,expect,zero,zero,zero,zero,healthy,landing,alternating)
    ground=torch.zeros((1,2));both=torch.ones((1,2),dtype=torch.bool)
    static=score(ground,both)
    # Add the existing upright survival component: standing must not favor suicide.
    assert (static+.02>0).all()
    correct=torch.tensor([[0.,.02]])
    lifted=score(correct,expected)
    assert (lifted>static+.05).all()
    assert score(torch.tensor([[0.,.005]]),expected)>static
    torch.testing.assert_close(lifted,score(correct.flip(-1),expected.flip(-1),expected.flip(-1)))
    assert score(torch.full((1,2),.02),none)<=static
    assert score(correct.flip(-1),expected.flip(-1))<lifted
    # Hopping cannot receive two alternating-landing event bonuses.
    torch.testing.assert_close(score(ground,both,landing=both,alternating=both),static)
    landing=torch.tensor([[True,False]])
    torch.testing.assert_close(score(ground,both,landing=landing,alternating=landing)-static,torch.tensor([.2]))


@pytest.fixture(scope='module')
def env():
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.course_env import CourseWarpEnv
    return CourseWarpEnv(count=2,stage='weight_shift')


def test_ground_contacts_are_separate_from_ball_and_static_stance_cannot_pass(env):
    env.configure('step_in_place');env.reset()
    torch.testing.assert_close(env.targets,env.start_positions)
    assert env.observe()[:,:,106:109].abs().max()<1e-5
    assert env.gait_height.min()>=.02 and env.gait_height.max()<=.03
    zero=torch.zeros((2,8,22),device=env.device)
    for _ in range(100):
        obs,_,_,_,_,info=env.step(zero)
    assert env.gait_contact.bool().any()
    assert not env.foot_hits.any()
    assert not env.gait_landings.any()  # neutral PD is not a walking gait
    assert not info['success'].any()
    assert env.gait_metrics['slip_mps']<.05
    env.check()
    env.reset_agents(torch.tensor([[0,1]],device=env.device))
    torch.testing.assert_close(env.targets[0,1],env.start_positions[0,1])
    assert env.observe()[0,1,106:109].abs().max()<1e-5


def test_independent_gait_reset_and_command_observation(env):
    env.configure('slow_walk');env.reset()
    for _ in range(4):env.step(torch.zeros((2,8,22),device=env.device))
    before=env.agent_steps.clone();period=env.gait_period.clone();q=env.qpos.clone()
    env.reset_agents(torch.tensor([[0,1]],device=env.device))
    assert env.agent_steps[0,1]==0 and env.agent_steps[0,0]==before[0,0]
    assert torch.equal(q[1],env.qpos[1]) and torch.equal(period[1],env.gait_period[1])
    assert env.gait_drift_max[0,1]==0 and env.gait_heading_max[0,1]==0 and env.gait_knee_credit[0,1]==0
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


def test_marching_observation_resume_preserves_input_semantics():
    from q1_4v4.course_config import marching_observation_mode
    assert marching_observation_mode()=='origin'
    assert marching_observation_mode(checkpoint={'reward_version':4})=='legacy'
    assert marching_observation_mode(checkpoint={'marching_target_observation':'origin'})=='origin'
    with pytest.raises(ValueError,match='fresh run'):
        marching_observation_mode('origin',{'reward_version':4})


def test_legacy_marching_targets_are_kept_only_for_compatible_resumption(env):
    env.marching_target_observation='legacy';env.configure('step_in_place');env.reset()
    offset=env.targets-env.start_positions
    assert (offset[:,:,0]*env.sign>=.6).all() and (offset[:,:,0]*env.sign<=1.).all()
    assert not env.gait_command.any()  # actual marching command remains stationary
    env.reset_agents(torch.tensor([[0,0]],device=env.device))
    assert (env.targets[0,0,0]-env.start_positions[0,0,0]).item()==pytest.approx(.8,abs=1e-5)
    env.marching_target_observation='origin';env.reset()
    torch.testing.assert_close(env.targets,env.start_positions)


def test_cost_keeps_improving_when_wide_stance_moves_toward_target():
    knees=torch.tensor([[[.02,.09085,-.188],[.02,-.09085,-.188]]])
    expected=torch.tensor([[True,False]]);zero=torch.zeros(1)
    def cost(width,drift):
        feet=torch.tensor([[[0.,width/2,-.377],[0.,-width/2,-.377]]])
        return marching_shaping(feet,knees,knees[0],torch.zeros((1,2)),torch.full((1,2),.03),expected,expected,
                                torch.tensor([drift]),zero,zero,zero)[1]
    assert cost(.4,.15)>cost(.3,.15)>cost(.2,.15)
    assert cost(.1754,.2)>cost(.1754,.12)>cost(.1754,.05)
