"""CPU checks for soccer locomotion, legal segments, sampling and resume migration."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from q1_4v4.match_task import MatchTask,MatchControl
from q1_4v4.match_locomotion import MatchLocomotion
from q1_4v4.learning import make_args,make_team,save_team,load_team,_input_map,configure_temporal
from q1_4v4.gpu_learning import GpuRollout


class Environment:
    def __init__(self,**values):self.__dict__.update(values)


def features():
    return dict(moving=torch.ones((1,8),dtype=torch.bool),healthy=torch.ones((1,8),dtype=torch.bool),
        speed=torch.full((1,8),.18),yaw_command=torch.zeros((1,8)),linear_error=torch.full((1,8),.18),
        angular_error=torch.zeros((1,8)),forward_speed=torch.zeros((1,8)),lift_credit=torch.zeros((1,8)),
        landing=torch.zeros((1,8,2),dtype=torch.bool),expected=torch.tensor([True,False]).expand(1,8,2).clone(),
        alternating=torch.zeros((1,8,2),dtype=torch.bool),slip=torch.zeros((1,8,2)),
        flight=torch.zeros((1,8),dtype=torch.bool),crossed=torch.zeros((1,8),dtype=torch.bool))


def test_movement_reward_has_no_standing_tracking_bonus_and_respects_modes():
    f=features();r=MatchLocomotion.rewards(f,.02)
    assert all((v==0).all() for v in r.values())
    f['linear_error'].zero_();f['forward_speed'].fill_(.18);f['lift_credit'].fill_(.8)
    r=MatchLocomotion.rewards(f,.02)
    assert (r['gait_velocity']>0).all() and (r['gait_swing']>0).all()
    f['moving'].zero_();r=MatchLocomotion.rewards(f,.02)
    assert all((v==0).all() for v in r.values())
    f['moving'].fill_(True);f['healthy'].zero_();r=MatchLocomotion.rewards(f,.02)
    assert (r['gait_velocity']==0).all() and (r['gait_swing']==0).all()


def test_landing_requires_motion_single_correct_foot_and_low_slip():
    f=features();f['forward_speed'].fill_(.18);f['landing'][:,:,0]=True;f['alternating'][:,:,0]=True
    assert torch.allclose(MatchLocomotion.rewards(f,.02)['gait_landing'],torch.full((1,8),.12))
    f['slip'].fill_(.2);assert not MatchLocomotion.rewards(f,.02)['gait_landing'].any()
    f['slip'].zero_();f['landing'].fill_(True);assert not MatchLocomotion.rewards(f,.02)['gait_landing'].any()
    f['landing'][:,:,1]=False;f['forward_speed'].zero_();assert not MatchLocomotion.rewards(f,.02)['gait_landing'].any()


def test_slip_and_distance_costs_are_bounded_but_distinguish_severe_errors():
    slips=MatchTask.slip_cost(torch.tensor([.02,.05,.10,.20,10.]))
    assert (slips[1:]>slips[:-1]).all() and (slips<=4).all()
    distances=MatchTask.distance_cost(torch.tensor([.2,1.,3.25,4.,6.,100.]))
    assert distances[0]==0 and (distances[1:]>distances[:-1]).all() and (distances<=3).all()


def test_sampling_covers_near_middle_far_and_standard_with_one_ball():
    n=4096;generator=torch.Generator().manual_seed(42)
    e=Environment(count=n,device=torch.device('cpu'),generator=generator,match_layout='mixed',
        sign=torch.tensor([1.]*4+[-1.]*4),rootq=torch.arange(8)*7,qpos=torch.zeros(n,56),ball=torch.zeros(n,3))
    e.tensor=lambda value:torch.tensor(value)
    e.uniform=lambda shape,lo=0.,hi=1.:torch.rand(shape,generator=generator)*(hi-lo)+lo
    task=MatchTask(e);ids=torch.arange(n);task.spawn_curriculum(ids)
    assert [(task.spawn_bucket==i).sum().item() for i in range(4)]==[1640,1228,818,410]
    for bucket,(lo,hi) in enumerate(task.control.distances[:3]):
        worlds=ids[task.spawn_bucket==bucket];team=task.kickoff_team[worlds]
        robot=task.kickoff_robot[worlds]
        assert (robot//4==team).all() and ((robot%4==2)|(robot%4==3)).all()
        distance=e.qpos[worlds,e.rootq[robot]].abs()
        assert (distance>=lo).all() and (distance<=hi).all()
        assert abs((team==0).float().mean().item()-.5)<.02
    assert not e.ball.any()


def test_mid_distance_evidence_and_individual_safety_gate_expansion():
    good={k:16. for k in MatchControl.counters}
    good.update(quality_red=0.,quality_blue=0.)  # Whole-team failures remain an evaluation measure.
    c=MatchControl()
    for _ in range(48):c.update([.99,.99],[1.,1.],good)
    assert c.level==1 and c.amplitude==.475
    good['mid_touch_blue']=0.;c=MatchControl()
    for _ in range(48):c.update([.99,.99],[1.,1.],good)
    assert c.level==0 and c.reason=='mid_touch'
    good['mid_touch_blue']=16.;good['training_quality_blue']=0.;c=MatchControl()
    for _ in range(48):c.update([.99,.99],[1.,1.],good)
    assert c.level==0 and c.reason=='posture'


@pytest.fixture
def physical_state():
    e=Environment(count=1,device=torch.device('cpu'),dt=.02,course='match',episode_steps=1500)
    e.sign=torch.tensor([1.]*4+[-1.]*4);e.bodies=torch.arange(8);e.feet=torch.arange(8,24).reshape(8,2)
    red=torch.tensor([[-5.8,0,.417],[-3.8,0,.417],[-1.,0,.417],[-4.,2.,.417]])
    e.xpos=torch.zeros((1,24,3));e.xpos[:,:8]=torch.cat((red,red*torch.tensor([-1.,-1.,1.])))[None]
    e.xmat=torch.eye(3).reshape(1,1,9).repeat(1,24,1);e.xmat[:,4:8]=torch.diag(torch.tensor([-1.,-1.,1.])).reshape(1,1,9)
    for a in range(8):
        for f,side in enumerate((1.,-1.)):
            e.xpos[0,e.feet[a,f]]=e.xpos[0,a]+torch.tensor([0.,float(e.sign[a])*side*.0877,-.417])
    e.qvel=torch.zeros((1,176));e.vidx=torch.arange(176).reshape(8,22)
    e.ball=torch.tensor([[0.,0,.101]]);e.start_ball=e.ball.clone();e.rows=torch.arange(1)
    e.active=torch.ones((1,8),dtype=torch.bool);e.steps=torch.tensor([250]);e.targets=torch.zeros((1,8,3))
    e.foot_slip=torch.zeros((1,8,2));e.gait_contact=torch.ones((1,8,2),dtype=torch.bool)
    e.gait_command=torch.zeros((1,8,3));e.gait_bad_contact=torch.zeros((1,8),dtype=torch.int32)
    e.foot_hits=torch.zeros((1,8,2),dtype=torch.int32);e.nonfoot_ball_hits=torch.zeros((1,8),dtype=torch.int32)
    e.touches=torch.zeros((1,8),dtype=torch.int32);e.ever_feet=torch.zeros((1,8,2),dtype=torch.bool)
    e.team_progress=torch.zeros((1,2));e.forward_kick=torch.zeros((1,8));e.side_kick=torch.zeros((1,8))
    e.tensor=lambda x,dtype=torch.float32:torch.as_tensor(x,dtype=dtype)
    e.observe=lambda:torch.zeros((1,8,135));e.state=lambda obs=None:torch.zeros((1,8,1084))
    return e,MatchTask(e)


def test_body_contact_breaks_potential_and_cannot_launder_invalid_advancement(physical_state):
    e,t=physical_state;t.discount=.999
    def step(x,foot=False,body=False):
        before=e.ball.clone();e.ball[0,0]=x;e.foot_hits.zero_();e.nonfoot_ball_hits.zero_()
        e.foot_hits[0,2,0]=int(foot);e.nonfoot_ball_hits[0,2]=int(body)
        t.step(torch.zeros((1,8,22)),torch.zeros((1,8,22)),before,e.xpos[:,:8].clone(),
               dict(goal=torch.zeros(1),out=torch.zeros(1,dtype=torch.bool)))
    step(.02,foot=True);assert t.foot_valid[0]
    step(.8,body=True)
    assert not t.foot_valid[0] and (t.last_terms[MatchTask.terms.index('ball_potential')]<=0).all()
    assert not t.attack_paid.any()
    step(.8,foot=True)
    assert t.foot_valid[0] and t.segment_start[0]==pytest.approx(.8)
    assert not t.attack_paid.any() and e.team_progress.max()<.03
    step(.92,foot=True);assert t.attack_paid[0,0] and not t.attack_paid[0,1]


def test_ball_motion_without_robot_motion_cannot_pay_pursuit(physical_state):
    e,t=physical_state;before=e.ball.clone();e.ball[0,0]=.5
    t.step(torch.zeros((1,8,22)),torch.zeros((1,8,22)),before,e.xpos[:,:8].clone(),
           dict(goal=torch.zeros(1),out=torch.zeros(1,dtype=torch.bool)))
    assert not t.last_terms[MatchTask.terms.index('approach')].any()


def test_disabling_gait_zeros_all_style_terms_but_keeps_football_rewards(physical_state):
    e,t=physical_state;t.gait_reward_enabled=False
    e.foot_slip.fill_(.2);e.qvel.fill_(5.)
    e.xpos[:,e.feet[:,0],1]+=.4
    f=features();f['linear_error'].zero_();f['forward_speed'].fill_(.18);f['lift_credit'].fill_(1.)
    f['landing'][:,:,0]=True;f['alternating'][:,:,0]=True;f['crossed'].fill_(True)
    e.match_gait_features=f;e.foot_hits[0,2,0]=1
    *_,reward,term,trunc,info=t.step(torch.ones((1,8,22)),torch.zeros((1,8,22)),e.ball.clone(),e.xpos[:,:8].clone(),
        dict(goal=torch.ones(1),out=torch.zeros(1,dtype=torch.bool)))
    for key in MatchTask.gait_terms:assert not t.last_terms[MatchTask.terms.index(key)].any()
    assert t.last_terms[MatchTask.terms.index('foot_contact'),0]>0
    assert t.last_terms[MatchTask.terms.index('goal_for'),0]==20
    assert t.last_residual==0 and torch.isfinite(reward).all()


def test_disabling_gait_removes_stance_width_from_survival_and_pursuit_quality():
    up=torch.ones(2);height=torch.full((2,),.417);width=torch.tensor([.1754,.6]);bad=torch.zeros(2,dtype=torch.bool)
    quality,healthy,_,_=MatchTask.posture(up,height,width,bad,width_reward=False)
    torch.testing.assert_close(quality,torch.ones(2))
    assert healthy.tolist()==[True,False]  # Diagnostics retain the actual stance measurement.
    assert MatchTask.posture(up,height,width,bad)[0][1]<1e-5


def test_gait_reward_cli_distinguishes_resume_default_from_explicit_disable():
    from train import parser
    assert parser().parse_args([]).match_gait_reward is None
    assert parser().parse_args(['--no-match-gait-reward']).match_gait_reward is False
    assert parser().parse_args(['--match-gait-reward']).match_gait_reward is True


def test_locomotion_and_match_observations_have_distinct_fields(monkeypatch):
    from q1_4v4.course_env import CourseWarpEnv
    from q1_4v4.warp_env import WarpSoccerEnv
    from q1_4v4 import gait
    e=CourseWarpEnv.__new__(CourseWarpEnv)
    e.ready=True;e.course='match';e.device='cpu';e.steps=torch.zeros(1);e.episode_steps=1500;e.course_amplitude=.45
    e.tensor=lambda x:torch.tensor(x)
    base=torch.zeros((1,8,121));base[:,:,115:119]=torch.eye(4).repeat(2,1)
    monkeypatch.setattr(WarpSoccerEnv,'observe',lambda _:base.clone())
    monkeypatch.setattr(gait,'observations',lambda _:torch.full((1,8,7),.25))
    e.match_task=SimpleNamespace(observations=lambda _:torch.full((1,8,7),.75))
    obs=e.observe()
    assert obs.shape==(1,8,135)
    assert (obs[:,:,115:122]==.25).all() and (obs[:,:,122:129]==.75).all()
    torch.testing.assert_close(obs[:,:,-6:-2],base[:,:,115:119])


def test_old_match_inputs_and_adam_moments_migrate_without_losing_columns():
    torch.set_num_threads(1)
    def policy(raw):
        env=SimpleNamespace(observation_space=Box(-np.inf,np.inf,(raw,),dtype=np.float32),
            share_observation_space=Box(-np.inf,np.inf,(raw*8+4,),dtype=np.float32),action_space=Box(-1.,1.,(22,),dtype=np.float32))
        return make_team(make_args(1,8,1),env,torch.device('cpu'),gpu_rollout=True,shared=True)[:2]
    old,trainer=policy(128)
    for module,opt in ((old.actor,old.actor_optimizer),(old.critic,old.critic_optimizer)):
        for param in module.parameters():opt.state[param]=dict(step=torch.tensor(42.),exp_avg=torch.full_like(param,.2),exp_avg_sq=torch.full_like(param,.3))
    saved=save_team(old,trainer);new,new_trainer=policy(135)
    load_team(saved,new,new_trainer,match_features=True)
    for name,central in (('actor',False),('critic',True)):
        source=getattr(old,name);target=getattr(new,name);key=next(k for k in source.state_dict() if k.endswith('mlp.fc1.0.weight'))
        mapping=_input_map(source.state_dict()[key].shape[-1],target.state_dict()[key].shape[-1],central,True)
        torch.testing.assert_close(target.state_dict()[key][:,mapping],source.state_dict()[key],rtol=0,atol=0)
        old_param=dict(source.named_parameters())[key];new_param=dict(target.named_parameters())[key]
        state=getattr(new,name+'_optimizer').state[new_param]
        torch.testing.assert_close(state['exp_avg'][:,mapping],getattr(old,name+'_optimizer').state[old_param]['exp_avg'],rtol=0,atol=0)
        assert state['step']==42
        if not central:assert not state['exp_avg'][:,115:122].any() and not target.state_dict()[key][:,115:122].any()
    reloaded,t=policy(135);load_team(save_team(new,new_trainer),reloaded,t)
    for a,b in zip(new.actor.parameters(),reloaded.actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_temporal_resume_and_timeout_masks_keep_correct_value_boundaries():
    args=make_args();options=SimpleNamespace(gamma=None,gae_lambda=None)
    checkpoint=dict(args=dict(gamma=.999,gae_lambda=.99))
    assert not configure_temporal(args,options,checkpoint) and args.gamma==.999 and args.gae_lambda==.99
    assert configure_temporal(args,SimpleNamespace(gamma=.998,gae_lambda=.98),checkpoint)
    with pytest.raises(ValueError):configure_temporal(args,SimpleNamespace(gamma=0,gae_lambda=.99))
    r=GpuRollout(1,1,'cpu',135);r.values[0].fill_(1);r.values[1].fill_(999);r.rewards[0].fill_(.999*3);r.masks.zero_()
    r.compute_returns(.999,.99);torch.testing.assert_close(r.returns,torch.full_like(r.returns,2.997))
    r.rewards.fill_(20);r.compute_returns(.999,.99);assert (r.returns==20).all()
