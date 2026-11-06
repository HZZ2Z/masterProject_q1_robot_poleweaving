"""Target separation, actor/Adam compatibility and rule attribution."""
from types import SimpleNamespace
import numpy as np
import torch
from gymnasium.spaces import Box
from q1_4v4.learning import make_args,make_team,load_team,save_team,with_team_identity
from q1_4v4.gpu_learning import GpuRollout,DeviceValueNorm
from q1_4v4.match_rules import MatchRules
from q1_4v4.amp import MotionPrior
from q1_4v4.scene import ROOT


def policy(grouped):
    args=make_args(1,2,1);args.amp_groups=grouped
    env=SimpleNamespace(observation_space=Box(-np.inf,np.inf,(135,),dtype=np.float32),
        share_observation_space=Box(-np.inf,np.inf,(1084,),dtype=np.float32),
        action_space=Box(-1.,1.,(22,),dtype=np.float32))
    p,t,_=make_team(args,env,'cpu',gpu_rollout=True,shared=True)
    return p,t,args


def test_grouped_head_keeps_actor_main_value_and_old_adam_exact():
    torch.set_num_threads(2)
    old,t,_=policy(False)
    for module,opt in ((old.actor,old.actor_optimizer),(old.critic,old.critic_optimizer)):
        for p in module.parameters():
            opt.state[p]=dict(step=torch.tensor(42.),exp_avg=torch.full_like(p,.2),exp_avg_sq=torch.full_like(p,.3))
    saved=save_team(old,t);new,trainer,_=policy(True);load_team(saved,new,trainer)
    state=torch.randn(8,1086);rnn=torch.zeros(8,1,128);mask=torch.ones(8,1)
    torch.testing.assert_close(old.get_values(state,rnn,mask),new.get_values(state,rnn,mask),rtol=0,atol=0)
    assert not new.critic.v_out.amp_value.any()
    for name,p in new.critic.named_parameters():
        adam=new.critic_optimizer.state[p]
        if '.amp_' in name:assert not adam
        else:
            assert adam['step']==42
            torch.testing.assert_close(adam['exp_avg'],torch.full_like(p,.2),rtol=0,atol=0)
    for a,b in zip(old.actor.parameters(),new.actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    again,ta,_=policy(True);load_team(save_team(new,trainer),again,ta)
    for a,b in zip(new.critic.parameters(),again.critic.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Explicitly disabled AMP can still load a grouped checkpoint.
    legacy,tl,_=policy(False);load_team(save_team(new,trainer),legacy,tl)


def test_grouped_returns_use_separate_rewards_values_and_individual_masks():
    r=GpuRollout(2,1,'cpu',amp_groups=True)
    r.rewards.fill_(100);r.amp_rewards.fill_(.3)
    r.values.fill_(2);r.amp_values.fill_(.1)
    r.masks[0,0,0]=0
    r.compute_returns(.9,1.)
    torch.testing.assert_close(r.returns[0,0,0],torch.tensor([100.]))
    torch.testing.assert_close(r.amp_returns[0,0,0],torch.tensor([.3]))
    torch.testing.assert_close(r.returns[0,0,1],torch.tensor([191.62]))
    torch.testing.assert_close(r.amp_returns[0,0,1],torch.tensor([.651]))


def test_grouped_ppo_trains_both_heads_without_losing_small_amp_signal():
    torch.set_num_threads(2)
    p,t,args=policy(True);r=GpuRollout(2,1,'cpu',135,amp_groups=True)
    r.obs.normal_();r.returns.normal_().mul_(1000);r.amp_returns.normal_().mul_(.0001)
    r.amp_active.fill_(True)
    # Preserve the likelihood used to sample exactly.
    with torch.no_grad():
        for step in range(2):
            obs=with_team_identity(r.obs[step,0],torch.arange(8)//4)
            a,lp,_=p.actor(obs,torch.zeros(8,1,128),torch.ones(8,1))
            r.actions[step,0]=a;r.log_probs[step,0]=lp
    metrics=r.update(None,t,args)
    assert metrics['amp_advantage_abs']>.4 and metrics['task_advantage_abs']>1
    assert all(np.isfinite(v) for v in metrics.values())
    assert p.critic.v_out.amp_weight.norm()>0 and t.amp_value_normalizer.debiasing_term>0
    # A minibatch/course without style eligibility is safe, with zero AMP loss.
    r.amp_active.zero_();metrics=r.update(None,t,args)
    assert metrics['amp_advantage_abs']==metrics['amp_value_loss']==0


def test_masked_value_norm_ignores_other_courses_and_empty_batches():
    norm=DeviceValueNorm('cpu')
    norm.update(torch.tensor([[1.],[1000.],[3.]]),torch.tensor([[1.],[0.],[1.]]))
    mean,var=norm.running_mean_var()
    torch.testing.assert_close(mean,torch.tensor([2.]));torch.testing.assert_close(var,torch.tensor([1.]))
    before={k:v.clone() for k,v in norm.state_dict().items()}
    norm.update(torch.ones(2,1),torch.zeros(2,1))
    for k,v in norm.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)


def test_penalty_areas_apply_symmetrically_and_exempt_goalkeepers():
    pos=torch.zeros(1,8,3);pos[0,:,0]=torch.tensor([-6.,-6.,6.,4.79,6.,-6.,6.,6.])
    pos[0,6,1]=2.81;pos[0,7,0]=7.01
    assert MatchRules.penalty_area(pos).tolist()==[[False,True,True,False,False,True,False,False]]


def test_last_touch_orders_physics_steps_and_splits_simultaneous_contacts():
    rules=MatchRules(2,'cpu');times=torch.full((2,8),-1.)
    times[0,1]=.002;times[0,6]=.018
    rules.update_touches(times)
    assert rules.last_touch_robot.tolist()==[6,-1] and rules.last_touch_team.tolist()==[1,-1]
    costs=rules.out_penalty(torch.tensor([True,True]));assert costs[0,6]==-1 and costs.sum()==-1
    times[0,2]=.018;rules.update_touches(times)
    assert rules.last_touch_robot[0]==-1 and rules.last_touch_team[0]==-1
    costs=rules.out_penalty(torch.tensor([True,False]));assert costs[0,2]==costs[0,6]==-.5
    rules.reset_worlds(torch.tensor([0]));assert not rules.last_touch_mask.any()


def test_amp_online_normalizer_and_bounded_score_do_not_have_dead_zone():
    torch.set_num_threads(2)
    prior=MotionPrior(ROOT/'resources/motions/q1','cpu',.3,0)
    prior.sample_count.zero_();prior.mean.zero_();prior.variance.zero_()
    pairs=torch.stack((torch.ones(118),torch.full((118,),3.)))
    prior.update_normalizer(pairs)
    torch.testing.assert_close(prior.mean,torch.full((59,),2.));torch.testing.assert_close(prior.variance,torch.ones(59))
    assert prior.style_score(torch.tensor(-1.)).item()>.6
    scores=prior.style_score(torch.tensor([-100.,-1.,0.,1.,100.]))
    assert torch.all(scores[1:]>scores[:-1]) and scores.min()>=0 and scores.max()<=2
