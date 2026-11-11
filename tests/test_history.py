"""History causality, exact policy handoff, optimizer continuation and AMP gates."""
from types import SimpleNamespace
import os
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box
from q1_4v4.history import ActorHistory,history_payload,configure_history
from q1_4v4.learning import make_args,make_team,save_team,load_team,with_team_identity
from q1_4v4.course_config import command
from q1_4v4.amp import MotionPrior
from q1_4v4.scene import MUJOCO_ROOT


def policy(frames=0):
    args=make_args(1,8,1);args.marching_exploration_revision=1;args.match_exploration_revision=1
    args.history_frames=frames;args.history_bins=5
    env=SimpleNamespace(observation_space=Box(-np.inf,np.inf,(135,),dtype=np.float32),
        share_observation_space=Box(-np.inf,np.inf,(1084,),dtype=np.float32),action_space=Box(-1.,1.,(22,),dtype=np.float32))
    return make_team(args,env,torch.device('cpu'),gpu_rollout=True,shared=True)[:2]


def test_history_is_causal_ordered_and_only_resets_selected_robot():
    history=ActorHistory(1,2,1,50,5,'cpu')
    for i in range(1,51):history.push(torch.full((1,2,1),float(i)))
    torch.testing.assert_close(history.pooled()[0,0,:,0].float(),torch.tensor([5.5,15.5,25.5,35.5,45.5]))
    history.push(torch.full((1,2,1),51.),torch.tensor([[True,False]]))
    pooled=history.pooled().float()
    assert not pooled[0,0,:4].any() and pooled[0,0,4,0].item()==pytest.approx(5.1,abs=.003)
    torch.testing.assert_close(pooled[0,1,:,0],torch.tensor([6.5,16.5,26.5,36.5,46.5]))


def test_history_preserves_old_policy_distribution_and_existing_adam_states():
    torch.set_num_threads(2)
    old,trainer=policy()
    for module,opt in ((old.actor,old.actor_optimizer),(old.critic,old.critic_optimizer)):
        for param in module.parameters():opt.state[param]=dict(step=torch.tensor(42.),exp_avg=torch.full_like(param,.2),exp_avg_sq=torch.full_like(param,.3))
    saved=save_team(old,trainer);new,t=policy(50);load_team(saved,new,t,match_features=True)
    obs=torch.randn(1,8,135);obs[:,:,109:115]=torch.tensor(command('match'))
    obs[:,:,-6:-2]=torch.eye(4).repeat(2,1)
    owner=torch.arange(8)//4
    current=with_team_identity(obs[0],owner)
    pooled=torch.randn(8,5,135).half()
    payload=history_payload(current,pooled,True,owner)
    rnn=torch.zeros(8,1,128);mask=torch.ones(8,1)
    before=old.actor(current,rnn,mask,deterministic=True)
    after=new.actor(payload,rnn,mask,deterministic=True)
    torch.testing.assert_close(before[0],after[0],rtol=0,atol=0)
    torch.testing.assert_close(before[1],after[1],rtol=0,atol=0)
    assert payload.shape==(8,822) and current.shape==(8,137)
    for name,param in new.actor.named_parameters():
        state=new.actor_optimizer.state[param]
        if 'history_encoder' in name:assert not state
        else:
            assert state['step']==42
            torch.testing.assert_close(state['exp_avg'],torch.full_like(param,.2),rtol=0,atol=0)
    # The new branch starts neutral but immediately receives a useful gradient.
    new.actor_optimizer.zero_grad();new.actor(payload,rnn,mask,deterministic=True)[0].sum().backward()
    assert new.actor.base.history_encoder.layers[-1].weight.grad.norm()>0
    new.actor_optimizer.step()
    again,ta=policy(50);load_team(save_team(new,t),again,ta)
    for a,b in zip(new.actor.parameters(),again.actor.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_history_configuration_restores_and_can_keep_legacy_mode():
    args=make_args();options=SimpleNamespace(history_frames=None,history_bins=None)
    assert configure_history(args,options,dict(args=dict(history_frames=50,history_bins=5))) is False
    assert args.history_frames==50 and args.history_bins==5
    args=make_args();options=SimpleNamespace(history_frames=None,history_bins=None)
    configure_history(args,options);assert args.history_frames==0
    with pytest.raises(ValueError):configure_history(args,SimpleNamespace(history_frames=49,history_bins=5))


def test_ppo_replays_the_same_history_used_for_sampling():
    from q1_4v4.gpu_learning import GpuRollout
    p,trainer=policy(50);args=make_args(1,2,1)
    rollout=GpuRollout(2,1,'cpu',135)
    rollout.obs.normal_();rollout.obs[:,:,:,109:115]=torch.tensor(command('match'))
    rollout.history=torch.randn(3,1,8,5,135).half()
    owners=torch.arange(8)//4;rnn=torch.zeros(8,1,128);mask=torch.ones(8,1)
    with torch.no_grad():
        for step in range(2):
            current=with_team_identity(rollout.obs[step,0],owners)
            payload=history_payload(current,rollout.history[step,0],True,owners)
            action,logprob,_=p.actor(payload,rnn,mask)
            rollout.actions[step,0]=action;rollout.log_probs[step,0]=logprob
    rollout.returns.normal_()
    metrics=rollout.update(None,trainer,args)
    assert all(np.isfinite(v) for v in metrics.values())
    assert p.actor.base.history_encoder.layers[-1].weight.norm()>0


def test_amp_expert_schema_learning_and_resumption():
    torch.set_num_threads(2)
    prior=MotionPrior(MUJOCO_ROOT/'resources/motions/q1','cpu',.3,100)
    assert prior.expert.pairs.shape[1]==118 and len(prior.expert.pairs)>100
    assert torch.isfinite(prior.normalized(prior.expert.pairs)).all()
    prior.replay[:128]=prior.expert.sample(128);prior.size=128
    values=prior.update();assert all(np.isfinite(v) for v in values.values())
    saved=prior.checkpoint();again=MotionPrior(MUJOCO_ROOT/'resources/motions/q1','cpu',.3,999,saved)
    assert again.start_iteration==100 and saved['motion_sha256']==again.expert.sha256
    for a,b in zip(prior.parameters(),again.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_amp_standing_and_stalling_cannot_collect_imitation_reward():
    prior=MotionPrior(MUJOCO_ROOT/'resources/motions/q1','cpu',.3,0);prior.iteration=200
    with torch.no_grad():
        for param in prior.parameters():param.zero_()
    features=torch.zeros((1,8,59));features[:,:,2]=-1
    prior.features=lambda env:features
    env=SimpleNamespace(count=1,device='cpu',dt=.02,active=torch.ones(1,8,dtype=torch.bool),
        bodies=torch.arange(8),feet=torch.arange(8,24).reshape(8,2),xpos=torch.zeros(1,24,3),
        gait_bad_contact=torch.zeros(1,8,dtype=torch.bool),match_task=SimpleNamespace(
            ball_skill=SimpleNamespace(near_stall=torch.zeros(1,8),far_stall=torch.zeros(1,8))))
    env.xpos[:,:8,2]=.417
    prior.begin_step([('match',env)])
    assert not prior.reward(env).any()
    env.xpos[:,env.feet[2],0]+=.01
    reward=prior.reward(env)
    assert reward[0,2].item()==pytest.approx(.3,abs=1e-7)
    assert reward.max()<=.6
    env.match_task.ball_skill.near_stall[0,2]=1
    assert not prior.reward(env).any()


def test_parallel_match_groups_share_amp_replay_without_cuda_stream_races():
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.mixed_env import MixedCourseEnv
    env=MixedCourseEnv(4,'cuda:0',42,{'teamplay':2,'match':2})
    prior=MotionPrior(MUJOCO_ROOT/'resources/motions/q1','cuda:0',.3,0);prior.iteration=200
    groups=[(name,group) for name,(_,group) in env.groups.items()]
    for _ in range(4):
        prior.begin_step(groups)
        env.step(torch.randn((4,8,22),device='cuda:0')*.2)
        env.finish_step()
    torch.cuda.synchronize()
    assert prior.reward_statistics[1]==4*4*8 and prior.size>0
    assert all(np.isfinite(v) for v in prior.update().values())
    for _,group in groups:assert torch.isfinite(group.match_task.style_reward).all()
