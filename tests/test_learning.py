import numpy as np
import torch
from q1_4v4.env import SoccerEnv
from q1_4v4.learning import make_args, make_team, save_team, load_team
from q1_4v4.learning import MarchingGaussian
from q1_4v4.learning import MatchGaussian


def test_match_distribution_desaturates_actions_and_preserves_other_tasks():
    from onpolicy.algorithms.utils.distributions import DiagGaussian
    head=DiagGaussian(8,22)
    with torch.no_grad():
        head.fc_mean.weight.zero_();head.fc_mean.bias.fill_(16.)
        head.logstd._bias[:12].fill_(-4.);head.logstd._bias[12:].fill_(2.)
    wrapped=MatchGaussian(head)
    x=torch.zeros((2,9));x[1,-1]=2
    dist=wrapped(x);old=head(x[:,:-1])
    torch.testing.assert_close(dist.mean[0],old.mean[0],rtol=0,atol=0)
    torch.testing.assert_close(dist.stddev[0],old.stddev[0],rtol=0,atol=0)
    assert (dist.mean[1].abs()<1.5).all()
    assert (dist.stddev[1,:12]>.2).all() and (dist.stddev[1,:12]<.65).all()
    assert (dist.stddev[1,12:]>.05).all() and (dist.stddev[1,12:]<.35).all()
    (dist.mean[1].sum()+dist.stddev[1].sum()).backward()
    assert (head.fc_mean.bias.grad>0).all() and (head.logstd._bias.grad>0).all()
    assert set(wrapped.state_dict())==set(head.state_dict())


def test_match_sampling_likelihood_and_checkpoint_are_identical():
    from types import SimpleNamespace
    from gymnasium.spaces import Box
    from q1_4v4.course_config import command
    env=SimpleNamespace(observation_space=Box(-np.inf,np.inf,(128,),dtype=np.float32),
        share_observation_space=Box(-np.inf,np.inf,(1028,),dtype=np.float32),action_space=Box(-1.,1.,(22,),dtype=np.float32))
    args=make_args(1,8,1);args.marching_exploration_revision=1;args.match_exploration_revision=1
    p,t,_=make_team(args,env,torch.device('cpu'),gpu_rollout=True,shared=True)
    obs=torch.zeros((8,130));obs[:4,109:115]=torch.tensor(command('match'));obs[4:,109:115]=torch.tensor(command('balance'))
    rnn=torch.zeros((8,1,args.hidden_size));mask=torch.ones((8,1))
    action,old_logprob,_=p.actor(obs,rnn,mask)
    new_logprob,_=p.actor.evaluate_actions(obs,rnn,action,mask)
    torch.testing.assert_close(old_logprob,new_logprob,rtol=0,atol=0)
    p2,t2,_=make_team(args,env,torch.device('cpu'),gpu_rollout=True,shared=True)
    load_team(save_team(p,t),p2,t2)
    torch.testing.assert_close(p.act(obs,rnn,mask,deterministic=True)[0],p2.act(obs,rnn,mask,deterministic=True)[0],rtol=0,atol=0)


def test_role_gradients_and_checkpoint_roundtrip():
    torch.set_num_threads(2)
    env = SoccerEnv()
    args = make_args(num_envs=1, horizon=8, epochs=1)
    p, t, b = make_team(args, env, torch.device('cpu'))
    assert b.action_log_probs.shape == (8, 1, 4, 1)
    obs = env.observe()[:4]
    states = np.zeros((4, 1, args.hidden_size), np.float32)
    masks = np.ones((4, 1), np.float32)
    actions, _ = p.act(obs, states, masks, deterministic=True)
    loss = actions.square().mean()
    loss.backward()
    grads = [x.grad for x in p.actor.base.role_encoder.parameters()]
    assert all(x is not None and torch.isfinite(x).all() for x in grads)
    assert sum(x.abs().sum() for x in grads) > 0
    p2, t2, _ = make_team(args, env, torch.device('cpu'))
    load_team(save_team(p, t), p2, t2)
    actions2, _ = p2.act(obs, states, masks, deterministic=True)
    assert torch.equal(actions, actions2)


def test_marching_distribution_preserves_other_courses_and_restores_gradients():
    from onpolicy.algorithms.utils.distributions import DiagGaussian
    head=DiagGaussian(8,22)
    with torch.no_grad():
        head.fc_mean.weight.zero_();head.fc_mean.bias.fill_(-8.)
        head.logstd._bias.fill_(-3.)
    wrapped=MarchingGaussian(head)
    features=torch.zeros((2,8));flag=torch.tensor([[0.],[1.]])
    legacy=head(features);dist=wrapped(torch.cat((features,flag),-1))
    torch.testing.assert_close(dist.mean[0],legacy.mean[0],rtol=0,atol=0)
    torch.testing.assert_close(dist.stddev[0],legacy.stddev[0],rtol=0,atol=0)
    joints=[0,3,6,9]
    assert (dist.mean[1,joints].abs()<2).all()
    assert (dist.stddev[1,joints]>=.35).all()
    torch.testing.assert_close(dist.mean[1,[1,2,4]],legacy.mean[1,[1,2,4]])
    dist.mean[1,joints].sum().backward()
    assert (head.fc_mean.bias.grad[joints]>.03).all()
    assert set(wrapped.state_dict())==set(head.state_dict())
    assert wrapped.mixing_at(100,100)==.25
    assert wrapped.mixing_at(164,100)==.625
    assert wrapped.mixing_at(228,100)==wrapped.mixing_at(1000,100)==1.
    wrapped.mean_mix=.25
    assert wrapped.std_floor==.15


def test_marching_checkpoint_and_ppo_log_probabilities_are_consistent():
    from types import SimpleNamespace
    from gymnasium.spaces import Box
    from q1_4v4.course_config import command
    env=SimpleNamespace(observation_space=Box(-np.inf,np.inf,(128,),dtype=np.float32),
        share_observation_space=Box(-np.inf,np.inf,(1028,),dtype=np.float32),
        action_space=Box(-1.,1.,(22,),dtype=np.float32))
    args=make_args(1,8,1);args.marching_exploration_revision=1
    p,t,_=make_team(args,env,torch.device('cpu'),gpu_rollout=True,shared=True)
    obs=torch.zeros((8,130));obs[:,109:115]=torch.tensor(command('step_in_place'))
    obs[4:,109:115]=torch.tensor(command('balance'))
    rnn=torch.zeros((8,1,args.hidden_size));mask=torch.ones((8,1))
    action,old_logprob,_=p.actor(obs,rnn,mask)
    recomputed,_=p.actor.evaluate_actions(obs,rnn,action,mask)
    torch.testing.assert_close(old_logprob,recomputed,rtol=0,atol=0)
    saved=save_team(p,t)
    p2,t2,_=make_team(args,env,torch.device('cpu'),gpu_rollout=True,shared=True)
    load_team(saved,p2,t2)
    recomputed2,_=p2.actor.evaluate_actions(obs,rnn,action,mask)
    torch.testing.assert_close(old_logprob,recomputed2,rtol=0,atol=0)
    # Original actor weights/optimizer states still load into the new wrappers.
    args.marching_exploration_revision=0
    old,trainer,_=make_team(args,env,torch.device('cpu'),gpu_rollout=True,shared=True)
    load_team(save_team(old,trainer),p2,t2)
    torch.testing.assert_close(old.act(obs[4:],rnn[4:],mask[4:],deterministic=True)[0],
        p2.act(obs[4:],rnn[4:],mask[4:],deterministic=True)[0],rtol=0,atol=0)
