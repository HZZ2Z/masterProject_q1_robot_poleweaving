from types import SimpleNamespace
import numpy as np
import pytest
import torch
from q1_4v4.learning import make_args,make_team,with_team_identity,policy_sharing
from q1_4v4.env import SoccerEnv
from q1_4v4.gpu_learning import GpuRollout,DeviceValueNorm


def test_identity_preserves_roles_and_both_teams_use_one_actor():
    env=SoccerEnv();args=make_args(1,8,1)
    policy,trainer,buffer=make_team(args,env,torch.device('cpu'),shared=True)
    assert buffer.obs.shape==(9,1,8,123)
    obs=with_team_identity(env.observe(),np.arange(8)//4)
    np.testing.assert_array_equal(obs[:,-6:-2],env.roles)
    np.testing.assert_array_equal(obs[:,-8:-6],np.eye(2)[np.arange(8)//4])
    recurrent=np.zeros((8,1,args.hidden_size),np.float32);mask=np.ones((8,1),np.float32)
    action,_=policy.act(obs,recurrent,mask,deterministic=True)
    for side in (slice(0,4),slice(4,8)):
        policy.actor.zero_grad()
        action[side].square().mean().backward(retain_graph=True)
        assert sum(p.grad.abs().sum() for p in policy.actor.parameters() if p.grad is not None)>0


def test_shared_ppo_mixes_red_blue_and_reconstructs_team_relative_state():
    rollout=GpuRollout(2,1,'cpu',obs_dim=128)
    for a in range(8):
        rollout.obs[:,:,a].fill_(a)
        rollout.actions[:,:,a,0]=a
        rollout.returns[:,:,a]=float(a)
    samples=[]
    def update(sample):
        samples.append(sample)
        return (*[torch.tensor(0.) for _ in range(5)],torch.ones((len(sample[0]),1)))
    trainer=SimpleNamespace(value_normalizer=DeviceValueNorm('cpu'),ppo_update=update,clip_param=.2)
    rollout.update(None,trainer,make_args(1,2,1))
    ids=torch.cat([sample[4][:,0].long() for sample in samples])
    assert torch.equal(ids.sort().values,torch.arange(8).repeat_interleave(2))
    for sample in samples:
        state,obs,_,_,action=sample[:5]
        robot=action[:,0].long();owner=robot//4
        assert obs.shape[1]==130 and state.shape[1]==1030
        torch.testing.assert_close(obs[:,-8:-6],torch.nn.functional.one_hot(owner,2).float())
        torch.testing.assert_close(state[:,-6:-4],torch.nn.functional.one_hot(owner,2).float())
        order=(torch.arange(8)[None]+owner[:,None]*4)%8
        torch.testing.assert_close(state[:,:1024].reshape(-1,8,128)[:,:,0],order.float())
        torch.testing.assert_close(state[:,-4:],torch.eye(4)[robot%4])
        # Advantages use a single normalization across all eight robots.
        expected=(robot.float()-3.5)/(torch.arange(8.).std(unbiased=False)+1e-5)
        torch.testing.assert_close(sample[10].flatten(),expected)


def test_resume_layout_is_explicit_and_cannot_silently_drop_a_team():
    assert policy_sharing(SimpleNamespace(policy_sharing=None))
    assert not policy_sharing(SimpleNamespace(policy_sharing=None),{'teams':[{},{}]})
    with pytest.raises(ValueError,match='start without'):
        policy_sharing(SimpleNamespace(policy_sharing='shared'),{'teams':[{},{}]})


def test_ppo_uses_participants_at_each_step_before_rally_reset():
    rollout=GpuRollout(2,1,'cpu',obs_dim=128)
    rollout.active.zero_();rollout.active[0,0,1]=True;rollout.active[1,0,6]=True
    for a in range(8):rollout.actions[:,:,a,0]=a
    samples=[]
    def update(sample):
        samples.append(sample)
        return (*[torch.tensor(0.) for _ in range(5)],torch.ones((len(sample[0]),1)))
    trainer=SimpleNamespace(value_normalizer=DeviceValueNorm('cpu'),ppo_update=update,clip_param=.2)
    rollout.update(None,trainer,make_args(1,2,1),active=rollout.active)
    assert sorted(torch.cat([s[4][:,0] for s in samples]).tolist())==[1.,6.]
