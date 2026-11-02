"""Official MAPPO optimizer/buffer with a learned role-conditioned actor base."""
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace
from gymnasium.spaces import Box
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.algorithms.utils.distributions import FixedNormal
from .course_config import command
from .history import HistoryEncoder

MARCHING_EXPLORATION_REVISION=1
MATCH_EXPLORATION_REVISION=1


class RoleBase(nn.Module):
    def __init__(self, base, hidden):
        super().__init__()
        self.base = base
        self.role_encoder = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, hidden), nn.Tanh())

    def forward(self, obs):
        return self.base(obs) + self.role_encoder(obs[..., -6:-2])


class MarchingRoleBase(RoleBase):
    """Carry a nonlearned course flag to the Gaussian, retaining all saved parameters."""
    def __init__(self,base,hidden,match_exploration=False):
        super().__init__(base,hidden)
        self.match_exploration=match_exploration
        self.register_buffer('marching_command',torch.tensor(command('step_in_place')),persistent=False)
        self.register_buffer('match_commands',torch.tensor([command('teamplay'),command('match')]),persistent=False)

    def forward(self,obs):
        features=super().forward(obs)
        marching=(obs[...,109:115]-self.marching_command).abs().amax(-1,keepdim=True)<1e-5
        flag=marching.to(features.dtype)
        if self.match_exploration:
            match=(obs[...,None,109:115]-self.match_commands).abs().amax(-1).amin(-1,keepdim=True)<1e-5
            flag=torch.where(match,2.,flag)
        return torch.cat((features,flag),-1)


class HistoricalRoleBase(MarchingRoleBase):
    def __init__(self, base, hidden, frame_dim, bins, match_exploration=False):
        super().__init__(base, hidden, match_exploration)
        self.frame_dim = frame_dim
        self.history_encoder = HistoryEncoder(frame_dim, bins, hidden)

    def forward(self, obs):
        current, history = obs[..., :self.frame_dim], obs[..., self.frame_dim:]
        features = super().forward(current)
        residual = self.history_encoder(history)
        return torch.cat((features[..., :-1] + residual, features[..., -1:]), -1)


class MarchingGaussian(nn.Module):
    """Same Gaussian likelihood in sampling and PPO; only course 3 changes distribution."""
    mean_limit=2.

    @property
    def std_floor(self):
        return .15+.20*min(max((self.mean_mix-.25)/.75,0.),1.)

    def __init__(self,original,mean_mix=1.):
        super().__init__()
        self.mean_mix=mean_mix
        # Keep upstream state-dict paths and optimizer parameter identities.
        self.fc_mean=original.fc_mean
        self.logstd=original.logstd
        joints=torch.zeros(self.fc_mean.out_features,dtype=torch.bool)
        joints[[0,3,6,9]]=True  # hip pitch and knee, mirrored left/right
        self.register_buffer('marching_joints',joints,persistent=False)

    def forward(self,features):
        mean=self.fc_mean(features[...,:-1])
        active=(features[...,-1:]==1)&self.marching_joints
        # Softsign retains a useful derivative even for inherited saturated logits.
        bounded=mean/(1+mean.abs()/self.mean_limit)
        mean=torch.where(active,(1-self.mean_mix)*mean+self.mean_mix*bounded,mean)
        std=self.logstd(torch.zeros_like(mean)).exp()
        std=torch.where(active,std.clamp(min=self.std_floor),std)
        return FixedNormal(mean,std)

    @staticmethod
    def mixing_at(iteration,start):
        return .25+.75*min(max((iteration-start)/128,0.),1.)


class MatchGaussian(MarchingGaussian):
    """Recover saturated match actions without replacing learned parameters."""
    def forward(self,features):
        original=super().forward(features)
        active=features[...,-1:]==2
        mean=original.mean/(1+original.mean.abs()/1.5)
        joint=torch.arange(mean.shape[-1],device=mean.device)
        low=torch.where(joint<12,.20,.05)
        high=torch.where(joint<12,.65,.35)
        std=low+(high-low)*torch.sigmoid(original.stddev.log())
        return FixedNormal(torch.where(active,mean,original.mean),
                           torch.where(active,std,original.stddev))


def make_args(num_envs=4, horizon=128, epochs=5, hidden=128):
    args = get_config().parse_args([])
    args.algorithm_name = 'mappo'
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.use_wandb = False
    args.use_valuenorm = True
    args.use_popart = False
    args.use_proper_time_limits = False  # Timeouts explicitly bootstrap terminal state in runner.
    args.n_rollout_threads = num_envs
    args.episode_length = horizon
    args.ppo_epoch = epochs
    args.hidden_size = hidden
    args.num_mini_batch = 1 if horizon < 32 else 4
    args.lr = args.critic_lr = 3e-4
    args.entropy_coef = .001
    args.max_grad_norm = 1.
    args.marching_exploration_revision=0  # Explicitly restored by training/play checkpoint readers.
    args.marching_exploration_mix=.25
    args.match_exploration_revision=0
    args.history_frames=0
    args.history_bins=5
    args.amp_groups=False
    return args


def configure_temporal(args,options,checkpoint=None):
    saved=checkpoint.get('args',{}) if checkpoint else {}
    changed=False
    for key in ('gamma','gae_lambda'):
        value=getattr(options,key,None)
        if value is None:value=saved.get(key,getattr(args,key))
        valid=(0<value<=1) if key=='gamma' else (0<=value<=1)
        if not valid:raise ValueError('Invalid '+key)
        changed|=bool(checkpoint and saved.get(key,getattr(args,key))!=value)
        setattr(args,key,float(value));setattr(options,key,float(value))
    return changed


def policy_sharing(options, checkpoint=None):
    saved = checkpoint.get('policy_sharing', 'independent') if checkpoint else 'shared'
    selected = getattr(options, 'policy_sharing', None) or saved
    if checkpoint and selected != saved:
        raise ValueError('Policy sharing differs from checkpoint; start without --checkpoint to train from scratch')
    options.policy_sharing = selected
    return selected == 'shared'


def with_team_identity(values, team, role_suffix=6):
    """Insert known red/blue one-hot identity, retaining the role suffix layout."""
    if isinstance(values, torch.Tensor):
        team = torch.as_tensor(team, device=values.device, dtype=torch.long)
        identity = torch.nn.functional.one_hot(team, 2).to(values.dtype)
        identity = torch.broadcast_to(identity, (*values.shape[:-1], 2))
        return torch.cat((values[..., :-role_suffix], identity, values[..., -role_suffix:]), -1)
    identity = np.eye(2, dtype=values.dtype)[team]
    identity = np.broadcast_to(identity, (*values.shape[:-1], 2))
    return np.concatenate((values[..., :-role_suffix], identity, values[..., -role_suffix:]), -1)


def make_team(args, env, device, gpu_rollout=False, shared=False):
    if shared:
        env = SimpleNamespace(observation_space=Box(-np.inf,np.inf,(env.observation_space.shape[0]+2,),dtype=np.float32),
            share_observation_space=Box(-np.inf,np.inf,(env.share_observation_space.shape[0]+2,),dtype=np.float32),
            action_space=env.action_space)
    policy = R_MAPPOPolicy(args, env.observation_space, env.share_observation_space, env.action_space, device)
    exploration=getattr(args,'marching_exploration_revision',0)
    if exploration not in (0,MARCHING_EXPLORATION_REVISION):raise ValueError('Unknown marching exploration revision')
    if exploration and (args.use_recurrent_policy or args.use_naive_recurrent_policy):
        raise ValueError('Marching course flag requires the feedforward MAPPO actor')
    match=getattr(args,'match_exploration_revision',0)
    if match not in (0,MATCH_EXPLORATION_REVISION):raise ValueError('Unknown match exploration revision')
    if match and not exploration:raise ValueError('Match exploration requires task-aware actor')
    if getattr(args,'history_frames',0):
        if not exploration:raise ValueError('History encoder requires task-aware actor')
        policy.actor.base=HistoricalRoleBase(policy.actor.base,args.hidden_size,env.observation_space.shape[0],
                                             args.history_bins,bool(match)).to(device)
    else:
        policy.actor.base=(MarchingRoleBase(policy.actor.base,args.hidden_size,bool(match)) if exploration else
                           RoleBase(policy.actor.base,args.hidden_size)).to(device)
    if exploration:
        head=MatchGaussian if match else MarchingGaussian
        policy.actor.act.action_out=head(policy.actor.act.action_out,args.marching_exploration_mix).to(device)
    policy.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.lr, eps=args.opti_eps)
    if getattr(args,'amp_groups',False):
        from .grouped_learning import GroupedValueHead,GroupedMAPPO
        policy.critic.v_out=GroupedValueHead(policy.critic.v_out)
        policy.critic_optimizer=torch.optim.Adam(policy.critic.parameters(),lr=args.critic_lr,eps=args.opti_eps)
        trainer=GroupedMAPPO(args,policy,device)
    else:trainer = R_MAPPO(args, policy, device)
    if gpu_rollout:
        from .gpu_learning import DeviceValueNorm
        trainer.value_normalizer = DeviceValueNorm(device)
        return policy, trainer, None
    buffer = SharedReplayBuffer(args, 8 if shared else 4, env.observation_space, env.share_observation_space, env.action_space)
    # Upstream Box actor returns ONE summed joint-action log probability.
    # Its generic buffer allocates action_dim columns; avoid broadcasting it 22 times.
    buffer.action_log_probs = np.zeros((*buffer.action_log_probs.shape[:-1], 1), dtype=np.float32)
    return policy, trainer, buffer


def numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def save_team(policy, trainer):
    state=dict(actor=policy.actor.state_dict(), critic=policy.critic.state_dict(),
                actor_optimizer=policy.actor_optimizer.state_dict(),
                critic_optimizer=policy.critic_optimizer.state_dict(),
                value_normalizer=trainer.value_normalizer.state_dict())
    if hasattr(trainer,'amp_value_normalizer'):state['amp_value_normalizer']=trainer.amp_value_normalizer.state_dict()
    return state



def _input_map(old_dim,new_dim,central=False,match_features=False):
    """Preserve input columns and Adam moments when adding distinct match fields."""
    if old_dim==new_dim:return None
    shared=(old_dim-6)%8==0 if central else old_dim in (123,130,137)
    suffix=6 if shared else 4
    old_obs=(old_dim-suffix)//8 if central else old_dim-(2 if shared else 0)
    new_obs=(new_dim-suffix)//8 if central else new_dim-(2 if shared else 0)
    if (old_obs,new_obs) not in ((121,128),(121,135),(128,135)):
        raise ValueError(f'Unsupported observation migration {old_dim}->{new_dim}')
    tail=new_obs-6
    if old_obs==121:local=list(range(115))+list(range(tail,new_obs))
    elif match_features:local=list(range(115))+list(range(122,129))+list(range(tail,new_obs))
    else:local=list(range(122))+list(range(tail,new_obs))
    if central:return [robot*new_obs+j for robot in range(8) for j in local]+list(range(8*new_obs,new_dim))
    if not shared:return local
    return local[:-6]+[tail,tail+1]+[j+2 for j in local[-6:]]


def _expand_input(tensor,shape,index,is_norm=False):
    result=torch.ones(shape,device=tensor.device,dtype=tensor.dtype) if is_norm else torch.zeros(shape,device=tensor.device,dtype=tensor.dtype)
    if tensor.ndim==1:result[index]=tensor
    else:result[:,index]=tensor
    return result


# Retain a strict path for equal-size checkpoints and migrate only supported input tensors.
def load_team(state,policy,trainer,match_features=False):
    for name,module,opt,central in [('actor',policy.actor,policy.actor_optimizer,False),
                                   ('critic',policy.critic,policy.critic_optimizer,True)]:
        target=module.state_dict()
        old=state[name]
        migrated={}
        for key,value in old.items():
            if key.startswith('base.history_encoder.') and key not in target:continue
            if key in ('v_out.amp_weight','v_out.amp_bias') and key not in target:continue
            if value.shape==target[key].shape:
                migrated[key]=value;continue
            if 'feature_norm.' not in key and 'mlp.fc1.0.weight' not in key:
                raise ValueError(f'Unexpected checkpoint shape change {name}/{key}')
            index=_input_map(value.shape[-1],target[key].shape[-1],central,match_features)
            migrated[key]=_expand_input(value,target[key].shape,index,is_norm=key.endswith('feature_norm.weight'))
        missing=module.load_state_dict(migrated,strict=False)
        if missing.unexpected_keys or any(not k.startswith('base.history_encoder.') and k not in
                                         ('v_out.amp_weight','v_out.amp_bias') for k in missing.missing_keys):
            raise ValueError(f'Unsupported missing checkpoint parameters: {missing}')
        optimizer=state[name+'_optimizer']
        if set(old)!=set(target):
            import copy
            optimizer=copy.deepcopy(optimizer)
            if len(optimizer['param_groups'])!=1:raise ValueError('History migration requires one actor parameter group')
            current_names=list(dict(module.named_parameters()))
            old_names=[k for k in old if k in current_names or k.startswith('base.history_encoder.') or
                       k in ('v_out.amp_weight','v_out.amp_bias')]
            old_ids=optimizer['param_groups'][0]['params']
            if len(old_ids)!=len(old_names):raise ValueError('Actor optimizer parameter layout differs')
            new_ids=opt.state_dict()['param_groups'][0]['params'];indices=dict(zip(current_names,new_ids))
            optimizer['state']={indices[k]:optimizer['state'][i] for k,i in zip(old_names,old_ids)
                                if k in indices and i in optimizer['state']}
            optimizer['param_groups'][0]['params']=new_ids
        opt.load_state_dict(optimizer)
        for pname,param in module.named_parameters():
            for key,value in opt.state.get(param,{}).items():
                if isinstance(value,torch.Tensor) and value.ndim and value.shape!=param.shape:
                    index=_input_map(value.shape[-1],param.shape[-1],central,match_features)
                    opt.state[param][key]=_expand_input(value,param.shape,index)
    trainer.value_normalizer.load_state_dict(state['value_normalizer'])
    if hasattr(trainer,'amp_value_normalizer') and 'amp_value_normalizer' in state:
        trainer.amp_value_normalizer.load_state_dict(state['amp_value_normalizer'])
