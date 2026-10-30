"""Official MAPPO optimizer/buffer with a learned role-conditioned actor base."""
import numpy as np
import torch
from torch import nn
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.utils.shared_buffer import SharedReplayBuffer


class RoleBase(nn.Module):
    def __init__(self, base, hidden):
        super().__init__()
        self.base = base
        self.role_encoder = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, hidden), nn.Tanh())

    def forward(self, obs):
        return self.base(obs) + self.role_encoder(obs[..., -6:-2])


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
    return args


def make_team(args, env, device, gpu_rollout=False):
    policy = R_MAPPOPolicy(args, env.observation_space, env.share_observation_space, env.action_space, device)
    policy.actor.base = RoleBase(policy.actor.base, args.hidden_size).to(device)
    policy.actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.lr, eps=args.opti_eps)
    trainer = R_MAPPO(args, policy, device)
    if gpu_rollout:
        from .gpu_learning import DeviceValueNorm
        trainer.value_normalizer = DeviceValueNorm(device)
        return policy, trainer, None
    buffer = SharedReplayBuffer(args, 4, env.observation_space, env.share_observation_space, env.action_space)
    # Upstream Box actor returns ONE summed joint-action log probability.
    # Its generic buffer allocates action_dim columns; avoid broadcasting it 22 times.
    buffer.action_log_probs = np.zeros((*buffer.action_log_probs.shape[:-1], 1), dtype=np.float32)
    return policy, trainer, buffer


def numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def save_team(policy, trainer):
    return dict(actor=policy.actor.state_dict(), critic=policy.critic.state_dict(),
                actor_optimizer=policy.actor_optimizer.state_dict(),
                critic_optimizer=policy.critic_optimizer.state_dict(),
                value_normalizer=trainer.value_normalizer.state_dict())



def _input_map(old_dim,new_dim,central=False):
    """Map old proprioception/task/role columns around inserted gait observations."""
    if old_dim==new_dim:return None
    old_obs=(old_dim-4)//8 if central else old_dim
    new_obs=(new_dim-4)//8 if central else new_dim
    if old_obs!=121 or new_obs!=128:raise ValueError(f'Unsupported observation migration {old_dim}->{new_dim}')
    local=list(range(115))+list(range(122,128))
    if not central:return local
    return [robot*new_obs+j for robot in range(8) for j in local]+list(range(8*new_obs,8*new_obs+4))


def _expand_input(tensor,shape,index,is_norm=False):
    result=torch.ones(shape,device=tensor.device,dtype=tensor.dtype) if is_norm else torch.zeros(shape,device=tensor.device,dtype=tensor.dtype)
    if tensor.ndim==1:result[index]=tensor
    else:result[:,index]=tensor
    return result


# Retain a strict path for equal-size checkpoints and migrate only supported input tensors.
def load_team(state,policy,trainer):
    for name,module,opt,central in [('actor',policy.actor,policy.actor_optimizer,False),
                                   ('critic',policy.critic,policy.critic_optimizer,True)]:
        target=module.state_dict()
        old=state[name]
        migrated={}
        for key,value in old.items():
            if value.shape==target[key].shape:
                migrated[key]=value;continue
            if 'feature_norm.' not in key and 'mlp.fc1.0.weight' not in key:
                raise ValueError(f'Unexpected checkpoint shape change {name}/{key}')
            index=_input_map(value.shape[-1],target[key].shape[-1],central)
            migrated[key]=_expand_input(value,target[key].shape,index,is_norm=key.endswith('feature_norm.weight'))
        module.load_state_dict(migrated)
        opt.load_state_dict(state[name+'_optimizer'])
        for pname,param in module.named_parameters():
            for key,value in opt.state.get(param,{}).items():
                if isinstance(value,torch.Tensor) and value.ndim and value.shape!=param.shape:
                    index=_input_map(value.shape[-1],param.shape[-1],central)
                    opt.state[param][key]=_expand_input(value,param.shape,index)
    trainer.value_normalizer.load_state_dict(state['value_normalizer'])
