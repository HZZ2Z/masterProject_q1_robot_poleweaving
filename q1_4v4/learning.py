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


def load_team(state, policy, trainer):
    policy.actor.load_state_dict(state['actor'])
    policy.critic.load_state_dict(state['critic'])
    policy.actor_optimizer.load_state_dict(state['actor_optimizer'])
    policy.critic_optimizer.load_state_dict(state['critic_optimizer'])
    trainer.value_normalizer.load_state_dict(state['value_normalizer'])
