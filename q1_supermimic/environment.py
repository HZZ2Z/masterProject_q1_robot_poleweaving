"""Q1 V3 soccer task adapter; policy sees only the 79 virtual-vision inputs."""
import copy
import hashlib
import json
import torch
import yaml
from envs.q1 import Q1
from . import ROOT
from .metrics import EpisodeMetrics


def load_environment(device, count):
    cfg = yaml.safe_load((ROOT / 'envs/Q1.yaml').read_text())
    cfg['basic'].update(device=device, headless=True)
    cfg['env']['num_envs'] = count
    cfg['physics']['num_threads'] = 2
    # Make all assets independent of the caller's current working directory.
    for section, key in [('asset', 'file'), ('asset', 'mujoco_file'), ('motion', 'motion_path')]:
        cfg[section][key] = str(ROOT / cfg[section][key])
    return SoccerEnvironment(cfg)


class SoccerEnvironment:
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        self.env = Q1(self.cfg)
        self.episode_metrics = EpisodeMetrics()
        self.env.transition_callback = self._capture
        self.env.set_training_iteration(0)
        self.env.reset()
        self.observation_dim = self.env.num_obs
        self.state_dim = self.state().shape[-1]
        task = {k: self.cfg[k] for k in ('rewards', 'init_state', 'curriculum', 'control', 'randomization', 'asset')}
        self.task_signature = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
        self.terminal_state = None

    def state(self):
        e = self.env
        # Privileged dynamics inputs include actual ball/root state and delayed
        # PD targets. The actor always receives only the first 79 coordinates.
        return torch.cat((e.obs_buf, e.critic_obs_buf, e.qpos, e.qvel,
                          e.last_dof_targets, e.pd_kp / 80, e.pd_kd / 3,
                          e.delay_steps[:, None].float() / 10,
                          e.ball_delay_steps[:, None].float() / 10,
                          e.ball_obs_buffer.flatten(1), e.default_dof_pos,
                          e.mass, e.friction.flatten(1),
                          e.episode_length_buf[:, None].float() * e.dt / 60,
                          e.goal_cnt[:, None] / 50,
                          e.task_progress[:, None],
                          e.near_ball_idle_s[:, None], e.far_ball_idle_s[:, None],
                          e.near_ball_active[:, None].float(),
                          e.near_ball_stall[:, None], e.far_ball_stall[:, None],
                          e.approach_window_steps[:, None].float() * e.dt,
                          e.approach_anchor_base, e.approach_anchor_ball,
                          e.near_start_episode[:, None].float(),
                          torch.full((e.num_envs, 1), e.near_start_progress, device=e.device),
                          torch.nn.functional.one_hot(e.reset_group.clamp_min(0), 3).float()), -1).clone()

    def _capture(self, env):
        self.terminal_state = self.state()
        forward_kick, side_kick = env._kick_reward_components()
        self.task_feedback = {
            'foot_ball_gap': env.foot_ball_gap.clone(),
            'kick_event': (forward_kick + side_kick > 0).clone(),
            'goal_distance': (env.ball_pos[:, :2] - env.ball_pos.new_tensor([7., 0.])).norm(dim=-1).clone(),
        }
        self.episode_metrics.update({
            'steps': torch.ones_like(env.rew_buf[:, 0]),
            'reward': env.rew_buf.sum(-1),
            'success_rate': env.extras['success'],
            'fall_rate': env.extras['fall'],
            'termination_height_rate': env.extras['termination_height'],
            'termination_velocity_rate': env.extras['termination_velocity'],
            **env.extras['rew_terms'], **env.extras['soccer_episode'],
        }, env.reset_buf)
        self.transition_metrics = {
            **{f'reward_terms/{k}': v.mean() for k, v in env.extras['rew_terms'].items()},
            **{f'curriculum/reset_group_{i}_fraction': (env.reset_group == i).float().mean() for i in range(3)},
            'soccer/near_ball_fraction': env.near_ball_active.float().mean(),
            'soccer/near_ball_stall_fraction': (env.near_ball_stall > 0).float().mean(),
            'soccer/far_ball_stall_fraction': (env.far_ball_stall > 0).float().mean(),
            'soccer/foot_ball_gap_mean': env.foot_ball_gap.mean(),
            'soccer/ball_forward_distance_per_step': env.extras['soccer_episode']['ball_forward_distance'].mean(),
            'soccer/kick_event_fraction': self.task_feedback['kick_event'].float().mean(),
        }

    def step(self, action):
        _, reward, done, extra = self.env.step(action)
        self.env.batch.check()
        # Reward is integrated over the control period by the soccer task.
        reward = reward.sum(-1).clone()
        done = done.clone()
        timeout = extra['time_outs'].clone()
        fall = done & ~timeout
        info = {'done': done, 'fall': fall, 'timeout': timeout,
                'task_feedback': self.task_feedback,
                'success': extra['success'].clone() * done,
                'terminated': fall | (extra['success'].bool() & done),
                'soccer_progress': self.env.soccer_progress, 'metrics': self.transition_metrics}
        return self.state(), self.terminal_state, reward, info
