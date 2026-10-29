# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import torch
import torch.nn.functional as F
from utils.wrapper import StackObservationWrapper
from utils.model import *
from utils.buffer import *
from utils.utils import discount_values, surrogate_loss
from utils.recorder import Recorder
from utils.motion import Motion
from envs import *


class Runner:

    def __init__(self):
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        torch.set_num_threads(self.cfg['runner'].get('torch_threads', 4))
        self._configure_multi_gpu()
        self._set_seed()
        task_class = task_registry.get_task_class(self.cfg["basic"]["task"])
        self.env = task_class(self.cfg)
        self.env = StackObservationWrapper(self.env, self.cfg["runner"]["num_stack"])
        self.device = self.cfg["basic"]["device"]

        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = ActorCritic(self.cfg).to(self.device)
        self.motion_sampler = Motion(self.cfg, self.device)
        self.amp_normalizer = Normalizer(self.cfg["env"]["num_amp_obs"], self.device)
        self.amp_mask = torch.ones(self.cfg['env']['num_amp_obs'], device=self.device)
        supervised = self.cfg['motion'].get('amp_joint_names')
        if supervised is not None:
            names = self.cfg['asset']['dof_names'][self.cfg['motion'].get('dof_start_index', 2):]
            for i, name in enumerate(names):
                if name not in supervised:
                    self.amp_mask[9 + i] = 0
                    self.amp_mask[9 + len(names) + i] = 0
        expert, following = next(self.motion_sampler.feed_forward_generator(1, 4096))
        self.amp_normalizer.update(torch.cat((expert, following), dim=0))
        self.discriminator = Discriminator(self.cfg["env"]["num_amp_obs"] * 2).to(self.device)
        self.amp_storage = ReplayBuffer(self.cfg["env"]["num_amp_obs"], self.cfg["motion"]["replay_buffer_size"], self.device)
        self.optimizer = torch.optim.Adam([{"name": "actor_critic", "params": self.model.parameters(), "lr": self.learning_rate}])
        self.discriminator_optimizer = torch.optim.RMSprop(
            [
                {
                    "name": "amp_trunk",
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": 1e-2,
                    "lr": self.cfg["algorithm"]["discriminator_lr"],
                },
                {
                    "name": "amp_head",
                    "params": self.discriminator.amp_linear.parameters(),
                    "weight_decay": 5e-1,
                    "lr": self.cfg["algorithm"]["discriminator_lr"],
                },
            ]
        )
        self.start_iteration = 0
        self.stop_requested = False
        self.policy_bootstrap_version = 0
        self._load_checkpoint()
        bootstrap = self.cfg.get('curriculum', {}).get('policy_bootstrap_version', 0)
        if bootstrap > self.policy_bootstrap_version:
            # Discard the failing action head once, retaining the learned encoder
            # and actor trunk. Future resumes retain the newly learned head.
            with torch.no_grad():
                self.model.actor[-1].weight.zero_()
                self.model.actor[-1].bias.zero_()
                self.model.logstd.fill_(-3.)
            for critic in self.model.critics:
                for layer in critic.modules():
                    if isinstance(layer, torch.nn.Linear):
                        layer.reset_parameters()
            self.optimizer.state.clear()
            self.learning_rate = self.cfg['algorithm']['learning_rate']
            for group in self.optimizer.param_groups:
                group['lr'] = self.learning_rate
            self.policy_bootstrap_version = bootstrap
            print('Standing bootstrap: reset action head and critics, retained encoder/actor trunk', flush=True)
        if self.world_size > 1:
            for p in self.model.parameters():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)
            for p in self.model.buffers():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)
            for p in self.discriminator.parameters():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)
            for p in self.discriminator.buffers():
                if p.data is not None:
                    torch.distributed.broadcast(p.data, src=0)

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.env.num_envs, self.device)
        self.buffer.add_buffer("actions", (self.env.num_actions,))
        self.buffer.add_buffer("obses", (self.env.num_obs,))
        self.buffer.add_buffer("critic_obses", (self.env.num_critic_obs,))
        self.buffer.add_buffer("privileged_obses", (self.env.num_privileged_obs,))
        self.buffer.add_buffer("stacked_obses", (self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("mirrored_obses", (self.env.num_obs,))
        self.buffer.add_buffer("mirrored_stacked_obses", (self.env.num_stack, self.env.num_obs))
        self.buffer.add_buffer("rewards", (2,))
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)

    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", default="Q1", type=str, help="Name of the task to run (default: Q1/MuJoCo Warp).")
        parser.add_argument("--num_threads", type=int, help="CPU threads used by the native MuJoCo reference backend.")
        parser.add_argument('--no_wandb', action='store_true', default=None, help='disable W&B for local checks')
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument(
            "--headless", action="store_true", default=None, help="Run headless without creating a viewer window. Overrides config file if provided."
        )
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--device", type=str, help="Device for simulation and training. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        parser.add_argument("--save_path", type=str, help="Save path of logs. Overrides config file if provided.")
        parser.add_argument("--motion_path", type=str, help="Path of motion files. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", task_registry.get_task_cfg(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                elif arg == "save_path":
                    self.cfg["runner"][arg] = getattr(self.args, arg)
                elif arg == "motion_path":
                    self.cfg["motion"][arg] = getattr(self.args, arg)
                elif arg == "num_threads":
                    self.cfg.setdefault("physics", {})[arg] = getattr(self.args, arg)
                elif arg == 'no_wandb':
                    self.cfg['runner']['use_wandb'] = False
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)

    def _configure_multi_gpu(self):
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.rank = int(os.getenv("RANK", "0"))
        if self.world_size == 1:
            return
        self.cfg["basic"]["device"] = f"cuda:{self.rank}"
        torch.distributed.init_process_group(backend="nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.rank)

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load_checkpoint(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
            root = self.cfg["runner"]["save_path"] if self.cfg["basic"]["task"] == "Q1" else "logs"
            candidates = glob.glob(os.path.join(root, "**/nn/model_*.pth"), recursive=True)
            if not candidates:
                raise FileNotFoundError(f"No checkpoint found under {root}; train this task first")
            self.cfg["basic"]["checkpoint"] = max(candidates, key=os.path.getmtime)
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        if model_dict.get('reward_version', 1) != self.cfg['rewards'].get('version', 1):
            print('Soccer reward changed: retaining policy/optimizer; episode rewards across runs are not directly comparable', flush=True)
        self.policy_bootstrap_version = model_dict.get('policy_bootstrap_version', 0)
        if hasattr(self.env, 'balance_release_iteration'):
            near_state = model_dict.get('near_start_state', {})
            if near_state.get('version') == self.cfg.get('curriculum', {}).get('near_start', {}).get('version'):
                # Restored through a public task method: wrapper keeps private
                # simulator state inaccessible to the generic runner.
                self.env.restore_near_start_state(near_state)
            if model_dict.get('curriculum_version', 1) == self.cfg.get('curriculum', {}).get('version', 1):
                self.env.balance_release_iteration = model_dict.get('balance_release_iteration')
                self.env.soccer_progress = model_dict.get('soccer_progress', 0.)
            else:
                print('Curriculum changed: rechecking standing stability before soccer; retaining policy weights', flush=True)
        self.model.load_state_dict(model_dict["model"], strict=self.cfg["basic"]["task"] == "Q1")
        compatible_amp = (model_dict.get('amp_normalizer_version', 1) == Normalizer.version and
                          model_dict.get('amp_joint_names') == self.cfg['motion'].get('amp_joint_names'))
        if compatible_amp:
            self.discriminator.load_state_dict(model_dict['discriminator'], strict=True)
        else:
            print('AMP normalization changed: keeping policy weights; resetting discriminator and AMP statistics', flush=True)
        if 'iteration' in model_dict:
            self.start_iteration = int(model_dict['iteration'])
        else:
            name = os.path.basename(os.path.realpath(self.cfg['basic']['checkpoint']))
            if name.startswith('model_') and name.endswith('.pth'):
                self.start_iteration = int(name[6:-4])
        if 'optimizer' in model_dict and 'discriminator_optimizer' in model_dict:
            self.optimizer.load_state_dict(model_dict['optimizer'])
            self.learning_rate = float(model_dict['learning_rate'])
            if compatible_amp:
                self.discriminator_optimizer.load_state_dict(model_dict['discriminator_optimizer'])
                normalizer = model_dict['amp_normalizer']
                self.amp_normalizer.mean.copy_(normalizer['mean'])
                self.amp_normalizer.var.copy_(normalizer['var'])
                self.amp_normalizer.count = normalizer['count']
                print(f'Resuming optimizer and AMP statistics at iteration {self.start_iteration}', flush=True)
        else:
            print(f'Legacy weights at iteration {self.start_iteration}: optimizer and AMP statistics start fresh', flush=True)
        old_scales = model_dict.get('action_scale')
        if old_scales is None:
            parent = os.path.dirname(os.path.realpath(self.cfg['basic']['checkpoint']))
            for path in (os.path.join(parent, 'config.yaml'), os.path.join(parent, '..', 'config.yaml')):
                if os.path.isfile(path):
                    with open(path) as f:
                        old_scales = yaml.safe_load(f)['control']['action_scale']
                    break
        if old_scales is not None:
            old_scale = torch.as_tensor(old_scales, device=self.device)
            new_scale = torch.as_tensor(self.cfg['control']['action_scale'], device=self.device)
            if not torch.allclose(old_scale, new_scale):
                ratio = old_scale / new_scale
                with torch.no_grad():
                    self.model.actor[-1].weight.mul_(ratio[:, None])
                    self.model.actor[-1].bias.mul_(ratio)
                    self.model.logstd.add_(ratio.log())
                self.optimizer.state.clear()
                self.learning_rate = self.cfg['algorithm']['learning_rate']
                for group in self.optimizer.param_groups:
                    group['lr'] = self.learning_rate
                print('Action ranges changed: rescaled policy outputs to preserve joint targets; reset policy optimizer moments', flush=True)

    def request_stop(self, signum=None, frame=None):
        self.stop_requested = True

    def _checkpoint(self, iteration):
        return {
            'model': self.model.state_dict(), 'discriminator': self.discriminator.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict(),
            'amp_normalizer': {'mean': self.amp_normalizer.mean, 'var': self.amp_normalizer.var,
                               'count': self.amp_normalizer.count},
            'iteration': iteration, 'learning_rate': self.learning_rate,
            'physics_backend': self.cfg['sim']['physics_engine'],
            'reward_version': self.cfg['rewards'].get('version', 1),
            'amp_normalizer_version': Normalizer.version,
            'amp_joint_names': self.cfg['motion'].get('amp_joint_names'),
            'policy_bootstrap_version': self.policy_bootstrap_version,
            'balance_release_iteration': getattr(self.env, 'balance_release_iteration', None),
            'curriculum_version': self.cfg.get('curriculum', {}).get('version', 1),
            'soccer_progress': getattr(self.env, 'soccer_progress', 0.),
            'near_start_state': self.env.near_start_state() if hasattr(self.env, 'near_start_state') else {},
            'action_scale': self.cfg['control']['action_scale'],
        }

    def _normalize_amp(self, observations):
        return self.amp_normalizer.normalize(observations) * self.amp_mask

    def train(self):
        if self.rank == 0:
            self.recorder = Recorder(self.cfg)
        if hasattr(self.env, 'set_training_iteration'):
            self.env.set_training_iteration(self.start_iteration)
        obs, infos = self.env.reset()
        critic_obs = infos["critic_obs"]
        privileged_obs = infos["privileged_obs"]
        stacked_obs = infos["stacked_obs"]
        amp_obs = infos["amp_obs"]
        for it in range(self.start_iteration, self.cfg["basic"]["max_iterations"]):
            if it != self.start_iteration and hasattr(self.env, 'set_training_iteration') and self.env.set_training_iteration(it):
                # Preserve live trajectories at the handoff; only future resets
                # gradually mix reference states. Save the standing policy first.
                if self.rank == 0 and not self.env.balance_phase:
                    self.recorder.save(self._checkpoint(it), it)
                    target = os.path.join(self.recorder.model_dir, 'standing.pth')
                    if os.path.lexists(target):
                        os.unlink(target)
                    os.symlink(f'model_{it}.pth', target)
            iteration_start = time.perf_counter()
            # within horizon_length, env.step() is called with same act
            for n in range(self.cfg["runner"]["horizon_length"]):
                self.buffer.update_data("obses", n, obs)
                self.buffer.update_data("critic_obses", n, critic_obs)
                self.buffer.update_data("privileged_obses", n, privileged_obs)
                self.buffer.update_data("stacked_obses", n, stacked_obs)
                self.buffer.update_data("mirrored_obses", n, self.env.mirror_obs(obs))
                self.buffer.update_data("mirrored_stacked_obses", n, self.env.mirror_obs(stacked_obs))
                with torch.no_grad():
                    dist, _ = self.model.act(obs, stacked_obs)
                    act = dist.sample()
                obs, rew, done, infos = self.env.step(act)
                critic_obs = infos["critic_obs"]
                privileged_obs = infos["privileged_obs"]
                stacked_obs = infos["stacked_obs"]
                last_amp_obs = infos["last_amp_obs"]
                self.amp_storage.insert(amp_obs, last_amp_obs)
                with torch.no_grad():
                    amp_obs = self._normalize_amp(amp_obs)
                    last_amp_obs = self._normalize_amp(last_amp_obs)
                    amp_rew = 1 + torch.tanh(0.4 * self.discriminator(torch.cat([amp_obs, last_amp_obs], dim=-1)).squeeze(-1))
                    reward = rew.clone()
                    amp_scale = self.cfg["algorithm"]["amp_reward_coef"] * getattr(self.env, 'amp_reward_scale', 1.)
                    amp_multiplier = infos.get('amp_reward_multiplier', 1.)
                    applied_amp = amp_scale * amp_rew * amp_multiplier
                    reward[:, 1] += applied_amp
                amp_obs = infos["amp_obs"]
                self.buffer.update_data("actions", n, act)
                self.buffer.update_data("rewards", n, reward)
                self.buffer.update_data("dones", n, done)
                self.buffer.update_data("time_outs", n, infos["time_outs"])
                ep_info = {
                    "reward": reward.sum(dim=-1),
                    "success_rate": infos["success"],
                    "fall_rate": infos["fall"],
                    'duration_s': torch.full_like(reward[:, 0], self.env.dt),
                }
                for cause in ('height', 'velocity'):
                    if f'termination_{cause}' in infos:
                        ep_info[f'termination_{cause}_rate'] = infos[f'termination_{cause}']
                infos["rew_terms"].update({"amp": applied_amp})
                ep_info.update(infos["rew_terms"])
                ep_info.update(infos.get('soccer_episode', {}))
                if self.rank == 0:
                    self.recorder.record_episode_statistics(done, ep_info, it, n == (self.cfg["runner"]["horizon_length"] - 1))

            with torch.no_grad():
                old_dist, _ = self.model.act(self.buffer["obses"], self.buffer["stacked_obses"])
                old_actions_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

                values = self.model.est_values(self.buffer["critic_obses"])
                last_values = self.model.est_values(critic_obs)
                self.buffer["rewards"][self.buffer["time_outs"]] += self.cfg["algorithm"]["gamma"] * values[self.buffer["time_outs"]]
                advantages = discount_values(
                    self.buffer["rewards"],
                    self.buffer["dones"] | self.buffer["time_outs"],
                    values,
                    last_values,
                    self.cfg["algorithm"]["gamma"],
                    self.cfg["algorithm"]["lam"],
                )
                if self.world_size > 1:
                    all_advantages = [torch.empty_like(advantages) for _ in range(self.world_size)]
                    torch.distributed.all_gather(all_advantages, advantages)
                    all_advantages = torch.cat(all_advantages, dim=1)
                else:
                    all_advantages = advantages
                returns = values + advantages
                advantages = (advantages - all_advantages.mean(dim=(0, 1))) / (all_advantages.std(dim=(0, 1)) + 1e-8)

            statistics = {}
            rollout_seconds = time.perf_counter() - iteration_start
            advantage_weights = torch.tensor(self.cfg['algorithm']['advantage_coef'], device=self.device)
            advantage_weights[0] *= getattr(self.env, 'amp_reward_scale', 1.)

            mini_batch_size = (self.env.num_envs * self.cfg["runner"]["horizon_length"]) // self.cfg["algorithm"]["num_mini_batches"]
            indices = torch.randperm(mini_batch_size * self.cfg["algorithm"]["num_mini_batches"], requires_grad=False, device=self.device)
            amp_policy_generator = self.amp_storage.feed_forward_generator(
                self.cfg["algorithm"]["num_learning_epochs"] * self.cfg["algorithm"]["num_mini_batches"],
                mini_batch_size,
            )
            amp_expert_generator = self.motion_sampler.feed_forward_generator(
                self.cfg["algorithm"]["num_learning_epochs"] * self.cfg["algorithm"]["num_mini_batches"],
                mini_batch_size,
            )
            for n in range(self.cfg["algorithm"]["num_learning_epochs"]):
                for i in range(self.cfg["algorithm"]["num_mini_batches"]):
                    start = i * mini_batch_size
                    end = (i + 1) * mini_batch_size
                    batch_idx = indices[start:end]
                    dist, privileged_obs_est = self.model.act(
                        self.buffer["obses"].flatten(0, 1)[batch_idx], self.buffer["stacked_obses"].flatten(0, 1)[batch_idx]
                    )

                    values = self.model.est_values(self.buffer["critic_obses"].flatten(0, 1)[batch_idx])
                    value_loss = F.mse_loss(values, returns.flatten(0, 1)[batch_idx])

                    actions_log_prob = dist.log_prob(self.buffer["actions"].flatten(0, 1)[batch_idx]).sum(dim=-1)
                    actor_loss = surrogate_loss(
                        old_actions_log_prob.flatten(0, 1)[batch_idx],
                        actions_log_prob,
                        (advantages.flatten(0, 1)[batch_idx] * advantage_weights).sum(dim=-1),
                    )

                    policy_state, policy_next_state = next(amp_policy_generator)
                    expert_state, expert_next_state = next(amp_expert_generator)
                    with torch.no_grad():
                        self.amp_normalizer.update(torch.cat((policy_state, policy_next_state, expert_state, expert_next_state), dim=0))
                        policy_state = self._normalize_amp(policy_state)
                        policy_next_state = self._normalize_amp(policy_next_state)
                        expert_state = self._normalize_amp(expert_state)
                        expert_next_state = self._normalize_amp(expert_next_state)
                    policy_d = self.discriminator(torch.cat([policy_state, policy_next_state], dim=-1))
                    expert_d = self.discriminator(torch.cat([expert_state, expert_next_state], dim=-1))
                    amp_loss = -torch.tanh(0.4 * expert_d).mean() + torch.tanh(0.4 * policy_d).mean()

                    expert_data = torch.cat([expert_state, expert_next_state], dim=-1)
                    policy_data = torch.cat([policy_state, policy_next_state], dim=-1)
                    alpha = torch.rand(expert_data.size(0), 1, device=self.device)
                    interpolates = alpha * expert_data + (1 - alpha) * policy_data
                    interpolates.requires_grad = True
                    interpolates_d = self.discriminator(interpolates)
                    grad = torch.autograd.grad(
                        outputs=interpolates_d,
                        inputs=interpolates,
                        grad_outputs=torch.ones(interpolates_d.size(), device=self.device),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True,
                    )[0]
                    grad_pen_loss = 10 * ((torch.sqrt(torch.sum(grad.view(grad.size(0), -1) ** 2, dim=1) + 1e-12) - 1) ** 2).mean()

                    bound_loss = (
                        torch.clip(dist.loc - self.cfg["normalization"]["clip_actions"], min=0.0).square().mean()
                        + torch.clip(dist.loc + self.cfg["normalization"]["clip_actions"], max=0.0).square().mean()
                    )

                    entropy = dist.entropy().sum(dim=-1).mean()

                    reconstruction_loss = F.mse_loss(privileged_obs_est, self.buffer["privileged_obses"].flatten(0, 1)[batch_idx])

                    mirrored_dist, _ = self.model.act(
                        self.buffer["mirrored_obses"].flatten(0, 1)[batch_idx], self.buffer["mirrored_stacked_obses"].flatten(0, 1)[batch_idx]
                    )
                    symmetric_loss = F.mse_loss(dist.loc, self.env.mirror_act(mirrored_dist.loc))

                    loss = (
                        value_loss
                        + actor_loss
                        + self.cfg["algorithm"]["bound_coef"] * bound_loss
                        + self.cfg["algorithm"]["entropy_coef"] * entropy
                        + self.cfg["algorithm"]["reconstruction_coef"] * reconstruction_loss
                        + self.cfg["algorithm"]["amp_coef"] * amp_loss
                        + self.cfg["algorithm"]["grad_pen_coef"] * grad_pen_loss
                        + self.cfg["algorithm"]["symmetric_coef"] * symmetric_loss
                    )
                    self.optimizer.zero_grad()
                    self.discriminator_optimizer.zero_grad()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at iteration {it + 1}")
                    (loss / self.world_size).backward()
                    if self.world_size > 1:
                        for p in self.model.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
                        for p in self.discriminator.parameters():
                            if p.grad is not None:
                                torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.discriminator_optimizer.step()
                    for key in [
                        "value_loss",
                        "actor_loss",
                        "bound_loss",
                        "entropy",
                        "reconstruction_loss",
                        "amp_loss",
                        "grad_pen_loss",
                        "symmetric_loss",
                    ]:
                        statistics[key] = statistics.get(key, 0) + eval(key).item()

                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(dist.scale / old_dist.scale.flatten(0, 1)[batch_idx])
                            + 0.5
                            * (torch.square(old_dist.scale.flatten(0, 1)[batch_idx]) + torch.square(dist.loc - old_dist.loc.flatten(0, 1)[batch_idx]))
                            / torch.square(dist.scale)
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if self.world_size > 1:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.world_size
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                        statistics["policy_pred"] = statistics.get("policy_pred", 0) + policy_d.mean()
                        statistics["expert_pred"] = statistics.get("expert_pred", 0) + expert_d.mean()
                        statistics["kl_mean"] = statistics.get("kl_mean", 0) + kl_mean
                        statistics["lr"] = statistics.get("lr", 0) + self.learning_rate
            for key in statistics.keys():
                statistics[key] /= self.cfg["algorithm"]["num_learning_epochs"] * self.cfg["algorithm"]["num_mini_batches"]

            if self.rank == 0:
                if hasattr(self.env, 'batch'):
                    self.env.batch.check()
                if torch.device(self.device).type == 'cuda':
                    torch.cuda.synchronize(self.device)
                    statistics['performance/cuda_peak_allocated_gib'] = torch.cuda.max_memory_allocated(self.device) / 2**30
                    statistics['performance/cuda_reserved_gib'] = torch.cuda.memory_reserved(self.device) / 2**30
                elapsed = time.perf_counter() - iteration_start
                statistics['performance/env_steps_per_second'] = self.env.num_envs * self.cfg['runner']['horizon_length'] / elapsed
                statistics['performance/rollout_seconds'] = rollout_seconds
                statistics['performance/update_seconds'] = elapsed - rollout_seconds
                statistics['stability/active_over_5s'] = (self.env.episode_length_buf * self.env.dt >= 5).float().mean().item()
                statistics['stability/reset_cohort_survival_5s'] = getattr(self.env, 'survival_rate_5s', 0.)
                statistics['stability/base_height_mean'] = self.env.base_pos[:, 2].mean().item()
                statistics['curriculum/amp_reward_scale'] = getattr(self.env, 'amp_reward_scale', 1.)
                statistics['curriculum/balance_phase'] = float(getattr(self.env, 'balance_phase', False))
                statistics['curriculum/reference_init_probability'] = getattr(self.env, 'reference_init_probability', 1.)
                if hasattr(self.env, 'near_ball_stall'):
                    statistics['soccer/foot_ball_gap_mean'] = self.env.foot_ball_gap.mean().item()
                    statistics['soccer/near_ball_fraction'] = self.env.near_ball_active.float().mean().item()
                    statistics['soccer/near_ball_stall_fraction'] = (self.env.near_ball_stall > 0).float().mean().item()
                    statistics['soccer/far_ball_stall_fraction'] = (self.env.far_ball_stall > 0).float().mean().item()
                    statistics['soccer/amp_multiplier_mean'] = self.env.amp_reward_multiplier.mean().item()
                    statistics['curriculum/near_start_fraction'] = self.env.near_start_fraction
                    statistics['curriculum/near_start_progress'] = self.env.near_start_progress
                    statistics['soccer/near_start_success_rate'] = self.env.near_start_success_rate
                    statistics['soccer/full_start_success_rate'] = self.env.full_start_success_rate
                    statistics['soccer/near_start_completed_trials'] = self.env.near_start_trials
                self.recorder.collect_evaluation()
                self.recorder.record_statistics(statistics, it)
                if self.stop_requested or (it + 1) % self.cfg["runner"]["save_interval"] == 0 or it + 1 == self.cfg["basic"]["max_iterations"]:
                    self.recorder.save(self._checkpoint(it + 1), it + 1)
                    interval = self.cfg['runner'].get('evaluation_interval', 0)
                    if self.cfg['basic']['task'] == 'Q1' and interval and (it + 1) % interval == 0:
                        self.recorder.start_evaluation(os.path.join(self.recorder.model_dir, f'model_{it + 1}.pth'), it + 1)
                print("epoch: {}/{} | {:.0f} env steps/s | actor {:.4f} | value {:.4f}".format(
                    it + 1, self.cfg["basic"]["max_iterations"],
                    statistics['performance/env_steps_per_second'], statistics['actor_loss'], statistics['value_loss']), flush=True)
            if self.stop_requested:
                break
        if self.rank == 0:
            self.recorder.close()

    def play(self):
        obs, infos = self.env.reset()
        privileged_obs = infos["privileged_obs"]
        stacked_obs = infos["stacked_obs"]
        if self.cfg["viewer"]["record_video"]:
            os.makedirs("videos", exist_ok=True)
            name = time.strftime("%Y-%m-%d-%H-%M-%S.mp4", time.localtime())
            record_time = self.cfg["viewer"]["record_interval"]
        while True:
            with torch.no_grad():
                act = self.model(obs, stacked_obs)
                obs, rew, done, infos = self.env.step(act)
                privileged_obs = infos["privileged_obs"]
                stacked_obs = infos["stacked_obs"]
            if self.cfg["viewer"]["record_video"]:
                record_time -= self.env.dt
                if record_time < 0:
                    record_time += self.cfg["viewer"]["record_interval"]
                    self.interrupt = False
                    signal.signal(signal.SIGINT, self.interrupt_handler)
                    with imageio.get_writer(os.path.join("videos", name), fps=int(1.0 / self.env.dt)) as writer:
                        for frame in self.env.camera_frames:
                            writer.append_data(frame)
                    if self.interrupt:
                        raise KeyboardInterrupt
                    signal.signal(signal.SIGINT, signal.default_int_handler)

    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True
