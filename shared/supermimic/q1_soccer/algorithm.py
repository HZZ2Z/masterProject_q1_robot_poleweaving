"""Manufacturer SuperMimic with ball-state loss and causal vision imagination."""

import torch
import torch.nn.functional as F

from supermimic.algorithm import SuperMimic, gae_returns
from supermimic.models import frozen, goal_features


class SoccerSuperMimic(SuperMimic):
    """Keep vendor autoregressive fitting, optimizers, clipping and critic update.

    Override only task-dependent state loss and imagination: the stock task has
    no ball and uses a 10-frame oldest-first proprioceptive history.
    """

    def _model_losses(self, state, logits, label, terminated):
        pieces = []
        for section, quat in [
            (slice(0, 57), slice(3, 7)),
            (slice(79, 92), slice(82, 86)),
            (slice(92, 102), slice(95, 99)),
            (slice(102, 109), slice(105, 109)),
        ]:
            aligned = label.clone()
            aligned[:, quat] = torch.where(
                (state[:, quat] * label[:, quat]).sum(-1, keepdim=True) < 0,
                -label[:, quat],
                label[:, quat],
            )
            pieces.append(
                ((state[:, section] - aligned[:, section]) / self.model.scale[section])
                .square()
                .sum(-1)
            )
        return torch.stack(pieces, -1).sum(
            -1
        ).mean(), F.binary_cross_entropy_with_logits(logits, terminated)

    def improve_policy(
        self,
        batch,
        critic_epochs=5,
        critic_minibatches=4,
        *,
        anchor_batch=None,
        anchor_weight=1.0,
        bootstrap_weight=1.0,
        fit_critic=True,
    ):
        state = batch.states[:, 0].detach()
        history = batch.history.detach()
        context = batch.context.detach()
        references = self.motions.sequence(
            batch.clips,
            batch.frames,
            torch.arange(self.horizon + 11, device=state.device),
        )
        remaining = self.motions.remaining(batch.clips, batch.frames)
        survival = state.new_ones(len(state))
        objective = state.new_zeros(len(state))
        states = []
        goals = []
        contexts = []
        weights = []
        rewards = []
        terminations = []
        values = []
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.model_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        with frozen(self.model, self.critic):
            for t in range(self.horizon):
                goal = goal_features(self.robot, references[:, t + 1 : t + 11])
                states.append(state.detach())
                goals.append(goal.detach())
                contexts.append(context.detach())
                weights.append(survival.detach())
                value = self.critic(state, goal, context)
                values.append(value)
                action = self.actor(history)
                next_state = self.model(state, action, context)
                reward = self.robot.reward(
                    state, next_state, references[:, t + 1], action, 0.02
                )
                terminal = self.model.termination_logits(
                    self.robot, next_state, references[:, t + 1]
                ).sigmoid()
                terminal = torch.where(
                    remaining <= t + 1, torch.ones_like(terminal), terminal
                )
                objective = objective + self.gamma**t * survival * reward
                survival = survival * (1 - terminal)
                rewards.append(reward)
                terminations.append(terminal)
                observation, context = self.robot.observe(next_state, context)
                history = torch.cat((observation[:, None], history[:, :-1]), 1)
                state = next_state
            final_goal = goal_features(
                self.robot, references[:, self.horizon + 1 : self.horizon + 11]
            )
            values.append(self.critic(state, final_goal, context))
            objective = (
                objective
                + bootstrap_weight * self.gamma**self.horizon * survival * values[-1]
            )
            loss = -objective.mean() / self.horizon
            anchor = loss.new_zeros(())
            if anchor_batch is not None:
                # Only strict successful expert episodes enter this batch.
                predicted = self.actor(anchor_batch.history)
                anchor = F.mse_loss(predicted.clamp(-1, 1), anchor_batch.actions[:, 0])
                loss = loss + anchor_weight * anchor
            self.distributed.require_finite(loss, "Nonfinite soccer imagined return")
            loss.backward()
        norm = self._update_actor()
        returns = gae_returns(
            torch.stack(rewards, 1),
            torch.stack(terminations, 1),
            torch.stack(values, 1),
        )
        critic_loss = (
            self._fit_critic(
                torch.stack(states, 1).flatten(0, 1),
                torch.stack(goals, 1).flatten(0, 1),
                torch.stack(contexts, 1).flatten(0, 1),
                returns.flatten(),
                torch.stack(weights, 1).flatten(),
                critic_epochs,
                critic_minibatches,
            )
            if fit_critic
            else None
        )
        result = dict(
            actor_loss=float(loss.detach()),
            anchor_loss=float(anchor.detach()),
            actor_grad_norm=float(norm),
            imagined_reward=float(torch.stack(rewards).mean().detach()),
        )
        if critic_loss is not None:
            result["critic_loss"] = float(critic_loss)
        return result

    def fit_real_critic(self, batch):
        """Fit finite-episode returns from actual MuJoCo trajectories.

        No learned-model prediction or untrained critic bootstraps these labels.
        Critic training continues even when an actor candidate is rolled back.
        """
        targets = []
        for clip, frame in zip(batch.clips.cpu(), batch.frames.cpu()):
            episode = self.motions.episodes[int(clip)]
            if "real_returns" not in episode:
                states = episode["states"]
                with torch.no_grad():
                    rewards = self.robot.reward(
                        states[:-1].to(batch.states.device),
                        states[1:].to(batch.states.device),
                        states[1:].to(batch.states.device),
                        episode["actions"].to(batch.states.device),
                        0.02,
                    ).cpu()
                returns = torch.zeros_like(rewards)
                future = torch.tensor(0.0)
                for t in range(len(rewards) - 1, -1, -1):
                    future = rewards[t] + self.gamma * future
                    returns[t] = future
                episode["real_returns"] = returns
            targets.append(episode["real_returns"][int(frame)])
        labels = torch.stack(targets).to(batch.states.device)
        references = self.motions.sequence(
            batch.clips, batch.frames, torch.arange(1, 11, device=batch.states.device)
        )
        goal = goal_features(self.robot, references)
        loss = self._fit_critic(
            batch.states[:, 0],
            goal,
            batch.context,
            labels,
            torch.ones_like(labels),
            5,
            4,
        )
        return {
            "critic_real_loss": float(loss),
            "critic_real_target_abs_mean": float(labels.abs().mean()),
            "critic_relative_rmse": float(
                loss.sqrt() / labels.square().mean().sqrt().clamp_min(1e-6)
            ),
        }
