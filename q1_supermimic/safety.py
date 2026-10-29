"""Fixed-seed real-physics safety floor and soccer-specific policy selection."""
import copy
import torch
from .environment import load_environment


@torch.no_grad()
def evaluate_actor(agent, progress=0., count=32, steps=300, seed=104729, zero=False):
    actor = copy.deepcopy(agent.actor).cpu().eval()
    mean, scale = agent.mean.cpu(), agent.scale.cpu()
    limit = float(agent.action_limit)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        wrapper = load_environment('cpu', count)
        env = wrapper.env
        if progress > 0:
            # Fixed evaluation task across curriculum changes: neutral starts,
            # no motion-reference resets or external pushes. Same seed/quotas.
            env.balance_release_iteration = 0
            env.soccer_progress = .1
            env.set_training_iteration(0)
            env.reference_init_probability = 0.
            env.reset()
        state = wrapper.state()
        near = env.reset_group == 2
        initial_gap = env._foot_ball_gap().clone()
        min_gap = initial_gap.clone()
        initial_goal = (env.ball_pos[:, :2] - torch.tensor([7., 0.])).norm(dim=-1)
        final_goal = initial_goal.clone()
        kicked = torch.zeros(count, dtype=torch.bool)
        scored = torch.zeros(count, dtype=torch.bool)
        alive = torch.ones(count, dtype=torch.bool)
        durations = torch.zeros(count)
        returns = torch.zeros(count)
        failures = torch.zeros(count, dtype=torch.bool)
        for _ in range(steps):
            if zero:
                action = torch.zeros(count, 22)
            else:
                obs = ((state-mean)/scale).clamp(-20,20)[:, :agent.obs_dim]
                action = limit * actor(obs).tanh()
            state, _, reward, info = wrapper.step(action)
            feedback = info['task_feedback']  # pre-reset snapshots, including terminal frames
            valid = alive & ~info['fall']
            min_gap = torch.where(valid, torch.minimum(min_gap, feedback['foot_ball_gap']), min_gap)
            final_goal = torch.where(alive, feedback['goal_distance'], final_goal)
            kicked |= valid & feedback['kick_event']
            scored |= alive & info['success'].bool()
            durations += alive.float() * env.dt
            returns += alive.float() * reward
            failures |= alive & info['fall']
            durations[alive & info['success'].bool()] = steps * env.dt
            alive &= ~info['done']
        def near_mean(value):
            return value[near].float().mean().item() if near.any() else 0.
        approach = near_mean((initial_gap-min_gap).clamp(0, .25))
        forward = near_mean((initial_goal-final_goal).clamp(-2, 2))
        kick_rate = near_mean(kicked)
        success = near_mean(scored)
        return {'duration_s': durations.mean().item(), 'return': returns.mean().item(),
                'survival_5s': (durations >= 4.99).float().mean().item(),
                'horizon_s': steps * env.dt,
                'fall_fraction': failures.float().mean().item(),
                'near_approach_m': approach, 'near_kick_event_rate': kick_rate,
                'near_net_goal_progress_m': forward, 'near_success_rate': success,
                'near_sample_count': int(near.sum()),
                'task_score': approach + .5 * kick_rate + 2 * forward + 5 * success}


def soccer_candidate_is_safe(measured, baseline):
    # Fixed floor, independent of total reward and of previous score improvements.
    return (measured['survival_5s'] >= max(.65, baseline['survival_5s'])
            and measured['fall_fraction'] <= .35)


class SafetyGate:
    def __init__(self, agent, count=32, steps=300):
        self.count, self.steps = count, steps
        self.baseline = evaluate_actor(agent, count=count, steps=steps, zero=True)
        self.best = dict(self.baseline)
        self.accepted_actor = copy.deepcopy(agent.actor.state_dict())
        self.accepted_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
        self.accepted_action_limit = float(agent.action_limit)
        self.accepted_critic = copy.deepcopy(agent.critic.state_dict())
        self.accepted_critic_optimizer = copy.deepcopy(agent.critic_optimizer.state_dict())
        self.soccer_best_actor = None
        self.soccer_best_action_limit = None
        self.soccer_best_metrics = None
        self.rejections = 0
        self.progress = 0.
        self.version = 3

    def check(self, agent, progress=0.):
        soccer = progress > 0
        if soccer and self.progress == 0:
            self.baseline = evaluate_actor(agent, 1., self.count, self.steps, zero=True)
            # Compare against the deployed incumbent on this fixed soccer task.
            candidate_actor = copy.deepcopy(agent.actor.state_dict())
            candidate_limit = float(agent.action_limit)
            try:
                agent.actor.load_state_dict(self.accepted_actor)
                agent.action_limit.fill_(self.accepted_action_limit)
                self.best = evaluate_actor(agent, 1., self.count, self.steps)
            finally:
                agent.actor.load_state_dict(candidate_actor)
                agent.action_limit.fill_(candidate_limit)
        self.progress = progress
        measured = evaluate_actor(agent, 1. if soccer else 0., self.count, self.steps)
        candidate = dict(measured)
        if soccer:
            # Safety controls rollback; soccer score ranks checkpoints separately.
            # Millimetre-level approach fluctuations must not erase safe learning.
            accepted = soccer_candidate_is_safe(measured, self.baseline)
        else:
            accepted = (measured['duration_s'] >= max(self.baseline['duration_s'], self.best['duration_s']) - .02
                        and measured['return'] >= max(self.baseline['return'], self.best['return']) - .1)
        if accepted:
            self.best = measured
            self.accepted_actor = copy.deepcopy(agent.actor.state_dict())
            self.accepted_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
            self.accepted_action_limit = float(agent.action_limit)
            self.accepted_critic = copy.deepcopy(agent.critic.state_dict())
            self.accepted_critic_optimizer = copy.deepcopy(agent.critic_optimizer.state_dict())
            eligible = (soccer and measured['near_kick_event_rate'] > 0
                        and measured['near_net_goal_progress_m'] >= .01)
            if eligible and (self.soccer_best_metrics is None or measured['task_score'] > self.soccer_best_metrics['task_score']):
                self.soccer_best_actor = copy.deepcopy(self.accepted_actor)
                self.soccer_best_action_limit = self.accepted_action_limit
                self.soccer_best_metrics = dict(measured)
        else:
            agent.actor.load_state_dict(self.accepted_actor)
            agent.actor_optimizer.load_state_dict(self.accepted_optimizer)
            agent.action_limit.fill_(self.accepted_action_limit)
            agent.critic.load_state_dict(self.accepted_critic)
            agent.critic_optimizer.load_state_dict(self.accepted_critic_optimizer)
            self.rejections += 1
            measured = evaluate_actor(agent, 1. if soccer else 0., self.count, self.steps)
        return {**{f'evaluation/{k}': v for k,v in measured.items()},
                **{f'evaluation/candidate_{k}': v for k,v in candidate.items()},
                'evaluation/policy_accepted': float(accepted),
                'evaluation/soccer_best_available': float(self.soccer_best_actor is not None),
                'evaluation/rollback_count': self.rejections,
                **{f'evaluation/neutral_{k}': v for k,v in self.baseline.items()}}

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        if state.get('version') == 3:
            self.__dict__.update(state)
        else:
            # Old acceptance scores were dominated by standing reward. Preserve
            # loaded policy/optimizers but re-evaluate with the new task criteria.
            self.rejections = state.get('rejections', 0)
