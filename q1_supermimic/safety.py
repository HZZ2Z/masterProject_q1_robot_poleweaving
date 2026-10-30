"""Paired real-physics safety floor and soccer-specific policy selection."""
import copy
import math
import torch
from .environment import load_environment


@torch.no_grad()
def evaluate_actor(agent, progress=0., count=32, steps=300, seed=104729, zero=False, noise_std=0.):
    actor = copy.deepcopy(agent.actor).cpu().eval()
    mean, scale = agent.mean.cpu(), agent.scale.cpu()
    limit = float(agent.action_limit)
    noise_rng = torch.Generator().manual_seed(seed+123)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        wrapper = load_environment('cpu', count)
        env = wrapper.env
        if progress > 0:
            # Fixed evaluation task across curriculum changes: neutral starts,
            # no motion-reference resets or external pushes. Same seed/quotas.
            env.balance_release_iteration = 0
            env.soccer_progress = .1
            env.set_training_iteration(0)
            env.reference_init_probability = 0.
            wrapper.reset()
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
                obs = agent.policy_observation(((state-mean)/scale).clamp(-20,20), mean=mean, scale=scale)
                action = limit * actor(obs).tanh()
            if noise_std:
                latent = torch.atanh((action/limit).clamp(-.999999, .999999))
                action = limit*(latent + noise_std/limit*torch.randn(action.shape, generator=noise_rng)).tanh()
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


def soccer_candidate_is_safe(measured, baseline, count=32):
    # Fixed floor, independent of total reward and of previous score improvements.
    return (measured['survival_5s'] >= max(.65, survival_floor(baseline['survival_5s'],count))
            and measured['fall_fraction'] <= .35)


def survival_floor(rate, count):
    """Wilson 95% lower bound: a finite test cohort is not an exact probability.

    Used as a regression tolerance, not a confidence claim for adaptive reuse
    of the fixed evaluation seeds. The task's soccer floor remains separate.
    """
    z2 = 1.96**2
    return (rate + z2/(2*count) - 1.96*math.sqrt(rate*(1-rate)/count + z2/(4*count**2))) / (1+z2/count)


class SafetyGate:
    def __init__(self, agent, count=32, steps=300, noise_std=0., seeds=None):
        self.count, self.steps, self.noise_std = count, steps, noise_std
        self.seeds = tuple(seeds or (104729,))
        self.checks = 0
        self.comparison = 'paired_real_cohorts'
        self.baseline = self._measure(agent, 0., self.seeds, zero=True)
        self.noisy_baseline = self._measure(agent, 0., self.seeds, zero=True, noise_std=noise_std)
        self.best, self.noisy_best = dict(self.baseline), dict(self.noisy_baseline)
        self.accepted_actor = copy.deepcopy(agent.actor.state_dict())
        self.accepted_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
        self.accepted_action_limit = float(agent.action_limit)
        self.soccer_best_actor = self.soccer_best_action_limit = self.soccer_best_metrics = None
        self.rejections = 0
        self.progress = 0.

    def _measure(self, agent, progress, seeds, **kwargs):
        results = [evaluate_actor(agent, progress, self.count, self.steps, seed=seed, **kwargs) for seed in seeds]
        return {key: (sum(row[key] for row in results) if key.endswith('sample_count')
                      else sum(row[key] for row in results)/len(results)) for key in results[0]}

    def check(self, agent, progress=0.):
        changed = (float(agent.action_limit) != self.accepted_action_limit or
                   any(not torch.equal(value,self.accepted_actor[key]) for key,value in agent.actor.state_dict().items()))
        self.checks += 1
        seeds = tuple(seed+self.checks*1000003 for seed in self.seeds)
        phase = 1. if progress > 0 else 0.
        n = self.count*len(seeds)
        if phase and self.progress == 0:
            self.baseline = self._measure(agent, phase, seeds, zero=True)
            self.noisy_baseline = self._measure(agent, phase, seeds, zero=True, noise_std=self.noise_std)
        self.progress = progress
        candidate_actor = copy.deepcopy(agent.actor.state_dict())
        candidate_limit = float(agent.action_limit)
        if changed:
            try:
                agent.actor.load_state_dict(self.accepted_actor)
                agent.action_limit.fill_(self.accepted_action_limit)
                incumbent = self._measure(agent, phase, seeds)
                incumbent_noisy = self._measure(agent, phase, seeds, noise_std=self.noise_std)
            finally:
                agent.actor.load_state_dict(candidate_actor)
                agent.action_limit.fill_(candidate_limit)
            candidate = self._measure(agent, phase, seeds)
            noisy = self._measure(agent, phase, seeds, noise_std=self.noise_std)
        else:
            candidate = incumbent = self._measure(agent, phase, seeds)
            noisy = incumbent_noisy = self._measure(agent, phase, seeds, noise_std=self.noise_std)
        if phase:
            safe = (soccer_candidate_is_safe(candidate,self.baseline,count=n)
                    and candidate['survival_5s'] >= survival_floor(incumbent['survival_5s'],n))
        else:
            floor = max(survival_floor(incumbent['survival_5s'],n),survival_floor(self.baseline['survival_5s'],n))
            safe = (candidate['survival_5s'] >= floor
                    and candidate['return'] >= incumbent['return']-.1
                    and noisy['return'] >= incumbent_noisy['return']-.1)
        safe = safe and noisy['survival_5s'] >= max(
            survival_floor(incumbent_noisy['survival_5s'],n),survival_floor(self.noisy_baseline['survival_5s'],n))
        if safe:
            measured, measured_noisy = candidate, noisy
            self.accepted_actor = copy.deepcopy(agent.actor.state_dict())
            self.accepted_optimizer = copy.deepcopy(agent.actor_optimizer.state_dict())
            self.accepted_action_limit = float(agent.action_limit)
            eligible = phase and candidate['near_kick_event_rate'] > 0 and candidate['near_net_goal_progress_m'] >= .01
            if eligible and (self.soccer_best_metrics is None or candidate['task_score'] > self.soccer_best_metrics['task_score']):
                self.soccer_best_actor = copy.deepcopy(self.accepted_actor)
                self.soccer_best_action_limit = self.accepted_action_limit
                self.soccer_best_metrics = dict(candidate)
        else:
            agent.actor.load_state_dict(self.accepted_actor)
            agent.actor_optimizer.load_state_dict(self.accepted_optimizer)
            agent.action_limit.fill_(self.accepted_action_limit)
            # Critic/model learning from collected failures must survive a rejected
            # actor proposal; otherwise the same mistaken proposal repeats forever.
            if changed:
                self.rejections += 1
            measured, measured_noisy = incumbent, incumbent_noisy
        self.best, self.noisy_best = dict(measured), dict(measured_noisy)
        return {**{f'evaluation/{key}':value for key,value in measured.items()},
                **{f'evaluation/candidate_{key}':value for key,value in candidate.items()},
                'evaluation/exploratory_survival_5s':measured_noisy['survival_5s'],
                'evaluation/policy_accepted':float(safe and changed),
                'evaluation/soccer_best_available':float(self.soccer_best_actor is not None),
                'evaluation/rollback_count':self.rejections,
                **{f'evaluation/neutral_{key}':value for key,value in self.baseline.items()}}

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        if state.get('comparison') == self.comparison:
            self.__dict__.update(state)
        else:
            self.rejections = state.get('rejections',0)


@torch.no_grad()
def sample_action_pairs(agent, count=32, seed=1000003, prefix_steps=20, amplitude=.05, progress=0.):
    """Paired native-physics transitions, preserving the training RNG.

    Training callers must use different seeds from held-out validation. Both
    actions are tested from exactly the same state and randomized physical model.
    """
    device = agent.mean.device
    actor = copy.deepcopy(agent.actor).cpu().eval()
    mean, scale = agent.mean.cpu(), agent.scale.cpu()
    direction = torch.randn(count, 22, generator=torch.Generator().manual_seed(seed+1))
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    targets, actions, initials = [], [], []
    def act(state):
        return float(agent.action_limit)*actor(agent.policy_observation(((state-mean)/scale).clamp(-20,20), mean=mean, scale=scale)).tanh()
    with torch.random.fork_rng(devices=[]):
        for sign in (-1., 1.):
            torch.random.default_generator.manual_seed(seed)
            wrapper = load_environment('cpu', count)
            if progress > 0:
                wrapper.env.balance_release_iteration = 0
                wrapper.env.soccer_progress = .1
                wrapper.env.set_training_iteration(0)
                wrapper.reset()
            state = wrapper.state()
            for _ in range(prefix_steps):
                state, _, _, _ = wrapper.step(act(state))
            action = (act(state)+sign*amplitude*direction).clamp(-1,1)
            _, target, _, _ = wrapper.step(action)
            initials.append(state)
            targets.append(target)
            actions.append(action)
    if not torch.equal(initials[0], initials[1]):
        raise RuntimeError('Counterfactual branches must start from identical states')
    return (initials[0].to(device), torch.stack(actions,1).to(device),
            torch.stack(targets,1).to(device))


@torch.no_grad()
def validate_action_response(agent, count=16, seed=130363):
    """Held-out physical and reward response; never added to training replay."""
    initial, actions, targets = sample_action_pairs(agent, count, seed)
    flat = initial[:,None].expand(-1,2,-1).flatten(0,1)
    pred = agent.predict(agent.normalize(flat), actions.flatten(0,1))[0]
    predicted = (pred*agent.scale+agent.mean).reshape_as(targets)
    a, b = agent.layout['physical_start'], agent.layout['physical_end']
    observed = (targets[:,1]-targets[:,0])[:,a:b]/agent.scale[a:b]
    estimated = (predicted[:,1]-predicted[:,0])[:,a:b]/agent.scale[a:b]
    def compare(real, model, prefix):
        return {prefix+'_cosine': torch.nn.functional.cosine_similarity(real.flatten(),model.flatten(),dim=0).item(),
                prefix+'_relative_error': ((model-real).square().sum()/real.square().sum().clamp_min(1e-12)).item()}
    result = compare(observed, estimated, 'model_validation/action_response')
    r = agent.layout.get('root_xy',a)-a
    result.update(compare(observed[:,r:r+7], estimated[:,r:r+7], 'model_validation/root_response'))
    if 'dt' in agent.layout:
        current = agent.normalize(flat)
        real_reward = agent.task_reward(current, agent.normalize(targets.flatten(0,1)), actions.flatten(0,1))[0].reshape(-1,2)
        model_reward = agent.task_reward(current, pred, actions.flatten(0,1))[0].reshape(-1,2)
        result.update(compare(real_reward[:,1]-real_reward[:,0], model_reward[:,1]-model_reward[:,0], 'model_validation/reward_response'))
    return result
