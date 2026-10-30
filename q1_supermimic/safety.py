"""Fixed-seed real-physics safety floor and soccer-specific policy selection."""
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
                obs = agent.policy_observation(((state-mean)/scale).clamp(-20,20))
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


def soccer_candidate_is_safe(measured, baseline):
    # Fixed floor, independent of total reward and of previous score improvements.
    return (measured['survival_5s'] >= max(.65, baseline['survival_5s'])
            and measured['fall_fraction'] <= .35)


def survival_floor(rate, count):
    """Wilson 95% lower bound: a finite test cohort is not an exact probability.

    Used as a regression tolerance, not a confidence claim for adaptive reuse
    of the fixed evaluation seeds. The task's soccer floor remains separate.
    """
    z2 = 1.96**2
    return (rate + z2/(2*count) - 1.96*math.sqrt(rate*(1-rate)/count + z2/(4*count**2))) / (1+z2/count)


class SafetyGate:
    def __init__(self, agent, count=32, steps=300, noise_std=0.):
        self.count, self.steps = count, steps
        self.noise_std = noise_std
        self.noisy_baseline = evaluate_actor(agent, count=count, steps=steps, zero=True, noise_std=noise_std)
        self.noisy_best = dict(self.noisy_baseline)
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
        self.peak_survival = self.baseline['survival_5s']
        self.peak_noisy_survival = self.noisy_baseline['survival_5s']
        self.version = 5

    def check(self, agent, progress=0.):
        policy_changed = any(not torch.equal(value, self.accepted_actor[key])
                             for key, value in agent.actor.state_dict().items())
        soccer = progress > 0
        if soccer and self.progress == 0:
            self.baseline = evaluate_actor(agent, 1., self.count, self.steps, zero=True)
            self.noisy_baseline = evaluate_actor(agent, 1., self.count, self.steps, zero=True, noise_std=self.noise_std)
            self.noisy_best = dict(self.noisy_baseline)
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
        noisy = evaluate_actor(agent, 1. if soccer else 0., self.count, self.steps, noise_std=self.noise_std)
        if soccer:
            # Safety controls rollback; soccer score ranks checkpoints separately.
            # Millimetre-level approach fluctuations must not erase safe learning.
            accepted = soccer_candidate_is_safe(measured, self.baseline)
        else:
            accepted = measured['survival_5s'] >= survival_floor(self.peak_survival, self.count)
        accepted = accepted and noisy['survival_5s'] >= survival_floor(self.peak_noisy_survival, self.count)
        if accepted:
            self.peak_survival = max(self.peak_survival, measured['survival_5s'])
            self.peak_noisy_survival = max(self.peak_noisy_survival, noisy['survival_5s'])
            self.noisy_best = noisy
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
                'evaluation/exploratory_survival_5s': noisy['survival_5s'] if accepted else self.noisy_best['survival_5s'],
                'evaluation/policy_accepted': float(accepted and policy_changed),
                'evaluation/soccer_best_available': float(self.soccer_best_actor is not None),
                'evaluation/rollback_count': self.rejections,
                **{f'evaluation/neutral_{k}': v for k,v in self.baseline.items()}}

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        if state.get('version') == 5:
            self.__dict__.update(state)
        else:
            # Old acceptance scores were dominated by standing reward. Preserve
            # loaded policy/optimizers but re-evaluate with the new task criteria.
            self.rejections = state.get('rejections', 0)


@torch.no_grad()
def validate_action_response(agent, count=16, seed=130363):
    """Held-out paired real transitions; checks action sensitivity, not just MSE.

    Both branches replay the same seed and actions before +/- perturbations.
    These cases never enter the model-training replay buffer.
    """
    device = agent.mean.device
    generator = torch.Generator().manual_seed(seed+1)
    direction = torch.randn(count, 22, generator=generator)
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    predicted, actual, initials = [], [], []
    with torch.random.fork_rng(devices=[]):
        for sign in (-1., 1.):
            torch.random.default_generator.manual_seed(seed)
            wrapper = load_environment('cpu', count)
            state = wrapper.state()
            for _ in range(20):
                state, _, _, _ = wrapper.step(agent.act(state.to(device)).cpu())
            initial = state.to(device)
            action = (agent.act(initial).cpu() + sign*.05*direction).clamp(-1, 1)
            _, target, _, _ = wrapper.step(action)
            pred = agent.predict(agent.normalize(initial), action.to(device))[0]
            predicted.append(pred*agent.scale + agent.mean)
            actual.append(target.to(device))
            initials.append(initial)
        if not torch.equal(initials[0], initials[1]):
            raise RuntimeError('Counterfactual branches must start from identical states')
    a, b = agent.layout['physical_start'], agent.layout['physical_end']
    observed = (actual[1]-actual[0])[:, a:b] / agent.scale[a:b]
    estimated = (predicted[1]-predicted[0])[:, a:b] / agent.scale[a:b]
    denom = observed.square().sum().clamp_min(1e-12)
    relative = (estimated-observed).square().sum()/denom
    cosine = torch.nn.functional.cosine_similarity(estimated.flatten(), observed.flatten(), dim=0)
    # Root pose is small in aggregate norm but determines balance gradients.
    r = agent.layout.get('root_xy', a) - a
    root_observed, root_estimated = observed[:,r:r+7], estimated[:,r:r+7]
    root_cosine = torch.nn.functional.cosine_similarity(root_estimated.flatten(), root_observed.flatten(), dim=0)
    root_error = (root_estimated-root_observed).square().sum()/root_observed.square().sum().clamp_min(1e-12)
    return {'model_validation/action_response_cosine': cosine.item(),
            'model_validation/action_response_relative_error': relative.item(),
            'model_validation/root_response_cosine': root_cosine.item(),
            'model_validation/root_response_relative_error': root_error.item()}
