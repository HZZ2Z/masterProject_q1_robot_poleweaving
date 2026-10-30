"""Reference-interface analytic actor/critic update through frozen dynamics.

This is the motor pretraining update, not the visual soccer training entry.
Simulator collection and independent real-physics evaluation remain external.
"""
import torch
from .objectives import frozen, imagined_return, lambda_returns, clip_actor_gradients


def tracking_update(policy, world, critic, head, observations, actor_optimizer,
                    critic_optimizer, state, history, references, context,
                    horizon=32, future=10, gamma=.99, lam=.95, critic_epochs=5):
    if horizon < 1 or future < 1 or references.shape[0] < horizon + future + 1:
        raise ValueError('Need reference frames for the entire rollout and terminal goal')
    if critic_epochs < 1:
        raise ValueError('At least one critic epoch is required')
    if history.shape[-1] != observations.dimension:
        raise ValueError('Invalid motor observation history')
    state, history = state.detach(), history.detach()
    previous_action = history[:, -1, -22:]
    references, context = references.detach(), context.detach()
    rewards, terminations, critic_inputs, values = [], [], [], []
    actor_optimizer.zero_grad(set_to_none=True)
    # Clearing old model/critic gradients makes isolation inspectable; neither
    # optimizer is stepped until its own phase.
    world.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    with frozen(world, critic):
        for t in range(horizon + 1):
            goal = observations.reference(references[t+1:t+future+1]).movedim(0, 1).flatten(1)
            critic_input = torch.cat((state, goal, context), -1)
            critic_inputs.append(critic_input.detach())
            values.append(critic(critic_input).squeeze(-1))
            if t == horizon:
                break
            action = policy.track(history.flatten(1), goal)
            following = world(state, action, context)
            reward, _, probability = head(following, references[t+1], action, previous_action)
            rewards.append(reward)
            terminations.append(probability)
            sensor = observations(following, action)
            history = torch.cat((history[:, 1:], sensor.unsqueeze(1)), 1)
            state, previous_action = following, action
        rewards = torch.stack(rewards)
        terminations = torch.stack(terminations)
        values = torch.stack(values)
        objective = imagined_return(rewards, terminations, values[-1], gamma).mean()
        if not torch.isfinite(objective):
            raise FloatingPointError('Non-finite imagined return')
        (-objective).backward()
        grad_norm = clip_actor_gradients(policy.parameters())
    actor_optimizer.step()
    targets = lambda_returns(rewards.detach(), terminations.detach(), values.detach(), gamma, lam)
    inputs = torch.stack(critic_inputs[:-1]).flatten(0, 1)
    targets = targets.flatten()
    for _ in range(critic_epochs):
        critic_optimizer.zero_grad(set_to_none=True)
        value_loss = (critic(inputs).squeeze(-1) - targets).square().mean()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
        critic_optimizer.step()
    return {'imagined_return': objective.detach().item(), 'actor_grad_norm': grad_norm.item(),
            'critic_loss': value_loss.detach().item()}
