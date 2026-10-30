"""Equations (2) and (3), arXiv:2610.28480. Time is the first axis.

Rewards and termination probabilities belong to the transition arriving at t+1.
The terminal transition earns its reward; only subsequent rewards are masked.
"""
from contextlib import contextmanager

import torch


@contextmanager
def frozen(*modules):
    """Freeze parameters, preserving gradients through module inputs."""
    parameters = list(dict.fromkeys(p for m in modules for p in m.parameters()))
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def imagined_return(rewards, terminated, terminal_value, gamma=0.99):
    if rewards.shape != terminated.shape or rewards.shape[1:] != terminal_value.shape:
        raise ValueError("Expected rewards/done [time, batch], terminal value [batch]")
    survival = torch.ones_like(terminal_value)
    total = torch.zeros_like(terminal_value)
    discount = 1.0
    for reward, done in zip(rewards, terminated):
        total = total + discount * survival * reward
        survival = survival * (1 - done)
        discount *= gamma
    return total + discount * survival * terminal_value


@torch.no_grad()
def lambda_returns(rewards, terminated, values, gamma=0.99, lam=0.95):
    """Detached, fixed critic targets from the same imagined rollout."""
    if rewards.shape != terminated.shape or values.shape != (rewards.shape[0] + 1, *rewards.shape[1:]):
        raise ValueError("Values must contain H+1 states for H transitions")
    result = torch.empty_like(rewards)
    following = values[-1]
    for t in reversed(range(rewards.shape[0])):
        following = rewards[t] + gamma * (1 - terminated[t]) * (
            (1 - lam) * values[t + 1] + lam * following
        )
        result[t] = following
    return result


def clip_actor_gradients(parameters):
    parameters = list(parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.clamp_(-0.1, 0.1)
    return norm
