import numpy as np
import torch
from q1_4v4.env import SoccerEnv
from q1_4v4.learning import make_args, make_team, save_team, load_team


def test_role_gradients_and_checkpoint_roundtrip():
    torch.set_num_threads(2)
    env = SoccerEnv()
    args = make_args(num_envs=1, horizon=8, epochs=1)
    p, t, b = make_team(args, env, torch.device('cpu'))
    assert b.action_log_probs.shape == (8, 1, 4, 1)
    obs = env.observe()[:4]
    states = np.zeros((4, 1, args.hidden_size), np.float32)
    masks = np.ones((4, 1), np.float32)
    actions, _ = p.act(obs, states, masks, deterministic=True)
    loss = actions.square().mean()
    loss.backward()
    grads = [x.grad for x in p.actor.base.role_encoder.parameters()]
    assert all(x is not None and torch.isfinite(x).all() for x in grads)
    assert sum(x.abs().sum() for x in grads) > 0
    p2, t2, _ = make_team(args, env, torch.device('cpu'))
    load_team(save_team(p, t), p2, t2)
    actions2, _ = p2.act(obs, states, masks, deterministic=True)
    assert torch.equal(actions, actions2)
