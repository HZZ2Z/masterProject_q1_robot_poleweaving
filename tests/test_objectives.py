import pytest
import torch
from torch import nn

from q1_supermimic.objectives import frozen, imagined_return, lambda_returns
from q1_supermimic.networks import MotorPolicy


def test_terminal_transition_keeps_reward_but_cuts_bootstrap():
    rewards = torch.tensor([[2.], [3.], [100.]])
    done = torch.tensor([[0.], [1.], [0.]])
    result = imagined_return(rewards, done, torch.tensor([999.]), gamma=0.5)
    torch.testing.assert_close(result, torch.tensor([3.5]))
    targets = lambda_returns(rewards, done, torch.zeros(4, 1), gamma=0.5, lam=1.)
    torch.testing.assert_close(targets[:2, 0], torch.tensor([3.5, 3.]))


def test_soft_survival_gradient_and_lambda_targets():
    done = torch.tensor([[0.25], [0.5]], requires_grad=True)
    rewards = torch.tensor([[2.], [4.]], requires_grad=True)
    values = torch.tensor([[0.], [5.], [8.]], requires_grad=True)
    result = imagined_return(rewards, done, values[-1], gamma=0.5)
    torch.testing.assert_close(result, torch.tensor([4.25]))
    result.sum().backward()
    torch.testing.assert_close(done.grad[:, 0], torch.tensor([-3., -1.5]))
    targets = lambda_returns(rewards, done, values, gamma=0.5, lam=1.)
    torch.testing.assert_close(targets[0], result.detach())
    assert not targets.requires_grad


def test_frozen_model_keeps_input_jacobian_and_restores_on_error():
    model = nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(3.)
    x = torch.tensor([[2.]], requires_grad=True)
    with pytest.raises(RuntimeError):
        with frozen(model):
            model(x).sum().backward()
            assert model.weight.grad is None
            raise RuntimeError("aborted update")
    assert model.weight.requires_grad
    torch.testing.assert_close(x.grad, torch.tensor([[3.]]))


def test_soccer_gradient_trains_encoder_through_frozen_decoder():
    torch.manual_seed(7)
    policy = MotorPolicy(6, 8, 4, 2, motor_widths=(16,), visual_widths=(8,))
    before = {k: p.clone() for k, p in policy.named_parameters()}
    optimizer = torch.optim.Adam(policy.visual_encoder.parameters(), lr=1e-3)
    with frozen(policy.decoder):
        loss = policy.soccer(torch.randn(5, 6), torch.randn(5, 4)).square().mean()
        loss.backward()
        optimizer.step()
    assert all(p.grad is None for p in policy.decoder.parameters())
    assert all(p.grad is None for p in policy.reference_encoder.parameters())
    assert any(not torch.equal(p, before[k]) for k, p in policy.named_parameters()
               if k.startswith("visual_encoder"))
    assert all(torch.equal(p, before[k]) for k, p in policy.named_parameters()
               if not k.startswith("visual_encoder"))
