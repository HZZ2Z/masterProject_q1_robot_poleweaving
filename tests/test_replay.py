import pytest
import torch

from q1_supermimic.replay import Replay


def append(replay, t, boundary=False, terminated=False):
    replay.append(state=torch.full((2, 1), float(t)),
                  next_state=torch.full((2, 1), float(t + 1)),
                  action=torch.full((2, 1), float(t) + 0.125),
                  boundary=torch.full((2,), boundary),
                  terminated=torch.full((2,), terminated), context=torch.zeros(2, 1))


def test_wrapped_windows_are_chronological_and_keep_executed_action():
    replay = Replay(6)
    for t in range(15):
        append(replay, t)
    batch = replay.sample(100, length=4)
    assert batch["state"].min() >= 9
    assert batch["state"].max() <= 14
    torch.testing.assert_close(batch["state"][1:], batch["next_state"][:-1])
    torch.testing.assert_close(batch["action"], batch["state"] + 0.125)


@pytest.mark.parametrize("terminated", [False, True])
def test_timeout_and_fall_both_cut_windows_but_keep_terminal_label(terminated):
    replay = Replay(5)
    for t in range(5):
        append(replay, t, boundary=t == 2, terminated=terminated and t == 2)
    batch = replay.sample(20, length=3)
    assert (batch["state"][0] == 0).all()
    assert batch["boundary"][-1].all()
    assert (batch["terminated"][-1] == terminated).all()
    assert (batch["next_state"][-1] == 3).all()


def test_never_invent_windows_across_resets():
    replay = Replay(8)
    for t in range(8):
        append(replay, t, boundary=True)
    with pytest.raises(ValueError, match="No contiguous"):
        replay.sample(4, length=2)


def test_invalid_post_terminal_states_cannot_seed_any_window():
    replay = Replay(5)
    for t in range(5):
        replay.append(state=torch.full((1,1),float(t)),next_state=torch.full((1,1),float(t+1)),
                      action=torch.zeros(1,1),context=torch.zeros(1,0),
                      boundary=torch.tensor([t>=1]),terminated=torch.tensor([t>=1]),
                      valid=torch.tensor([t<=1]))
    batch = replay.sample(40,length=1)
    assert batch['state'].max() <= 1
    assert replay.sample(10,length=2)['state'][0].eq(0).all()
