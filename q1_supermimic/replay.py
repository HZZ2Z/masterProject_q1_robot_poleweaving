"""Bounded temporal replay. Reset boundaries never become dynamics labels."""
import torch


class Replay:
    """Stores explicit pre-reset next states and executed (noisy) actions.

    All fields have leading [environment] dimension. `boundary` includes falls,
    time limits and externally reset episodes; `terminated` excludes time limits.
    Keeping both is essential: timeout cuts a sequence, but is not a fall label.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.cursor = self.size = 0
        self.storage = {}

    @torch.no_grad()
    def append(self, **fields):
        required = {"state", "next_state", "action", "boundary", "terminated", "context"}
        if not required.issubset(fields):
            raise ValueError(f"Missing replay fields: {required - fields.keys()}")
        n = fields["state"].shape[0]
        if any(v.shape[0] != n for v in fields.values()):
            raise ValueError("Replay fields must share their environment axis")
        if fields["boundary"].shape != (n,) or fields["terminated"].shape != (n,):
            raise ValueError("boundary and terminated must be scalar per environment")
        if not self.storage:
            self.storage = {k: torch.empty((self.capacity, *v.shape), dtype=v.dtype, device=v.device)
                            for k, v in fields.items()}
        if fields.keys() != self.storage.keys():
            raise ValueError("Replay schema changed")
        for key, value in fields.items():
            if value.shape != self.storage[key].shape[1:]:
                raise ValueError(f"Replay shape changed for {key}")
        for key, value in fields.items():
            self.storage[key][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, length=8, generator=None):
        if count < 1 or length < 1 or self.size < length:
            raise ValueError("Insufficient replay or invalid sample shape")
        device = self.storage["state"].device
        oldest = self.cursor if self.size == self.capacity else 0
        starts = torch.arange(self.size - length + 1, device=device)
        offsets = torch.arange(length, device=device)
        indices = (oldest + starts[:, None] + offsets) % self.capacity
        # A boundary on the LAST transition is valid: its terminal next state
        # is supervised. A boundary earlier in the window breaks continuity.
        allowed = ~self.storage["boundary"][indices[:, :-1]].bool().any(dim=1)
        if "valid" in self.storage:
            allowed = allowed & self.storage["valid"][indices].bool().all(dim=1)
        candidates = allowed.nonzero()
        if not len(candidates):
            raise ValueError("No contiguous replay windows; collect more transitions")
        selected = candidates[torch.randint(len(candidates), (count,), device=device, generator=generator)]
        time = indices[selected[:, 0]].T
        env = selected[:, 1].unsqueeze(0).expand(length, -1)
        return {key: value[time, env] for key, value in self.storage.items()}
