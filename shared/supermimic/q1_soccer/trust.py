"""One candidate actor step with a measured action-space trust constraint."""

import copy
import torch


class PolicyTrust:
    def __init__(self, algorithm, rms_limit=0.01, max_limit=0.05):
        self.algorithm = algorithm
        self.rms_limit = rms_limit
        self.max_limit = max_limit
        self.retained = self.snapshot()
        self.accepted_updates = 0
        self.consecutive_rejections = 0

    def snapshot(self):
        return {
            name: copy.deepcopy(getattr(self.algorithm, name).state_dict())
            for name in ["actor", "actor_optimizer"]
        }

    def restore(self, snapshot):
        for name, state in snapshot.items():
            getattr(self.algorithm, name).load_state_dict(state)

    @torch.no_grad()
    def constrain(self, previous, history, previous_actions):
        """Backtrack actor weights only; optimizer momentum tracks the accepted path.

        Compare raw commands as well as executed commands. Saturated actions must
        not hide large changes that become visible later in the episode.
        """
        candidate = self.snapshot()
        for attempt in range(9):
            fraction = 0.5**attempt
            if attempt:
                interpolated = {
                    name: previous["actor"][name]
                    + fraction * (value - previous["actor"][name])
                    for name, value in candidate["actor"].items()
                }
                self.algorithm.actor.load_state_dict(interpolated)
            actions = self.algorithm.actor(history)
            delta = actions - previous_actions
            rms = float(delta.square().mean().sqrt())
            maximum = float(delta.abs().max())
            if rms <= self.rms_limit and maximum <= self.max_limit:
                return {
                    "trust_passed": True,
                    "action_delta_rms": rms,
                    "action_delta_max": maximum,
                    "step_fraction": fraction,
                }
        self.restore(previous)
        return {
            "trust_passed": False,
            "action_delta_rms": rms,
            "action_delta_max": maximum,
            "step_fraction": 0.0,
        }

    def accept(self):
        self.retained = self.snapshot()
        self.accepted_updates += 1
        self.consecutive_rejections = 0

    def reject(self):
        self.restore(self.retained)
        self.consecutive_rejections += 1
        # Critic and model keep their real-data learning; only the rejected
        # actor and its optimizer are rolled back.
        for group in self.algorithm.actor_optimizer.param_groups:
            group["lr"] = max(group["lr"] * 0.5, 1e-9)
        self.retained = self.snapshot()
