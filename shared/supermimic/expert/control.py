"""Explicit bounded controller variant; the archived expert runtime stays intact."""
import numpy as np

from .runtime import SoccerPlayer


class BoundedSoccerPlayer(SoccerPlayer):
    control_contract = 'joint_target_and_torque_limits_v1'

    def step(self, auto_reset=True, on_episode_end=None, on_control=None, on_physics=None):
        def bounded_control(obs, history, raw):
            joints = self.model.actuator_trnid[:, 0]
            limits = self.model.jnt_range[joints]
            self.targets[:] = np.clip(self.targets, limits[:, 0], limits[:, 1])
            if on_control is not None:
                on_control(obs, history, raw)
        return super().step(auto_reset, on_episode_end, bounded_control, on_physics)
