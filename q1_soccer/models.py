"""Soccer state and vision adapters around the manufacturer's network classes."""

import copy
import torch
from torch import nn

from expert.policy import ActorCritic
from supermimic.models import MotionPolicy, WorldModel, Critic, mlp
from supermimic.robot import normalize_quat


class VisionPolicy(MotionPolicy):
    """Teacher-initialized causal 79 x 50 policy, with no reference/true ball input."""

    def __init__(self, cfg, checkpoint=None):
        nn.Module.__init__(self)
        teacher = ActorCritic(cfg)
        if checkpoint is not None:
            teacher.load_state_dict(
                torch.load(checkpoint, map_location="cpu", weights_only=True)["model"],
                strict=True,
            )
        self.encoder = teacher.encoder
        self.decoder = teacher.actor

    def encode_goal(self, goal):
        # The task goal is already a sensor feature; future reference is privileged.
        return goal.new_zeros((len(goal), 0))

    def decode(self, history, encoding=None):
        latent = self.encoder(history.flatten(1))
        return self.decoder(torch.cat((history[:, 0], latent), -1))

    def forward(self, history, goal=None):
        return self.decode(history)


class SoccerWorldModel(WorldModel):
    def __init__(self, robot, hidden=(512, 512, 512)):
        # Use the vendor network initialization and extend its physical output
        # to include the ball. Neither camera queues nor reference motion drive physics.
        super().__init__(robot, hidden=hidden)
        self.register_buffer(
            "scale",
            torch.cat(
                (
                    self.scale,
                    torch.tensor(
                        [1.0] * 7 + [3.0] * 6 + [1.0] * 7 + [3.0] * 3 + [1.0] * 7
                    ),
                )
            ),
        )
        self.dynamics = mlp(109 + 22, hidden, 87)
        self.termination = mlp(109, (128, 64), 1)
        nn.init.normal_(self.dynamics[-1].weight, std=0.001)
        nn.init.zeros_(self.dynamics[-1].bias)
        nn.init.constant_(self.termination[-1].bias, -4.0)
        self.register_buffer("neutral", robot.neutral.clone())
        self.register_buffer("action_scale", robot.action_scale.clone())
        self.register_buffer("limits", robot.joint_limits.clone())
        self.clip_actions = robot.cfg["normalization"]["clip_actions"]

    def forward(self, state, action, context):
        # Equivalent effective target makes target clipping explicit to model
        # gradients; stored previous action remains the normalized raw command.
        executed = action.clamp(-self.clip_actions, self.clip_actions)
        target = (self.neutral + self.action_scale * executed).clamp(
            self.limits[:, 0], self.limits[:, 1]
        )
        effective = (target - self.neutral) / self.action_scale
        residual = self.dynamics(torch.cat((state / self.scale, effective), -1))
        robot = state[:, :57] + 0.02 * self.scale[:57] * residual[:, :57]
        ball = state[:, 79:92] + 0.02 * self.scale[79:92] * residual[:, 57:70]
        sensors = state[:, 92:109] + 0.02 * self.scale[92:109] * residual[:, 70:]
        return torch.cat(
            (
                robot[:, :3],
                normalize_quat(robot[:, 3:7]),
                robot[:, 7:],
                executed,
                ball[:, :3],
                normalize_quat(ball[:, 3:7]),
                ball[:, 7:],
                sensors[:, :3],
                normalize_quat(sensors[:, 3:7]),
                sensors[:, 7:13],
                normalize_quat(sensors[:, 13:17]),
            ),
            -1,
        )

    def termination_logits(self, robot, state, target):
        return self.termination(state / self.scale).squeeze(-1)

    def advance_context(self, context, executed_action):
        # Physical dynamics do not depend on sensor queues. Imagination advances
        # the actual queues through SoccerRobot.observe, which also knows state.
        return context


class SoccerCritic(Critic):
    def __init__(self, robot):
        super().__init__(robot)
        self.register_buffer(
            "scale",
            torch.cat(
                (
                    self.scale,
                    torch.tensor(
                        [1.0] * 7 + [3.0] * 6 + [1.0] * 7 + [3.0] * 3 + [1.0] * 7
                    ),
                )
            ),
        )


class RuntimePolicy(nn.Module):
    """Native expert environment's (observation, history) calling convention."""

    def __init__(self, policy):
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu().eval()

    def forward(self, observation, history):
        return self.policy(history)
