"""Goal interfaces share a motor decoder, never a privileged actor input."""
import torch
from torch import nn


def mlp(input_dim, output_dim, widths):
    layers = []
    for width in widths:
        layers.extend((nn.Linear(input_dim, width), nn.ELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class MotorPolicy(nn.Module):
    """Separate reference and visual encoders; bounded 32-dimensional latent.

    The caller supplies sensor history, never physical state, to these methods.
    Reference targets are available to the training-only tracking interface.
    Visual goal features must be constructed by the PPO-compatible camera.
    """

    def __init__(self, proprio_dim, reference_dim, visual_dim, action_dim,
                 motor_widths=(2048, 1024, 512, 256), visual_widths=(256, 256)):
        super().__init__()
        self.reference_encoder = mlp(reference_dim, 32, motor_widths)
        self.visual_encoder = mlp(visual_dim, 32, visual_widths)
        self.decoder = mlp(proprio_dim + 32, action_dim, motor_widths)

    def track(self, proprio_history, reference):
        latent = self.reference_encoder(reference).tanh()
        return self.decoder(torch.cat((proprio_history, latent), dim=-1))

    def soccer(self, proprio_history, visual_history):
        latent = self.visual_encoder(visual_history).tanh()
        return self.decoder(torch.cat((proprio_history, latent), dim=-1))
