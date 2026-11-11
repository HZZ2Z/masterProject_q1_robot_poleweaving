# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Tsinghua University and ByteDance Ltd.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import torch


class ActorCritic(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.critics = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(self.cfg["env"]["num_critic_obs"], 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 1),
                ),
                torch.nn.Sequential(
                    torch.nn.Linear(self.cfg["env"]["num_critic_obs"], 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 256),
                    torch.nn.ELU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ELU(),
                    torch.nn.Linear(128, 1),
                ),
            ]
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(
                self.cfg["env"]["num_observations"] + self.cfg["algorithm"]["history_embedding_dim"],
                256,
            ),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.cfg["env"]["num_actions"]),
        )
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.cfg["env"]["num_observations"] * self.cfg["runner"]["num_stack"], 1024),
            torch.nn.ELU(),
            torch.nn.Linear(1024, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.cfg["algorithm"]["history_embedding_dim"]),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.cfg["algorithm"]["history_embedding_dim"], 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, self.cfg["env"]["num_privileged_obs"]),
        )
        self.logstd = torch.nn.Parameter(torch.full((self.cfg["env"]["num_actions"],), fill_value=-2.0))

    def act(self, obs, stacked_obs):
        embedding = self.encoder(stacked_obs.flatten(start_dim=-2))
        action_mean = self.actor(torch.cat((obs, embedding), dim=-1))
        action_std = torch.exp(self.logstd).expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        privileged_obs_est = self.decoder(embedding)
        return dist, privileged_obs_est

    def est_values(self, critic_obs):
        return torch.stack([critic(critic_obs).squeeze(-1) for critic in self.critics], dim=-1)

    def forward(self, obs, stacked_obs):
        embedding = self.encoder(stacked_obs.flatten(start_dim=-2))
        return self.actor(torch.cat((obs, embedding), dim=-1))
