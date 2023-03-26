import json
from typing import Callable, Tuple, List

from model import CBISClassifierModel

import torch


class MultiAgent:
    def __init__(
            self, n_agents: int,
            belief_lstm_size: int,
            action_lstm_size: int,
            window_size: int,
            message_size: int,
            step_size: int,
            batch_size: int,
            model: CBISClassifierModel,
            observations: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
            transformations: Callable[[torch.Tensor,
                                       torch.Tensor, int, List[int]], torch.Tensor]
    ) -> None:

        self.n_agents = n_agents
        self.window_size = window_size
        self.batch_size = batch_size

        self.belief_lstm_size = belief_lstm_size
        self.action_lstm_size = action_lstm_size

        self.msg_size = message_size

        self.positions = None
        self.t = 0

        self.observation = observations
        self.model = model

        self.hidden_state_belief = None
        self.cell_state_belief = None

        self.hidden_state_action = None
        self.cell_state_action = None

        self.msg = None

        self.action_probabilities = None

        self.cuda = False
        self.device = "cpu"

    def init_episode(self, batch_size, img_size: List[int]) -> None:
        # hidden state of belief lstm
        self.hidden_state_belief = [
            torch.zeros(self.n_agents, batch_size, self.belief_lstm_size,
                        device=torch.device(self.device))
        ]
        # cell belief lstm
        self.cell_state_belief = [
            torch.zeros(self.n_agents, batch_size, self.belief_lstm_size,
                        device=torch.device(self.device))
        ]

        # hidden action lstm
        self.hidden_state_action = [
            torch.zeros(self.n_agents, batch_size, self.action_lstm_size,
                        device=torch.device(self.device))
        ]
        # cell action lstm
        self.cell_state_action = [
            torch.zeros(self.n_agents, batch_size, self.action_lstm_size,
                        device=torch.device(self.device))
        ]

        self.msg = [
            torch.zeros(self.n_agents, batch_size, self.msg_size,
                        device=torch.device(self.device))
        ]

        self.action_probabilities = [
            torch.ones(self.n_agents, batch_size,
                       device=torch.device(self.device))
            / 4  # can move in 4 directions
        ]

        self.positions = torch.stack([
            torch.randint(i_s - self.window_size, (self.n_agents,
                          batch_size), device=torch.device(self.device))
            for i_s in img_size], dim=-1)

    def step(self, img: torch.Tensor, epsilon: float) -> None:
        img_sizes = [s for s in img.size()[2:]]
        n_agents = self.n_agents

        observation_t = self.observation(img, self.positions, self.window_size)
        features_t = self.model(
            self.model.module_observation_map, observation_t.flatten(0, 1)).view(n_agents, self.batch_size, -1)

        current_msg = self.msg[self.t]
        current_msg_mean = current_msg.mean(dim=0)
        aggregate_msg = (
            (current_msg_mean * n_agents - current_msg) / (n_agents - 1)
        )

    @ property
    def is_cuda(self) -> bool:
        return self.cuda

    def cuda(self) -> None:
        self.cuda = True
        self.device = "cuda"

    def cpu(self) -> None:
        self.cuda = False
        self.device = "cpu"

    @ property
    def positions(self) -> torch.Tensor:
        return self.positions

    def __len__(self) -> int:
        return self.n_agents
