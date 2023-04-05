import json
from typing import Callable, Tuple, List, Any

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
            transition: Callable[[torch.Tensor,
                                       torch.Tensor, int, List[int]], torch.Tensor]
    ) -> None:

        self.n_agents = n_agents
        self.window_size = window_size
        self.batch_size = batch_size
        self.actions = [[step_size, 0], [-step_size, 0],
                        [0, step_size], [0, -step_size]]

        self.belief_lstm_size = belief_lstm_size
        self.action_lstm_size = action_lstm_size

        self.msg_size = message_size

        self.__positions = None
        self.t = 0

        self.observation = observations
        self.transition = transition
        self.model = model

        self.hidden_state_belief = None
        self.cell_state_belief = None

        self.hidden_state_action = None
        self.cell_state_action = None

        self.msg = None

        self.action_probabilities = None

        self.__cuda = False
        self.device = "cpu"

    def init_episode(self, batch_size, img_size: List[int]) -> None:
        self.t = 0
        self.batch_size = batch_size
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

        self.__positions = torch.stack([
            torch.randint(i_s - self.window_size, (self.n_agents,
                          batch_size), device=torch.device(self.device))
            for i_s in img_size], dim=-1)

    def step(self, img_batch: torch.Tensor, epsilon: float) -> None:
        img_sizes = [s for s in img_batch.size()[2:]]
        n_agents = self.n_agents

        # observation at t (for each agent, for each image in batch, for each channel in image,
        # a (window_size x window_size) matrix representing agent's partial observation of the image)
        observation_t = self.observation(
            img_batch, self.__positions, self.window_size)

        # agent and batch size of the observation is flattened and passed to the feature extraction module
        # features are retrieved and reshaped appropriately
        features_t = self.model(
            self.model.module_feature_map, observation_t.flatten(0, 1)
        ).view(n_agents, self.batch_size, -1)

        # self.msg is list of messages at each time step
        # msg shape (for each agent for each img in batch, a vector of size msg_size)
        current_msg = self.msg[self.t]

        # taking mean of msgs over all agents (dim=0 is n_agent)
        current_msg_mean = current_msg.mean(dim=0)

        # each agent takes the mean of all messages EXCEPT its own as the aggregated message
        aggregate_msg = (
            (current_msg_mean * n_agents - current_msg) / (n_agents - 1)
        )

        # print("self.positions shape", self.__positions.shape)
        # print("[img_sizes]", [img_sizes])
        # print("[[img_sizes]]", [[img_sizes]])
        # print("tensor shape [img_sizes]", torch.tensor([img_sizes]).shape)
        # print("tensor shape [[img_sizes]]", torch.tensor([[img_sizes]]).shape)

        # each position is divided by the size of the image to normalize the position tensor with values between 0 and 1
        normalized_positions = self.__positions.to(
            torch.float) / torch.tensor([[img_sizes]], device=torch.device(self.device))

        # information about the agents positional state is calcualted
        # module_position_map turns 2 dimenstional positions into tensor.
        # shape: for each agents, for each image in batch, a vector of size state_size
        spatial_state = self.model(
            self.model.module_position_map, normalized_positions)

        # agent's information unit consists of
        # 1. extracted features
        # 2. received messages
        # 3. spatial state
        # concatenated along the first 2 dimensions of agents and img batch
        information_unit_t = torch.cat(
            (features_t, aggregate_msg, spatial_state), dim=2)

        # updating the hidden and cell state of belief LSTM for next step
        # by passing current hidden and cell state and the information unit
        next_hidden_state_belief, next_cell_state_belief = self.model(
            self.model.module_belief_unit,
            self.hidden_state_belief[self.t],
            self.cell_state_belief[self.t],
            information_unit_t
        )

        self.hidden_state_belief.append(next_hidden_state_belief)
        self.cell_state_belief.append(next_cell_state_belief)

        # doing the same for action LSTM
        next_hidden_state_action, next_cell_state_action = self.model(
            self.model.module_action_unit, self.hidden_state_action[self.t], self.cell_state_action[self.t], information_unit_t)

        self.hidden_state_action.append(next_hidden_state_action)
        self.cell_state_action.append(next_cell_state_action)

        # inserting into the msg list,
        # the evaluated message for the next step
        # calculated using the recently calculated hidden_state_belief
        self.msg.append(
            self.model(
                self.model.module_message_eval,
                self.hidden_state_belief[self.t + 1]
            )
        )

        # action probability function pi(action, hidden_state_action) -> probability of that particular action
        action_probabilites = self.model(
            self.model.module_action_policy,
            self.hidden_state_action[self.t + 1],
        )

        actions = torch.tensor(
            self.actions,
            device=torch.device(self.device)
        )

        # agent has 4 possible actions (movement in each of the 4 directions)
        # we select the actions for each agent for each img in batch with the most probability
        _, most_probable_actions = action_probabilites.max(dim=-1)

        # generating random action for each agent for each img in batch
        random_actions = torch.randint(
            0, len(self.actions),
            (self.n_agents, self.batch_size),
            device=torch.device(self.device)
        )

        # generating random number for each agent and for each img in batch
        # then, comparing with epsilon, creates a boolean matrix
        # if true, then that agent on that image will use greedy approach
        # if false, then that agent on that image will use random approach
        use_greedy = torch.gt(
            torch.rand(
                (self.n_agents, self.batch_size),
                device=torch.device(self.device)
            ),
            epsilon
        ).to(torch.int)

        # choosing final action selected from either
        # random action or the most probable action selected by the policy
        # depending on the use_greedy boolean matrix
        final_actions = (
            use_greedy * most_probable_actions +
            (1 - use_greedy) * random_actions
        )

        # selecting which action to choose from available actions
        # each action a 2-element vector (movement in x and y)
        next_actions_t = actions[final_actions]

        self.action_probabilities.append(
            action_probabilites
            .gather(-1, final_actions.unsqueeze(-1))
            .squeeze(-1)
        )

        # update agent position using the selected actions
        self.__positions = self.transition(
            self.positions.to(torch.float),
            next_actions_t, self.window_size,
            img_sizes
        ).to(torch.long)

        self.t += 1

    def predict(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.model(
                self.model.module_prediction,
                self.hidden_state_belief[-1]
            ).mean(dim=0),
            self.action_probabilities[-1].log().sum(dim=0)
        )

    @ property
    def is_cuda(self) -> bool:
        return self.__cuda

    def cuda(self) -> None:
        self.__cuda = True
        self.device = "cuda"

    def cpu(self) -> None:
        self.__cuda = False
        self.device = "cpu"

    @ property
    def positions(self) -> torch.Tensor:
        return self.__positions

    def __len__(self) -> int:
        return self.n_agents
