from typing import Any, Set, List
from os.path import exists, isfile
import torch as th
import torch.nn as nn
import json

from feature_extractor import CBISFeatureExtractor, AgentStateToFeatures
from messages import MessageSender
from lstm import LSTMCell
from policy import Policy
from prediction import Prediction


class CBISClassifierModel(nn.Module):

    module_observation_map: str = "feature_map"
    module_position_map: str = "state_map"
    module_message_eval: str = "msg_map"
    module_belief_unit: str = "belief_unit"
    module_action_unit: str = "action_unit"
    module_policy: str = "policy_map"
    module_prediction: str = "prediction_map"

    param_list: Set[str] = {
        module_observation_map,
        module_position_map,
        module_message_eval,
        module_belief_unit,
        module_action_unit,
        module_policy,
        module_prediction
    }

    def __init__(self, window_size: int,
                 hidden_belief: int,
                 hidden_action: int,
                 message_size: int,
                 state_size: int,
                 step_size: int,
                 hidden_layer_size_belief: int,
                 hidden_layer_size_action: int,
                 ) -> None:

        self.window_size = window_size
        self.hidden_belief = hidden_belief
        self.hidden_action = hidden_action
        self.message_size = message_size
        self.state_size = state_size
        self.step_size = step_size
        self.hidden_layer_size_action = hidden_layer_size_action
        self.hidden_layer_size_belief = hidden_layer_size_belief

        observation_module = CBISFeatureExtractor(window_size)

        self.network_dict = nn.ModuleDict({
            self.module_observation_map: observation_module,
            self.module_position_map: AgentStateToFeatures(2, state_size),
            self.module_message_eval: MessageSender(hidden_belief, message_size, hidden_layer_size_belief),
            self.module_belief_unit: LSTMCell(observation_module.out_size + state_size + message_size, hidden_belief),
            self.module_action_unit: LSTMCell(observation_module.out_size + state_size + message_size, hidden_action),
            self.module_policy: Policy(step_size, hidden_action, hidden_layer_size_action),
            self.module_prediction: Prediction(
                hidden_belief, 2, hidden_layer_size_belief)
        })

    def forward(self, module: str, *args):
        return self.network_dict[module](*args)

    @property
    def n_class(self) -> int:
        return 2

    @property
    def window_size(self) -> int:
        return self.window_size

    def get_params(self, module_names: List[str]) -> List[th.Tensor]:
        return [
            name for module in module_names
            for name in self.network_dict[module].parameters()
        ]

    def json_args(self, out_json_path: str) -> None:
        json_f = open(out_json_path, "w")

        args_d = {
            "window_size": self.window_size,
            "hidden_belief": self.hidden_belief,
            "hidden_action": self.hidden_action,
            "message_size": self.message_size,
            "state_size": self.state_size,
            "step_size": self.step_size,
            "hidden_layer_size_action": self.hidden_layer_size_action,
            "hidden_layer_size_belief": self.hidden_layer_size_belief,
        }

        json.dump(args_d, json_f)

        json_f.close()

    # TODO: test this method
    @classmethod
    def from_json(cls, json_path):
        if not exists(json_path):
            raise ValueError(f"JSON file not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        try:
            window_size = data["window_size"]
            hidden_belief = data["hidden_belief"]
            hidden_action = data["hidden_action"]
            message_size = data["message_size"]
            state_size = data["state_size"]
            step_size = data["step_size"]
            hidden_layer_size_action = data["hidden_layer_size_action"]
            hidden_layer_size_belief = data["hidden_layer_size_belief"]
        except KeyError as e:
            raise ValueError(f"Missing required key in JSON file: {str(e)}")

        return cls(window_size, hidden_belief, hidden_action, message_size, state_size,
                   step_size, hidden_layer_size_action, hidden_layer_size_belief)
