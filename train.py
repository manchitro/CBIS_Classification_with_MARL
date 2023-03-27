from os import mkdir
from os.patorch import exists, isdir, join
from typing import Tuple

import json
import mlflow
import torch
import tqdm

from dataset import CBISDataset
from agent import MultiAgent
from model import CBISClassifierModel
from metrics import ConfusionMeter, LossMeter
from observation import observation
from torch.utils.data import DataLoader


def train(
        mlflow_id: str,
        n_epochs: int,
        steps: int,
        cuda: bool,
        n_agents: int,
        belief_lstm_size: int,
        action_lstm_size: int,
        hidden_layer_size_belief: int,
        hidden_layer_size_action: int,
        state_size: int,
        message_size: int,
        window_size: int,
        step_size: int,
        epsilon: float,
        epsilon_decay: float,
        learning_rate: float,
        batch_size: int,
) -> None:
    resource_dir = "cbis"
    output_dir = "output"
    model_dir = "models"

    if not exists(join(output_dir, model_dir)):
        mkdir(join(output_dir, model_dir))
    if exists(join(output_dir, model_dir)) and not isdir(join(output_dir, model_dir)):
        raise Exception(f"\"{join(output_dir, model_dir)}\""
                        f"is not a directory.")

    mlflow.set_experiment("CBIS_MARL")
    mlflow.start_run(run_name=f"train_{mlflow_id}")

    mlflow.log_param("output_dir", output_dir)
    mlflow.log_param("model_dir", join(output_dir, model_dir))

    model = CBISClassifierModel(
        window_size,
        belief_lstm_size,
        action_lstm_size,
        message_size,
        state_size,
        step_size,
        hidden_layer_size_belief,
        hidden_layer_size_action,
        observation,

    )

    dataset_constructor = CBISDataset

    dataset = dataset_constructor(resource_dir)

    multi_agent = MultiAgent(
        n_agents,
        belief_lstm_size,
        action_lstm_size,
        window_size,
        message_size,
        step_size,
        batch_size,
        model,
        observation
    )

    mlflow.log_params({
        "window_size": window_size,
        "belief_lstm_size": belief_lstm_size,
        "action_lstm_size": action_lstm_size,
        "hidden_size_msg": message_size,
        "hidden_size_state": state_size,
        "step_size": step_size,
        "number_of_agents": n_agents,
        "epsilon": epsilon,
        "epsilon_decay": epsilon_decay,
        "number_of_epochs": n_epochs,
        "learning_rate": learning_rate,
        "steps": steps,
        "batch_size": batch_size,
    })

    json_f = open(join(output_dir, "class_to_idx.json"), "w")
    json.dump(dataset.class_to_idx, json_f)
    json_f.close()

    device_str = "cpu"

    if cuda:
        model.cuda()
        multi_agent.cuda()
        device_str = "cuda"

    mlflow.log_param("device", device_str)

    adam_optimizer = torch.optim.Adam(model.get_params(
        CBISClassifierModel.param_list), lr=learning_rate)

    train_dataset = dataset.dataset_mass_train
    test_dataset = dataset.dataset_mass_test

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
    )

    step = 0

    conf_meter_train = ConfusionMeter(
        2,
        window_size=64
    )

    patorch_loss_meter = LossMeter(window_size=64)
    reward_meter = LossMeter(window_size=64)
    loss_meter = LossMeter(window_size=64)

    for epoch in range(n_epochs):
        model.train()

        progress_bar = tqdm(train_dataloader)
        for x, y in progress_bar:
            x, y = x.to(torch.device(device_str)), \
                y.to(torch.device(device_str))

            predicitons, log_probabilities = exec_episode(
                multi_agent, x, epsilon, steps, device_str
            )


def exec_episode(multi_agent: MultiAgent, img_batch: torch.Tensor, epsilon: float, steps: int, device: str) -> Tuple(torch.Tensor, torch.Tensor):
    img_size = [size for size in img_batch.size()[2:]]
    batch_size = img_batch.size(0)

    multi_agent.init_episode(batch_size, img_size)

    img_batch = img_batch.to(torch.device(device))

    step_positions = torch.zeros(
        steps, *multi_agent.positions.size(), dtype=torch.long,
        device=torch.device(device)
    )

    step_predictions = torch.zeros(
        steps, batch_size, 2, device=torch.device(device)
    )

    step_probabilities = torch.zeros(
        steps, batch_size, device=torch.device(device)
    )

    for t in range(steps):
        multi_agent.step(img_batch, epsilon)

    return multi_agent.predict()

