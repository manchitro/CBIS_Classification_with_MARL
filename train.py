from os import mkdir
from os.path import exists, isdir, join
from typing import Tuple

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as Func
import json
import mlflow
import torch
from tqdm import tqdm

from dataset import CBISDataset
from agent import MultiAgent
from model import CBISClassifierModel
from metrics import ConfusionMeter, LossMeter
from observation import observation
from transition import transition


def train(
    dataset_to_train: str,
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

    if not exists(output_dir):
        mkdir(output_dir)
    if exists(output_dir) and not isdir(output_dir):
        raise Exception(f"\"{output_dir}\""
                        f"is not a directory.")
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
    )

    dataset_constructor = CBISDataset

    dataset = dataset_constructor(resource_dir, dataset_to_train)

    multi_agent = MultiAgent(
        n_agents,
        belief_lstm_size,
        action_lstm_size,
        window_size,
        message_size,
        step_size,
        batch_size,
        model,
        observation,
        transition
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

    train_test_ratio = 0.80
    index = torch.tensor(list(range(0, len(dataset)-1)))
    index_train = index[:int(train_test_ratio * index.size()[0])]
    index_test = index[int(train_test_ratio * index.size()[0]):]
    print("index:", index)
    print("idx_train", index_train)
    print("idx_test", index_test)

    # # save augmented images
    # aug_img_dir = "aug_imgs"
    # if not exists(join(output_dir, aug_img_dir)):
    #     mkdir(join(output_dir, aug_img_dir))
    # if exists(join(output_dir, aug_img_dir)) and not isdir(join(output_dir, model_dir)):
    #     raise Exception(f"\"{join(output_dir, aug_img_dir)}\""
    #                     f"is not a directory.")
    # offset = 4
    # for i, index in enumerate(dataset.augments_indices):
    #     print(i, index)
    #     img, label = dataset.__getitem__(
    #         (index[0] if i == 0 else index[0]-1) + offset)
    #     fig = plt.figure()
    #     plt.imshow(img.permute(1, 2, 0), cmap="gray")
    #     plt.title(dataset.augments[i])
    #     plt.savefig(join(output_dir, aug_img_dir,
    #                 f"img_"+dataset.augments[i]+".png"))
    #     plt.close(fig)

    train_dataset = Subset(dataset, index_train)
    print('train length: ', len(train_dataset))
    test_dataset = Subset(dataset, index_test)
    print('test length: ', len(test_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=True, num_workers=6, drop_last=False, pin_memory=True
    )

    i = 0

    conf_meter_train = ConfusionMeter(
        2,
        window_size=64
    )

    path_loss_meter = LossMeter(window_size=64)
    reward_meter = LossMeter(window_size=64)
    loss_meter = LossMeter(window_size=64)

    for epoch in range(n_epochs):
        model.train()

        progress_bar = tqdm(train_dataloader)
        for x, y in progress_bar:
            x, y = x.to(torch.device(device_str)), \
                y.to(torch.device(device_str))

            predictions, action_probabilities, step_positions = train_episode(
                multi_agent, x, epsilon, steps, device_str
            )

            last_prediction = predictions[-1, :, :]

            reward = -Func.cross_entropy(
                last_prediction, y,
                reduction="none"
            )
            path_sum = action_probabilities.sum(dim=0)
            path_loss = path_sum.exp() * reward.detach()

            loss = -(path_loss + reward).mean()

            adam_optimizer.zero_grad()

            loss.backward()

            adam_optimizer.step()

            conf_meter_train.add(last_prediction.detach(), y)

            path_loss_meter.add(path_sum.mean().item())
            reward_meter.add(reward.mean().item())
            loss_meter.add(loss.item())

            precision, recall = (
                conf_meter_train.precision(),
                conf_meter_train.recall()
            )

            if i % 100 == 0:
                mlflow.log_metrics(step=i, metrics={
                    "reward": reward.mean().item(),
                    "path_loss": path_sum.mean().item(),
                    "loss": loss.item(),
                    "train_prec": precision.mean().item(),
                    "train_rec": recall.mean().item(),
                    "epsilon": epsilon
                })

            progress_bar.set_description(
                f"Epoch {epoch} - Train, "
                f"precision = {precision.mean().item():.3f}, "
                f"recall = {recall.mean().item():.3f}, "
                f"loss = {loss_meter.loss():.4f}, "
                f"reward = {reward_meter.loss():.4f}, "
                f"path = {path_loss_meter.loss():.4f}, "
                f"eps = {epsilon:.4f}"
            )

            epsilon *= epsilon_decay
            epsilon = max(epsilon, 0.)

            i += 1

        model.eval()
        conf_meter_eval = ConfusionMeter(2, None)

        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)
            for x_test, y_test in progress_bar:
                x_test, y_test = x_test.to(torch.device(device_str)), \
                    y_test.to(torch.device(device_str))

                predictions, _ = eval_episode(multi_agent, x_test, 0., steps)

                conf_meter_eval.add(predictions.detach(), y_test)

                precision, recall = (
                    conf_meter_eval.precision(),
                    conf_meter_eval.recall()
                )

                progress_bar.set_description(
                    f"Epoch {epoch} - Eval, "
                    f"precision = {precision.mean().item():.4f}, "
                    f"recall = {recall.mean().item():.4f}"
                )

        precision, recall = (
            conf_meter_eval.precision(),
            conf_meter_eval.recall()
        )

        conf_meter_eval.save_conf_matrix(epoch, output_dir, "eval")

        mlflow.log_metrics(step=i, metrics={
            "eval_precision": precision.mean().item(),
            "eval_recall": recall.mean().item()
        })

        model.json_args(
            join(output_dir,
                 model_dir,
                 f"marl_epoch_{epoch}.json")
        )
        torch.save(
            model.state_dict(),
            join(output_dir, model_dir,
                 f"nn_models_epoch_{epoch}.pt")
        )


def train_episode(multi_agent: MultiAgent, img_batch: torch.Tensor, epsilon: float, steps: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    img_size = [size for size in img_batch.size()[2:]]
    batch_size = img_batch.size(0)

    multi_agent.init_episode(batch_size, img_size)

    img_batch = img_batch.to(torch.device(device))

	# step position = (for each step, for each agent, for each image in batch, 2 position values x and y)
    step_positions = torch.zeros(
        steps, *multi_agent.positions.size(), dtype=torch.long,
        device=torch.device(device)
    )

	# step predictions = (for each step, for each image in batch, 2 values: probalibility of benign and malignant)
    step_predictions = torch.zeros(
        steps, batch_size, 2, device=torch.device(device)
    )

	# step probabilities = (for each step, for each image in batch, one value for the probability of taking next step?)
    step_probabilities = torch.zeros(
        steps, batch_size, device=torch.device(device)
    )

    for t in range(steps):
        multi_agent.step(img_batch, epsilon)

        step_positions[t, :, :] = multi_agent.positions

        predictions, probabilities = multi_agent.predict()

        step_predictions[t, :, :] = predictions
        step_probabilities[t, :] = probabilities

    return step_predictions, step_probabilities, step_positions


def eval_episode(
        agents: MultiAgent,
        img_batch: torch.Tensor,
        epsilon: float,
        steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    img_sizes = [s for s in img_batch.size()[2:]]
    agents.init_episode(img_batch.size(0), img_sizes)

    for t in range(steps):
        agents.step(img_batch, epsilon)

    return agents.predict()
