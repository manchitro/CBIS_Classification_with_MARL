import torch
import torch.nn as nn

from util import Permute


class CBISFeatureExtractor(nn.Module):
    # TODO: build a better model
    def __init__(self, window_size: int) -> None:
        super().__init__()

        self.conv_sequential = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),

            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (window_size // 8) ** 2

        for module in self.conv_sequential:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.conv_sequential(obs)

    @property
    def out_size(self) -> int:
        return self.__out_size


class AgentStateToFeatures(nn.Module):
    def __init__(self, dimensions: int, state_size: int) -> None:
        super().__init__()

        self.dimensions = dimensions
        self.state_size = state_size

        self.linear_sequential = nn.Sequential(
            nn.Linear(in_features=self.dimensions,
                      out_features=self.state_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(self.state_size),
            Permute([2, 0, 1]),
        )

        for module in self.linear_sequential:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, position_t: torch.Tensor) -> torch.Tensor:
        return self.linear_sequential(position_t)


################################################################
        # from https://jovian.ml/aakashns/05-cifar10-cnn

cifar10 = nn.Sequential(
    # in: 1 x window_size x window_size (1 x 32 x 32 default)
    nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

    nn.Flatten(1, -1)
)


default = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),

    nn.Flatten(1, -1)
)
