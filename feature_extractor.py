import torch
import torch.nn as nn


class CBISFeatureExtractor(nn.Module):
    # TODO: build a better model
    def __init__(self, window_size: int) -> None:
        super().__init__()

        self.__seq_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(1, -1)
        )

        self.__out_size = 32 * (window_size // 4) ** 2

        for m in self.__seq_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, o_t: torch.Tensor) -> torch.Tensor:
        o_t = o_t[:, 0, None, :, :]  # grey scale
        return self.__seq_conv(o_t)

    @property
    def out_size(self) -> int:
        return self.__out_size


class AgentStateToFeatures(nn.Module):
    def __init__(self, dimensions: int, state_size: int) -> None:
        super().__init__()

        self.dimensions = dimensions
        self.state_size = state_size

        self.__seq_lin = nn.Sequential(
            nn.Linear(in_features=self.dimensions,
                      out_features=self.state_size),
            nn.ReLU()
        )

    def forward(self, p_t: torch.Tensor) -> torch.Tensor:
        return self.__seq_lin(p_t)
