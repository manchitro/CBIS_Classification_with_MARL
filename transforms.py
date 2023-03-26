from abc import ABCMeta, abstractmethod
import torch


class ImgTransform(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, img_data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            self.__class__.__name__ +
            ".__call__ method is not implemented, must be overridden !"
        )

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Normalization(ImgTransform):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - torch.mean(x)) / torch.std(x)
