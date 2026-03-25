import random
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Sequence

import numpy as np

from DNAnet.data.data_models.structs import Prediction


class Image(ABC):
    """
    Abstract base class for an image, that holds data, an annotation and meta information.
    """

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """
        The raw data content of the image.
        """
        raise NotImplementedError




class InMemoryDataset(Sequence[Image]):
    """
    A base class for a dataset holding images. The dataset should at least be
    iterable and splittable.
    """
    def __init__(self, shuffle: Optional[bool] = False):
        self.shuffle = shuffle
        self._data = []

    def __len__(self) -> int:
        """
        The number of images that the dataset holds.
        """
        return len(self._data)

    def __iter__(self) -> Iterator[Image]:
        """
        An iterator method to allow iteration over the dataset and apply shuffling if desired.
        """
        if self.shuffle:
            yield from random.Random().sample(self._data, len(self._data))
        else:
            yield from self._data

    def __getitem__(self, index: int) -> Image:
        return self._data[index]


class SimpleDataset(InMemoryDataset):
    """
    A simple dataset class that can handle a sequence of images directly, without relying on
    internal parsing first.
    """
    def __init__(self, data: Sequence[Image], shuffle: Optional[bool] = False):
        super().__init__(shuffle)
        self._data = data



class Metric:
    def __init__(self, func, name: str = None):
        self.func = func
        self.__name__ = name or func.__name__

    def __call__(self, images: Sequence[Image], predictions: Sequence[Prediction], **kwargs) -> float:
        return self.func(images, predictions, **kwargs)
