import torch
import numpy as np
import scipy.stats
from torch import Tensor
from typing import Tuple, Callable


def tgt_tokenise_fn(target_type: str = 'onehot') -> Callable:
    if target_type == 'onehot':
        return _one_hot
    elif target_type == 'multihot':
        return _multi_hot
    else:
        raise ValueError(f"Cannot recognise {target_type}")

def _one_hot(target: list, labelset: list) -> Tensor:
    r"""A simple way to generate one-hot label for few-shot algorithms.
    Returns: A tensor with shape = (len(target), len(labelset)).
    e.g., a = _one_hot(target=['a', 'b'], labelset=['a', 'b', 'c'])  # [[1, 0, 0], [0, 1, 0]]
    """
    # Create dict: target -> class_id
    _tokeniser = dict()
    for idx, l in enumerate(labelset):
        _tokeniser[l] = idx
    one_hot = torch.zeros(len(target), len(labelset))
    for idx, tgt in enumerate(target):
        one_hot[idx][_tokeniser[tgt]] = 1
    return one_hot

def _multi_hot(target: list, labelset: list, sep: str = ', ') -> Tensor:
    r"""A simple way to generate one-hot label for few-shot algorithms.
    Returns: A tensor with shape = (len(target), len(labelset)).
    e.g., a = _multi_hot(target=['a', 'a,b'], labelset=['a', 'b', 'c'])  # [[1, 0, 0], [1, 1, 0]]
    """
    # Create dict: target -> class_id
    _tokeniser = dict()
    for idx, l in enumerate(labelset):
        _tokeniser[l] = idx
    multi_hot = torch.zeros(len(target), len(labelset))
    for idx, tgt in enumerate(target):
        if isinstance(tgt, str):
            tgt = tgt.split(sep)
            
        for t in tgt:
            multi_hot[idx][_tokeniser[t]] = 1
    return multi_hot

class CustomDistance():
    r""" A collection of distance functions."""
    def __init__(self, type='l2'):
        if type == 'l2':
            self.distance_fn = self.l2_distance
        elif type == 'cosine':
            self.distance_fn = self.cosine_distance
        elif type == 'dot':
            self.distance_fn = self.dot
        else:
            raise KeyError("metric should be selected from `l2_distance`, `cosine`, `dot`")

    def __call__(self, x: Tensor, y: Tensor, **kargs) -> Tensor:
        return self.distance_fn(x, y, **kargs)

    def l2_distance(self, x: Tensor, y: Tensor, square: bool = True) -> Tensor:
        r"""Calculate l2 distance between x and y, following 'sub-power^2-sum.'
        Args:    
            x, size = (n_x, n_features)
            y, size = (n_y, n_features)
            square, bool. default is True to follow the setting in 'Prototypical Networks for Few-shot Learning, Snell et al.'
        Return: torch.tensor, size = (n_x, n_y)
        """
        n_x = x.size(dim=0)
        n_y = y.size(dim=0)

        sub = x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)
        if square:
            return sub.pow(2).sum(dim=2)
        else:
            return sub.pow(2).sum(dim=2).sqrt()

    def cosine_distance(self, x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
        r"""Calculate cosine distance of `x` and `y`.
        Args: x, size = (n_x, n_features)
              y, size = (n_y, n_features)
              eps, float, very small number.
        """
        n_x = x.size(dim=0)
        n_y = y.size(dim=0)

        norm_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)
        norm_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)

        cosine_similarity = (norm_x.unsqueeze(1).expand(n_x, n_y, -1) * norm_y.unsqueeze(0).expand(n_x, n_y, -1)).sum(dim=2)
        return 1 - cosine_similarity

    def dot(self, x: Tensor, y: Tensor) -> Tensor:
        """ Calculate dot distance of `x` and `y`.
        Args: x, size = (n_x, n_features)
              y, size = (n_y, n_features)
        """
        n_x = x.size(dim=0)
        n_y = y.size(dim=0)
        return -(x.unsqueeze(1).expand(n_x, n_y, -1) * y.unsqueeze(0).expand(n_x, n_y, -1)).sum(dim=2)

def confidence_interval(x: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    n_sample = x.shape[0]
    mean, std = x.mean(), x.std()
    dof = n_sample - 1  # degree of freedom
    t = np.abs(scipy.stats.t.ppf((1 - confidence) / 2, dof))
    interval = t * std / np.sqrt(n_sample)
    return mean, interval

def normc2d(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Normalise each colunm of `x` with l2-norm."""
    norm_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)
    return norm_x


if __name__ == "__main__":
    x = np.asarray(
        [0.8846222758293152, 0.9564666152000427, 0.907200038433075, 0.978244423866272, 0.8993777632713318
        ]
    )
    mean, interval = confidence_interval(x=x, confidence=0.95)
    print(mean, interval)
