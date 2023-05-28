import os
import torch
import torchaudio
import numpy as np


class NaiveDataset():
    r"""An abstract dataset class."""
    def _load_audio(self, audio_path: str, sr: int = 44100, mono: bool = True) -> torch.Tensor:
        wav, ori_sr = torchaudio.load(audio_path, normalize=True)
        if not ori_sr == sr:
            wav = torchaudio.functional.resample(wav, orig_freq=ori_sr, new_freq=sr)
        wav = wav.squeeze() if mono == True else wav
        return wav

class BalancedSampler():
    r"""A balanced sampler for fully-supervised learning. \n
        This is a batch sample.
    Args:
        dataset: data source, e.g. instance of torch.data.Dataset, data[item] = {filename: labels}.
        labelset: a list of novel classes.
        n_sample: number of samples per class.
    """
    def __init__(self, dataset: torch.nn.Module, labelset: list, n_sample: int, batch_size: int, drop_last: bool = False) -> None:
        # assert (n_sample * len(labelset)) % batch_size == 0
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        if self.drop_last:
            self.n_batch = n_sample * len(labelset) // self.batch_size
        else:
            self.n_batch = torch.ceil(torch.tensor(n_sample * len(labelset) / self.batch_size)).int().item()
        # Create a data subset whose label is in `labelset`
        self.subset, _tmp = list(), dict(zip(labelset, [[] for _ in labelset]))
        for fpath, lbl in self.dataset.meta.items():
            if lbl in labelset:
                _tmp[lbl].append(fpath)
        for lbl, f in _tmp.items():
            files = np.stack(f)
            files = np.random.choice(files, size=n_sample, replace=False).tolist()
            self.subset.extend(files)
        np.random.shuffle(self.subset)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        r"""Returns a list of format: (`file_path`, `target_id`)."""
        for idx in range(self.n_batch):
            if not self.drop_last and idx == (self.n_batch - 1):
                batch_x = self.subset[idx * self.batch_size :]
            else:
                batch_x = self.subset[idx * self.batch_size : (idx + 1) * self.batch_size]
            batch_y = [self.dataset.meta[x] for x in batch_x]
            yield zip(batch_x, batch_y)

class SimpleFewShotSampler():
    r"""A simple few-shot sampler.
    Args:
        dataset: data source, e.g. instance of torch.data.Dataset, data[item] = {filename: labels}.
        labelset: a set of novel classes.
        n_class: number of novel classes in an episode (i.e., 'n' ways).
        n_supports: number of support samples in each novel class (i.e., 'k' shot).
        n_queries: number of queries for each novel class.
        n_task: total number of tasks (or episodes) in one epoch.
    """
    def __init__(
            self,
            dataset: torch.nn.Module,
            labelset: list, 
            n_class: int = 15,
            n_supports: int = 5,
            n_queries: int = 5,
            n_task: int = 100,
            **kargs
    ) -> None:
        self.dataset = dataset
        self.labelset = labelset
        self.n_cls = n_class
        self.n_supports = n_supports
        self.n_queries = n_queries
        self.n_task = n_task

    def __len__(self):
        return self.n_task

    def __iter__(self):
        r"""Returns a list of format: (`file_path`, `target_id`)."""
        for _ in range(self.n_task):
            batch_x, batch_y = list(), list()
            selected_classes = np.random.choice(self.labelset, size=self.n_cls, replace=False)
            # Create a data subset containing attached to novel classes ONLY
            subset = {'fpath': [], 'label': []}
            for fpath, lbl in self.dataset.meta.items():
                if lbl in selected_classes:
                    subset['fpath'].append(fpath)
                    subset['label'].append(lbl)
            subset['fpath'] = np.stack(subset['fpath'])
            subset['label'] = np.stack(subset['label'])
            # Sample support examples and assign with labels
            supports = dict()
            for n in selected_classes:
                _candidate = subset['fpath'][subset['label'] == n]
                _samples = np.random.choice(_candidate, size=self.n_supports, replace=False)
                batch_x.extend(_samples.tolist())
                batch_y.extend([n] * self.n_supports)
                supports[n] = _samples  # store values instead of use them again when sampling queries
            # Sample query examples and assign with labels
            for n in selected_classes:
                _candidate = subset['fpath'][subset['label'] == n]
                _samples = np.random.choice(_candidate[np.isin(_candidate, supports[n], invert=True)], size=self.n_queries, replace=False)
                batch_x.extend(_samples.tolist())
                batch_y.extend([n] * self.n_queries)
            yield zip(batch_x, batch_y)
