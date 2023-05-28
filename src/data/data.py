from typing import Tuple
from torch.nn import Module
from .naive_dataset import SimpleFewShotSampler, BalancedSampler
from .esc50 import ESC50, fs_label_splits
from .fsdkaggle18k import FSDKaggle18K, fsdkaggle18k_labelsets
from .fsd_fs import FSD_FS, MLFewShotSampler, fsd50k_splits


def prepare_data(data_source: str) -> Tuple[Module, list]:
    r"""Returns a dataloader and a set of novel class labels."""
    if data_source == "esc50":
        return ESC50, SimpleFewShotSampler, fs_label_splits
    elif data_source == "fsdkaggle18k":
        return FSDKaggle18K, SimpleFewShotSampler, fsdkaggle18k_labelsets
    elif data_source == 'fsd_fs':
        return FSD_FS, MLFewShotSampler, fsd50k_splits
    elif data_source == "esc50_fewshot_finetune":
        return ESC50, BalancedSampler, fs_label_splits
    else:
        raise ValueError(f"Cannot find a datasource name {data_source}.")
