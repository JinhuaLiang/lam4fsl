import os
import torch
import csv
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from .naive_dataset import NaiveDataset, SimpleFewShotSampler

torch.manual_seed(42)
np.random.seed(42)


fsdkaggle18k_labelsets = [
    ['Glockenspiel', 'Tambourine', 'Applause', 'Cowbell', 'Hi-hat', 'Squeak', 'Meow', 'Finger_snapping', 'Scissors', 'Telephone'],
    ['Hi-hat', 'Laughter', 'Saxophone', 'Snare_drum', 'Gong', 'Burping_or_eructation', 'Tearing', 'Electric_piano', 'Violin_or_fiddle', 'Keys_jangling'],
    ['Tambourine', 'Double_bass', 'Cough', 'Drawer_open_or_close', 'Keys_jangling', 'Harmonica', 'Bus', 'Bark', 'Squeak', 'Violin_or_fiddle'],
    ['Bass_drum', 'Laughter', 'Harmonica', 'Trumpet', 'Knock', 'Scissors', 'Bus', 'Chime', 'Double_bass', 'Computer_keyboard'],
    ['Gong', 'Laughter', 'Acoustic_guitar', 'Electric_piano', 'Knock', 'Bus', 'Tambourine', 'Fireworks', 'Saxophone', 'Bark']
    ]

class FSDKaggle18K(NaiveDataset):
    r"""FSDKaggle18K dataset."""
    def __init__(
            self,
            audio_dir: str,
            csv_path: str, *,
            data_type: str = 'path',
            sample_rate: int = 44100,
            target_type: str = 'category',
            **kargs
    ) -> None:
        super().__init__()
        audio_csv_dirs = list(zip(audio_dir, csv_path))
        self.meta = self._load_meta(audio_csv_dirs)  # {path/to/file: class_category}
        assert len(self.meta) > 0
        self.indices = list(self.meta.keys())  # create indices for filename
        self.cfgs = {
            'data_type': data_type,
            'target_type': target_type
        }
        if self.cfgs['data_type'] == 'audio':
            self.cfgs['sr'] = sample_rate
        if self.cfgs['target_type'] == 'id':
            self.tokeniser = self._tokenise_category()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: Tuple[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Returns tensor of audio and its label."""
        if isinstance(item, tuple):
            x, y = item
        elif isinstance(item, int):
            x = self.indices[item]
            y = self.meta[x]
        if self.cfgs['data_type'] == 'audio':
            x = self._load_audio(x, sr=self.cfgs.sr)
        if self.cfgs['target_type'] == 'id':
            y = self.tokeniser[y]
        return x, y

    def _load_meta(self, audio_csv_dirs: str) -> dict:
        r"""Load meta information.
        Args:
            csv_path: list of tuple, [(trainset_dir, trainset_csv), ]
        Returns:
            Dict of format: path/to/file -> class_category 
        """
        meta = dict()
        for audio_dir, csv_path in audio_csv_dirs:
            with open(csv_path, 'r') as f:  # FSDKaggle18K organise the meta in the terms of ['fname','label','manually_verified','freesound_id','license']
                rows = csv.DictReader(f)
                for r in rows:
                    fname = os.path.join(audio_dir, r['fname'])
                    meta[fname] = r['label']
        return meta

    @property
    def labelset(self):
        if self.cfgs['target_type'] == 'category':
            return list(set(list(self.meta.values())))
        else:
            categories = list(set(list(self.meta.values())))
            return [self.tokeniser(c) for c in categories]
    
    def _tokenise_category(self):
        r"""Returns a dict: class_category -> class_id."""
        categories = list(set(list(self.meta.values())))
        tokeniser = dict()
        for idx, c in categories.items():
            tokeniser[c] = idx
        return tokeniser
            

if __name__ == '__main__':
    audio_dirs = ['/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.audio_train', '/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.audio_test']
    csv_paths = [
        '/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.meta/train_post_competition.csv', 
        '/data/EECS-MachineListeningLab/datasets/FSDKaggle2018/FSDKaggle2018.meta/test_post_competition_scoring_clips.csv'
        ]
    fsd = FSDKaggle18K(audio_dirs=audio_dirs, csv_paths=csv_paths, data_type='path', target_type='category')
    n_fold = 5
    n_novel_cls = 10
    # labelset = fsd.labelset
    # for i in range(n_fold):
    #     val_labelset = np.random.choice(labelset, size=n_novel_cls, replace=False)
    #     assert len(val_labelset) == n_novel_cls
    #     print(val_labelset)
    val_labelset = val_labelsets[1]
    base_labelset = [x for x in range(len(fsd.labelset)) if x not in val_labelset]
    sampler = SimpleFewShotSampler(dataset=fsd, labelset=val_labelset, n_class=10, n_supports=1, n_queries=5, n_task=10)
    dataloader = DataLoader(fsd, batch_sampler=sampler, num_workers=4, pin_memory=True)
    for x, y in dataloader:
        print(f"x={x}\n")
        print(f"target={y}\n")
