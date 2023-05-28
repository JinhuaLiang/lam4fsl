import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import argparse
import hydra
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from typing import Optional, Callable
from omegaconf import OmegaConf
sys.path.insert(0, "../..")
from src.engine import clap_backend
from src.data import prepare_data, SimpleFewShotSampler
from retrieve import compute_similarity
from utils import CustomDistance, confidence_interval, normc2d, tgt_tokenise_fn

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)


class Finetune():
    def __init__(
        self, 
        train_dataloader: torch.nn.Module,
        eval_dataloader: torch.nn.Module,
        model_name: str,
        weights_pth: str, 
        cuda: bool,
        tgt_tokeniser: Callable,
        n_class: int,
        train_epochs: int = 20,
        train_lr: float = 1e-4,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model_name = model_name
        self.weights_pth = weights_pth
        self.cuda = cuda
        self.tgt_tokeniser = tgt_tokeniser
        self.tr_cfgs = {
            'n_class': n_class,
            'train_epochs': train_epochs,
            'train_lr': train_lr,
            'verbose': True
        }

    def forward(self) -> dict:
        CLAPWrapper = clap_backend(self.model_name)
        clap = CLAPWrapper(self.weights_pth, n_class=self.tr_cfgs['n_class'], train=True, use_cuda=self.cuda)
        model = clap.clap.audio_encoder
        get_audio_fn = clap.preprocess_audio
        labelset = self.train_dataloader.dataset.labelset
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.tr_cfgs['train_lr'], eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, self.tr_cfgs['train_epochs'])
        model = self.train(model, get_audio_fn, self.train_dataloader, labelset, optimiser, scheduler, n_epochs=self.tr_cfgs['train_epochs'], verbose=self.tr_cfgs['verbose'])
        cls_results, cls_counts, predictions, ground_truth = self.eval(model, get_audio_fn, self.eval_dataloader, labelset)
        return cls_results, cls_counts

    def train(self, model: torch.nn.Module, get_audio_fn: Callable, train_dataloader: torch.nn.Module, labelset: list, optimiser: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, n_epochs: int, verbose: bool = False) -> torch.nn.Module:
        model.train()
        for epoch in range(n_epochs):
            _running_loss = 0
            for batch_x, batch_y in train_dataloader:
                batch_x = get_audio_fn(batch_x, resample=False)
                batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[2])
                # Append [0] the audio emebdding, [1] has output class probabilities
                logits = model(batch_x)[1]
                
                batch_y = self.tgt_tokeniser(batch_y, labelset=labelset).to(device=batch_x.device)
                loss = torch.nn.functional.cross_entropy(logits, batch_y.argmax(dim=-1))
                _running_loss += loss.detach().item()
                # Update params
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                scheduler.step()
            if verbose:
                print(f"The {epoch}-th training loss: {_running_loss/len(train_dataloader)}")
        return model

    @torch.no_grad()
    def eval(self, model: torch.nn.Module, get_audio_fn: Callable, eval_dataloader: torch.nn.Module, labelset: list) -> tuple:
        cls_results = dict(zip(labelset, [0 for _ in labelset]))
        cls_counts = dict(zip(labelset, [0 for _ in labelset]))
        predictions, ground_truth = list(), list()
        model.eval()
        for batch_x, batch_y in eval_dataloader:
            batch_x = get_audio_fn(batch_x, resample=False)
            batch_x = batch_x.reshape(batch_x.shape[0], batch_x.shape[2])
            logits = model(batch_x)[1]
            preds = logits.argmax(dim=1).tolist()
            predictions.extend(preds)
            ground_truth.extend(batch_y)
            for idx, p in enumerate(preds):
                if labelset[p] == batch_y[idx]:
                    cls_results[labelset[p]] += 1
                cls_counts[batch_y[idx]] += 1
        return cls_results, cls_counts, predictions, ground_truth
        

def esc50(cfgs: OmegaConf) -> None:
    print(f"The settings for the esc50 experiment of {cfgs}.")
    # I/O
    audio_dir = cfgs['esc50']['audio_dir']
    csv_path = cfgs['esc50']['csv_path']
    weights_pth = cfgs['model_weights_path']
    
    history = dict()
    for fold in range(1, 6):
        print(f"Cross-validation: {fold}/5")
        DataSet, Sampler, fs_label_splits = prepare_data(data_source='esc50_fewshot_finetune')
        tgt_tokeniser = tgt_tokenise_fn('onehot')
        train_database = DataSet(
            audio_dir=audio_dir, 
            csv_path=csv_path, 
            fold=[f for f in range(1, 6) if f != fold],
            data_type='path',
            target_type='category',
            )
        eval_database = DataSet(
            audio_dir=audio_dir, 
            csv_path=csv_path, 
            fold=[fold],
            data_type='path',
            target_type='category',
            )
        val_labelset = list(range(50))
        train_sampler = Sampler(
            dataset=train_database, 
            labelset=val_labelset, 
            n_sample=cfgs['fewshot']['n_supports'], 
            batch_size=cfgs['fewshot']['batch_size']
        )
        eval_sampler = Sampler(
            dataset=eval_database, 
            labelset=val_labelset,
            n_sample=cfgs['fewshot']['n_queries'],
            batch_size=cfgs['fewshot']['batch_size']
            )
        train_dataloader = DataLoader(train_database, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
        eval_dataloader = DataLoader(eval_database, batch_sampler=eval_sampler, num_workers=4, pin_memory=True)
        prompt = 'this is a sound of '
        r""" Now begin our experiment."""
        algorithm = Finetune(
            train_dataloader=train_dataloader, 
            eval_dataloader=eval_dataloader,
            model_name=cfgs['model_name'],
            weights_pth=weights_pth, 
            n_class=cfgs['fewshot']['n_class'],
            cuda=True,
            tgt_tokeniser=tgt_tokeniser,
            train_epochs=cfgs['fewshot']['train_epochs'],
            train_lr=cfgs['fewshot']['learning_rate'],
            )
        cls_results, cls_counts = algorithm.forward()
        for cat, cnt in cls_results.items():
            acc = cnt / cls_counts['cat']
            print(f"Class {cat}'s accuracy={acc}")
            try:
                history[cat].append(acc)
            except:
                history[cat] = [acc]
    print(r"======")
    print(r"Summary on 5-fold validation:")
    overall_acc = list()
    for cat, acc in history.items():
        mean, interval = confidence_interval(x=np.stack(acc), confidence=0.95)
        overall_acc.append(mean)
        print(f"Acc. of class {cat}={mean} +- {interval}.")
    overall_acc = np.stack(overall_acc).mean()
    print(f"Overall accuracy = {overall_acc}")
    print(r"======")

@hydra.main(version_base=None, config_path='../cfgs', config_name='esc50_fullsize')
def main(cfgs: OmegaConf) -> None:
    if cfgs['database'] == 'esc50':
        esc50(cfgs)
    elif cfgs['database'] == 'fsdkaggle18k_fullsize':
        fsdkaggle18k(cfgs)


if __name__ == '__main__':
    main()
