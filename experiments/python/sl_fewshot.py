import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
import torch
import argparse
import hydra
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Callable
from omegaconf import OmegaConf
sys.path.insert(0, "../..")
from src.engine import clap_backend
from src.data import prepare_data, SimpleFewShotSampler
from retrieve import compute_similarity
from utils import CustomDistance, confidence_interval, normc2d, tgt_tokenise_fn

torch.manual_seed(42)


class SingleLabelFewShot():
    def __init__(
        self, 
        dataloader: torch.nn.Module, 
        model_name: str,
        weights_pth: str, 
        prompt: str, 
        n_class: int, 
        n_supports: int, 
        n_queries: int, 
        a: float, 
        b: float,
        train_a: bool,
        distance: str,
        cuda: bool,
        tgt_tokeniser: Callable,
        fine_tune: bool = False,
        adapter_type: str = 'match', # xattention
        xatt_disturb: bool = True,
        train_epochs: int = 20,
        train_lr: float = 1e-4,
    ) -> None:
        self.dataloader = dataloader
        self.model_name = model_name
        self.weights_pth = weights_pth
        self.prompt = prompt
        self.fewshot_cfgs = {
            'n_class': n_class,
            'n_supports': n_supports,
            'n_queries': n_queries
        }
        self.a = a
        self.b = b
        self.train_a = train_a
        self.distance_fn = CustomDistance(type=distance)
        self.cuda = cuda
        self.tgt_tokeniser = tgt_tokeniser

        self.fine_tune = fine_tune
        if self.fine_tune:
            self.tr_cfgs = {
                'train_epochs': train_epochs,
                'train_lr': train_lr,
            }
            self.adapter_type = adapter_type
            self.xatt_disturb = xatt_disturb
        
        if self.fewshot_cfgs['n_supports'] == 0 and self.fine_tune:
            print(r"Warning: cannot train the model when `n_supports` = 0")
        print(r"now we do a zero-shot classification.")
        
    def forward(self, verbose: bool = False):
        CLAPWrapper = clap_backend(self.model_name)
        clap_model = CLAPWrapper(self.weights_pth, use_cuda=self.cuda)
        acc = list()
        # for xs, ys in tqdm(self.dataloader):
        for xs, ys in self.dataloader:
            if self.fewshot_cfgs['n_supports'] == 0:
                _running_acc = self._zero_shot_on_batch(model=clap_model, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs)
            else:
                if self.fine_tune: 
                    _running_acc = self._adapt_on_batch(model=clap_model, wav_paths=xs, targets=ys, a=self.a, b=self.b, train_a=self.train_a, **self.fewshot_cfgs, **self.tr_cfgs)
                else:
                    _running_acc = self._test_on_batch(model=clap_model, wav_paths=xs, targets=ys, a=self.a, b=self.b, **self.fewshot_cfgs)
            acc.append(_running_acc)
            if verbose:
                print(f"Running accuracy: {_running_acc}")
        acc = torch.tensor(acc, dtype=torch.float).mean()
        return acc

    def _adapt_on_batch(self, model: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float, train_a: bool, train_epochs: int, train_lr: float) -> Tensor:
        r"""Trainable version of few- & zero-shot classification."""
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [self.prompt + l for l in labelset]
        # CLAP forward
        with torch.no_grad():
            audio_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
            audio_embeddings = normc2d(audio_embeddings)  # normalise each column of audio embeddings
            text_embeddings = model.get_text_embeddings(caps)
        support_embeddings, query_embeddings = audio_embeddings[:n_supports * n_class], audio_embeddings[n_supports * n_class:]
        # Predict labels with affinity between support and query embeddings
        _support_targets, query_targets = targets[:n_supports * n_class], targets[n_supports * n_class:]
        support_onehots = self.tgt_tokeniser(target=_support_targets, labelset=labelset).to(device=query_embeddings.device)
        # Initialise adapter using support embeddings
        if self.adapter_type == 'match':
            adapter = torch.nn.Linear(support_embeddings.size(dim=1), support_embeddings.size(dim=0), bias=False).to(device=audio_embeddings.device)  # dtype=model.clap.dtype, device=model.clap.device
            adapter.weight = torch.nn.Parameter(support_embeddings)
        elif self.adapter_type == 'xattention':
            embed_dim = support_embeddings.size(dim=1)
            adapter = torch.nn.Linear(embed_dim, embed_dim, bias=False).to(device=audio_embeddings.device)
            if self.xatt_disturb:
                init_w = torch.eye(embed_dim) + 1e-4 * (torch.rand((embed_dim, embed_dim)) - 0.5)
            else:
                init_w = torch.eye(embed_dim)
            adapter.weight = torch.nn.Parameter(init_w.to(audio_embeddings.device))

        if self.train_a:
            alpha = torch.nn.Parameter(torch.tensor(a))
            optimiser = torch.optim.AdamW([*list(adapter.parameters()), alpha], lr=train_lr, eps=1e-4)
        else:
            optimiser = torch.optim.AdamW(adapter.parameters(), lr=train_lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, train_epochs)
        r"""Fine-tune adapter using training (support) data."""
        shuffle_idx = torch.randperm(support_embeddings.size(dim=0))
        train_emb, train_onehots = support_embeddings[shuffle_idx], support_onehots[shuffle_idx]
        adapter.train()
        for id in range(train_epochs):
            assert train_emb.size(dim=1) == support_embeddings.size(dim=1)
            if self.adapter_type == 'match':
                _attention = adapter(train_emb)  # similarity(train_emb, support_embeddings)
            elif self.adapter_type == 'xattention':
                _new_train_emb, _new_support_emb = adapter(train_emb), adapter(support_embeddings)
                _attention = _new_train_emb @ _new_support_emb.T

            _fewshot_logits = torch.exp(- b + b * _attention) @ support_onehots
            _zeroshot_logits = model.compute_similarity(train_emb, text_embeddings)
            _overall_logits = (alpha * _fewshot_logits + _zeroshot_logits)

            loss = torch.nn.functional.cross_entropy(_overall_logits, train_onehots.argmax(dim=-1))
            # Update params
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
        r"""Evaluate adapter using eval (query) data."""
        print(f"alpha: {alpha}")
        adapter.eval()
        if self.adapter_type == 'match':
            attention = adapter(query_embeddings)
        elif self.adapter_type == 'xattention':
            new_query_emb, new_support_emb = adapter(query_embeddings), adapter(support_embeddings)
            attention = new_query_emb @ new_support_emb.T
        fewshot_logits = torch.exp(- b + b * attention) @ support_onehots

        zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)
        preds = (alpha * fewshot_logits + zeroshot_logits)
        # Compare predictions with query targets
        preds = preds.argmax(dim=1).tolist()
        correct_cnt = 0
        for idx, p in enumerate(preds):
            if labelset[p] == query_targets[idx]:
                correct_cnt += 1
        _running_acc = correct_cnt / len(preds)
        return _running_acc

    def _test_on_batch(self, model: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float) -> Tensor:
        r"""Predict query logits using support embeddings and targets."""
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [self.prompt + l for l in labelset]
        # CLIP forward
        audio_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
        support_embeddings, query_embeddings = audio_embeddings[:n_supports * n_class], audio_embeddings[n_supports * n_class:]
        # Predict labels with affinity between support and query embeddings
        support_targets, query_targets = targets[:n_supports * n_class], targets[n_supports * n_class:]
        support_onehots= self.tgt_tokeniser(target=support_targets, labelset=labelset).to(device=query_embeddings.device)
        fewshot_logits = self._affinity_predict(q_x=query_embeddings, s_x=support_embeddings, s_y=support_onehots, b=self.b)
        # Predict labels with similarity between audio and text embeddings
        text_embeddings = model.get_text_embeddings(caps)
        zeroshot_logits = model.compute_similarity(query_embeddings, text_embeddings)  # size = (n_wav, n_class)
        preds = (a * fewshot_logits + zeroshot_logits).softmax(dim=-1)
        # Compare predictions with query targets
        preds = preds.argmax(dim=1).tolist()
        correct_cnt = 0
        for idx, p in enumerate(preds):
            if labelset[p] == query_targets[idx]:
                correct_cnt += 1
        _running_acc = correct_cnt / len(preds)
        return _running_acc
    
    @torch.no_grad()
    def _zero_shot_on_batch(self, model: torch.nn.Module, wav_paths: list, targets: list, n_class: int, n_supports: int, n_queries: int, a: float, b: float) -> Tensor:
        # Generate a list of selected labels and corresponding captions
        labelset = list(set(targets))
        caps = [self.prompt + l for l in labelset]
        # CLIP forward
        query_embeddings = model.get_audio_embeddings(wav_paths, resample=False)
        # Predict labels with similarity between audio and text embeddings
        text_embeddings = model.get_text_embeddings(caps)
        preds = model.compute_similarity(query_embeddings, text_embeddings).softmax(dim=-1) # size = (n_wav, n_class)
        # Compare predictions with query targets
        preds = preds.argmax(dim=1).tolist()
        correct_cnt = 0
        for idx, p in enumerate(preds):
            if labelset[p] == targets[idx]:
                correct_cnt += 1
        _running_acc = correct_cnt / len(preds)
        return _running_acc
    
    def _affinity_predict(self, q_x: Tensor, s_x: Tensor, s_y: Tensor, b: float) -> Tensor:
        r"""Predict query labels using affinity matrix between query and supports as:
            :math: `\text{logits} = \alpha As_y`
            :math: `A = \exp(-\beta d_{cos}(f(q_x), f(s_x)))`
        """
        attention = torch.exp(- b * self.distance_fn(q_x, s_x))
        return torch.mm(attention, s_y) 


@hydra.main(version_base=None, config_path='../cfgs', config_name='experiments_cfgs')
def main(cfgs: OmegaConf) -> None:
    print(f"The seetings for the experiment of {cfgs}.")
    # I/O
    db_name = cfgs['database']
    exp_type = cfgs['experiment']
    weights_pth = cfgs['model_weights_path']
    audio_dir = cfgs[db_name]['audio_dir']
    csv_path = cfgs[db_name]['csv_path']
    model_name = cfgs['model_name']
    
    DataSet, Sampler, fs_label_splits = prepare_data(data_source=db_name)
    database = DataSet(
        audio_dir=audio_dir, 
        csv_path=csv_path, 
        data_type='path', 
        target_type='category',
        )
    tgt_tokeniser = tgt_tokenise_fn('multihot') if db_name == 'fsd_fs' else tgt_tokenise_fn('onehot')
    history = list()
    for idx, val_labelset in enumerate(fs_label_splits):
        print(f"Cross-validation: {idx}/{len(fs_label_splits)}")
        # train_labelset = [x for x in range(len(database.labelset)) if x not in val_labelset]
        sampler = Sampler(
            dataset=database, 
            labelset=val_labelset,
            n_class=cfgs['fewshot']['n_class'],
            n_supports=cfgs['fewshot']['n_supports'],
            n_queries=cfgs['fewshot']['n_queries'],
            n_task=cfgs['fewshot']['n_task']
            )
        dataloader = DataLoader(database, batch_sampler=sampler, num_workers=4, pin_memory=True)
        prompt = 'this is a sound of '
        r""" Now begin our experiment."""
        fewshot = SingleLabelFewShot(
            dataloader=dataloader, 
            model_name=model_name,
            weights_pth=weights_pth, 
            prompt=prompt, 
            n_class=cfgs['fewshot']['n_class'],
            n_supports=cfgs['fewshot']['n_supports'],
            n_queries=cfgs['fewshot']['n_queries'],
            a=cfgs['fewshot']['a'],
            b=cfgs['fewshot']['b'],
            train_a=cfgs['fewshot']['train_a'],
            distance='cosine', 
            cuda=True,
            tgt_tokeniser=tgt_tokeniser,
            fine_tune=cfgs['fewshot']['fine_tune'],
            adapter_type=cfgs['fewshot']['adapter'],
            xatt_disturb=cfgs['fewshot']['xattention']['disturb'],
            train_epochs=cfgs['fewshot']['train_epochs'],
            train_lr=cfgs['fewshot']['learning_rate'],
            )
        _acc = fewshot.forward()
        history.append(_acc)
        print(f"Final accuracy={_acc}")
    mean, interval = confidence_interval(x=np.asarray(history), confidence=0.95)
    print(f"The {len(fs_label_splits)}-fold cross-validation: mean={mean}, var={interval}.")


if __name__ == '__main__':
    main()