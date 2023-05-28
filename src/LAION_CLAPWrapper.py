import torch
import torchaudio
import torch.nn as nn
from typing import Optional
from transformers import RobertaTokenizer
from .laion_clap.open_clip import create_model
from .laion_clap.training.data import get_audio_features


class CLAPWrapper(nn.Module):
    def __init__(self, pretrained_path: str, sampling_rate: int = 32000, duration: int = 220500, train: bool = False, use_cuda: bool = False) -> None:
        super().__init__()
        device = 'cpu'
        precision = 'fp32'
        amodel = 'HTSAT-tiny' # or 'PANN-14'
        tmodel = 'roberta' # the best text encoder in our training
        enable_fusion = False # False if you do not want to use the fusion model
        fusion_type = 'aff_2d'
        pretrained = pretrained_path
        self.sampling_rate = sampling_rate
        self.duration = duration   
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        self.use_cuda = use_cuda
        
        self.model, self.model_cfg = create_model(amodel, tmodel, pretrained, precision=precision, device=device, enable_fusion=enable_fusion, fusion_type=fusion_type)
        if not train:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    @torch.no_grad()
    def get_text_embeddings(self, texts):
        r"""Get a single/list of sentences and return text embeddings.
        Args:
            text: str or list of str. e.g., 
            emb = get_text_embeddings("Dog"), or
            emb = get_text_embeddings(["Dog", "Cat"])
        """
        text_tokens = self.tokenizer(texts)
        embed = self.model.get_text_embedding(text_tokens)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        return embed

    @torch.no_grad()
    def get_audio_embeddings(self, audio_files: list, resample: bool = True) -> torch.Tensor:
        r"""Get list of audio files and returns audio embeddings."""
        sr = self.sampling_rate if resample else None
        audios = self._preprocess_audio(audio_files, sample_rate=sr)
        audio_dict_list = list()
        for a in audios:
            audio_dict = get_audio_features(dict(), a, self.duration, data_truncating='fusion', data_filling='repeatpad', audio_cfg=self.model_cfg['audio_cfg'])
            audio_dict_list.append(audio_dict)
        embed = self.model.get_audio_embedding(audio_dict_list)
        embed = torch.nn.functional.normalize(embed, dim=-1)
        return embed
        
    def _preprocess_audio(self, audio_files: list, sample_rate: Optional[int] = None) -> list:
        r"""Return a list of torch.Tensor, shape = (, n_time_samples)."""
        audios = list()
        # Load and reample audio
        for file in audio_files:
            wav, sr = torchaudio.load(file)
            if sample_rate and (sr != sample_rate):
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=sample_rate)
            wav = wav.squeeze().cuda() if self.use_cuda and torch.cuda.is_available() else wav.squeeze()
            audios.append(wav)
        return audios

    def batch_to_list(self, batch):
        ret = []
        for i in range(batch.size(0)):
            ret.append(batch[i])
        return ret
            
    def forward(self, batch, modality: str):        
        # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
            
        if(modality == "audio"):
            with torch.no_grad():
                audio_dict_list = []
                # assert self.sampling_rate == 32000, "We only support 32000 sampling rate"
                # batch: [bs, 1, t-samples]
                # adapt to the pre-trained checkpoint by resampling
                batch = torchaudio.functional.resample(batch, orig_freq=self.sampling_rate, new_freq=48000)  #
                for waveform in self.batch_to_list(batch):
                    audio_dict = {}
                    audio_dict = get_audio_features(audio_dict, waveform, 480000, data_truncating='fusion', data_filling='repeatpad', audio_cfg=self.model_cfg['audio_cfg'])
                    audio_dict_list.append(audio_dict)
                # [bs, 512]
                embed = self.model.get_audio_embedding(audio_dict_list)
        elif(modality == "text"):
            with torch.no_grad():
                # the 'fusion' truncate mode can be changed to 'rand_trunc' if run in unfusion mode
                text_data = self.tokenizer(batch)
                embed = self.model.get_text_embedding(text_data)
        return embed

    def tokenizer(self, text):
        result = self.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        return result
        # return {k: v.squeeze(0) for k, v in result.items()}
    
    def compute_similarity(self, audio_embeddings, text_embeddings):
        r"""Compute similarity between text and audio embeddings."""
        logit_scale = torch.nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07))).exp()
        similarity = logit_scale*text_embeddings @ audio_embeddings.T
        return similarity.T

if __name__ == '__main__':
    """ Testify code."""
    audio_path = '/data/EECS-MachineListeningLab/datasets/ESC-50/audio/2-141682-A-36.wav'
    label = ['dog', 'cat']
    pretrained_path = '/data/EECS-MachineListeningLab/jinhua/ALM4FSL/ckpts/epoch_top_0_audioset_no_fusion.pt'
    clap = CLAPWrapper(pretrained_path, sampling_rate=44100, duration=220500, train=False, use_cuda=False)
    audio_emb = clap.get_audio_embeddings([audio_path])
    txt_emb = clap.get_text_embeddings(label)
    sim = clap.compute_similarity(audio_emb, txt_emb)
    print(sim)
