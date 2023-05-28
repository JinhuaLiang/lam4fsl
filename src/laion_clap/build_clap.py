import torch
import torch.nn as nn
from model.CLAP.open_clip import create_model
from model.CLAP.training.data import get_audio_features
import torchaudio
from transformers import RobertaTokenizer

class CLAPModel(nn.Module):
    def __init__(self, pretrained_path, sampling_rate=32000):
        super().__init__()
        device = 'cpu'
        precision = 'fp32'
        amodel = 'HTSAT-tiny' # or 'PANN-14'
        tmodel = 'roberta' # the best text encoder in our training
        enable_fusion = False # False if you do not want to use the fusion model
        fusion_type = 'aff_2d'
        pretrained = pretrained_path
        self.sampling_rate = sampling_rate        
        self.tokenize = RobertaTokenizer.from_pretrained('roberta-base')
        
        self.model, self.model_cfg = create_model(amodel, tmodel, pretrained, precision=precision, device=device, enable_fusion=enable_fusion, fusion_type=fusion_type)
        for p in self.model.parameters():
            p.requires_grad = False
            
        self.model.eval()
        
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
                batch = torchaudio.functional.resample(batch, orig_freq=self.sampling_rate, new_freq=48000)
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
        return {k: v.squeeze(0) for k, v in result.items()}
