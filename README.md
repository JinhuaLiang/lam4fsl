# Training-efficient (Treff) adapter for contrastive language-audio models

This is the official repo for the INTERSPEECH2023 paper "Adapting Language-Audio Models as Few-Shot Audio Learners". We proposed Treff adapter to boostrap language-audio pretraining model (such as CLAPs) with a few of labelled examples. It beats the zero-shot classifier and existing few-shot learning methods even WITHOUT training.

<img src="./assets/Treff_adapter.png" width="800">

## Setup

You are required to just install the dependencies: `pip install -r requirements.txt` using Python 3 to get started.

## CLAP weights
Access CLAP weights: [Pretrained Model \[Zenodo\]](https://zenodo.org/record/7312125#.Y22vecvMIQ9)
Alternatively, you may also want to try out other version of CLAP model in [Pretrained Model](https://github.com/LAION-AI/).


## Usage

Please take a look at [scripts](https://github.com/JinhuaLiang/lam4fsl/tree/main/scripts) for usage examples. Ones may need to change the `WORK_DIR` and `STORAGE_DIR` to their custom direction. Also, to try out Treff adapter in more scenarios, one can directly adjust hyper-parameters, such as `MODEL_NAME` and `BATCH_SIZE`.



## Citation
If you are using this repo, please cite this work.
```
@misc{liang2023adapting,
      title={Adapting Language-Audio Models as Few-Shot Audio Learners}, 
      author={Jinhua Liang and Xubo Liu and Haohe Liu and Huy Phan and Emmanouil Benetos and Mark D. Plumbley and Wenwu Wang},
      year={2023},
      eprint={2305.17719},
      archivePrefix={arXiv},
      primaryClass={eess.AS}
}
```
## Acknowledge
This repo is build upon the great jobs of [Microsoft CLAP](https://github.com/microsoft/CLAP) and [LAION CLAP](https://github.com/LAION-AI/CLAP).
