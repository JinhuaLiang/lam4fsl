import os
import sys
import torch

sys.path.insert(0, "../..")


def compute_similarity(weights_path: str,
                       class_labels: list,
                       wav_paths: list,
                       *,
                       use_cuda: bool = False) -> torch.Tensor:
    r"""Compute similarity score using CLAP."""
    clap_model = CLAPWrapper(weights_path, use_cuda=use_cuda)
    text_embeddings = clap_model.get_text_embeddings(class_labels)
    audio_embeddings = clap_model.get_audio_embeddings(wav_paths,
                                                       resample=False)
    # size of similarity = (n_wav, n_class)
    similarity = clap_model.compute_similarity(audio_embeddings,
                                               text_embeddings)
    return similarity.softmax(dim=-1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str)
    arg = parser.parse_args()
    dataset_dir = arg.dataset
    audio_dir = os.path.join(dataset_dir, 'datasets/ESC-50/audio')
    csv_path = os.path.join(dataset_dir, 'datasets/ESC-50/meta/esc50.csv')
    syn_audio_dir = os.path.join(dataset_dir, 'datasets/ESC-50/synthetic_data')
    model_name = 'ms_clap'

    sys.path.append("/data/home/eey340/WORKPLACE/class_dropout")
    from engine.data.esc50 import SynESC50

    if model_name == 'ms_clap':
        from src.MS_CLAPWrapper import CLAPWrapper
    elif model_name == 'laion_clap':
        from src.LAION_CLAPWrapper import CLAPWrapper
    audio_dir = '/data/EECS-MachineListeningLab/datasets/ESC-50/audio'
    weights_pth = f'/data/EECS-MachineListeningLab/jinhua/ALM4FSL/ckpts/{model_name}_weights.pt'

    esc50 = SynESC50(
        audio_dir=audio_dir,
        csv_path=csv_path,
        syn_audio_dir=syn_audio_dir,
        output_fmt=['file_name', 'class_category'],
        cfg_path=
        '/data/home/eey340/WORKPLACE/class_dropout/config/esc50_config.yaml',
        fold=[1, 3, 4, 5],
    )

    dataloader = torch.utils.data.DataLoader(esc50,
                                             batch_size=1,
                                             num_workers=4,
                                             pin_memory=False)
    res = []
    for fname, cat in dataloader:
        res.append(
            compute_similarity(weights_pth, cat,
                               [os.path.join(esc50.audio_dir, fname[0])]))
        print(f"The similarity score is: {res}")
    res = torch.stack(res)
    overall = res.size(dim=0)
    one = res[res == 1].size(dim=0)
    seven = res[res > 0.7].size(dim=0)

    print(f"total correct: {one / overall}")
    print(f"total correct: {seven / overall}")