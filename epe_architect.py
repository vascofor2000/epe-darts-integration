import itertools
from pathlib import Path
from typing import Dict, Optional

import fire
import numpy as np
import torch
from pytorch_lightning.core.saving import load_hparams_from_yaml
from torch import nn

from epe_darts.controller import SearchController
from epe_darts.data import DataModule
from epe_darts.epe_nas import get_batch_jacobian, eval_score_per_class
from epe_darts.search_cnn import SearchCNNController
from epe_darts.utils import fix_random_seed, PathLike


def extract_architecture(darts_model_path: Optional[PathLike],
                         hparams_file: PathLike,
                         dataset: str,
                         project: str = 'epe-architect',
                         nb_architectures: int = 1000,
                         batch_size: int = 32,
                         nb_batches: int = 10,
                         seed: int = 42,
                         workers: int = 4,
                         data_path: Path = Path('datasets'),
                         save_path: Path = Path('epe_architecture')):
    """
    Extract a discrete architecture from a pretrained DARTS model using EPE-NAS scoring heuristic
    if the model is provided. Otherwise generate a random model

    :param darts_model_path: Path to pretrained DARTS super-net
    :param hparams_file: Path to hyperparams file of the search phase for DARTS super-net
    :param dataset: CIFAR10 / CIFAR100 / ImageNet / MNIST / FashionMNIST
    :param project: Name of the project (to log in wandb)
    :param nb_architectures: Number of different architectures to try
    :param batch_size: Batch size
    :param nb_batches: Number of batches to run per each architecture (scores are averaged out)
    :param seed: Random seed
    :param workers: # of workers for data loading if None will be os.cpu_count() - 1
    :param data_path: Path to the dataset (download in that location if not present)
    :param save_path: Where to save the results
    """
    fix_random_seed(seed, fix_cudnn=True)
    # torch.autograd.set_detect_anomaly(True)

    data = DataModule(dataset=dataset, data_dir=data_path, split_train=False, cutout_length=0,
                      batch_size=batch_size, workers=workers)
    data.setup()
    data_iterator = itertools.cycle(data.val_dataloader())

    # Setup and load the DARTS network
    hparams = load_hparams_from_yaml(hparams_file)
    hparams['input_channels'] = data.input_channels
    hparams['n_classes'] = data.n_classes
    hparams['n_layers'] = hparams.get('n_layers', hparams.get('layers', 8))
    print('Hyper-params from', hparams_file, ':', hparams)

    net = SearchCNNController(**hparams)
    if darts_model_path is not None:
        SearchController.load_from_checkpoint(darts_model_path, net=net, image_log_path=Path('alphas'))
        print('Loaded the DARTS model')

    n_nodes = len(net.alpha_normal)
    n_ops = net.alpha_normal[0].shape[-1]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'#Nodes: {n_nodes}, #Ops: {n_ops}, device: {device}')

    scores: Dict[str, float] = {}
    for architecture in range(nb_architectures):
        for i in range(n_nodes):
            p = 3 / ((i + 2) * n_ops)
            normal = np.random.choice([0., 1.], size=(i + 2, n_ops), p=[1 - p, p])
            reduce = np.random.choice([0., 1.], size=(i + 2, n_ops), p=[1 - p, p])
            normal[:, -1] = 0.5
            reduce[:, -1] = 0.5
            net.alpha_normal[i] = nn.Parameter(torch.from_numpy(normal))
            net.alpha_reduce[i] = nn.Parameter(torch.from_numpy(reduce))

        net = net.to(device)
        net_scores = []
        for batch in range(nb_batches):
            x, target = next(data_iterator)
            x = x.to(device)
            jacobs_batch = get_batch_jacobian(net, x)
            jacobs = jacobs_batch.reshape(jacobs_batch.size(0), -1).detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            s = eval_score_per_class(jacobs, target, n_classes=data.n_classes)
            net_scores.append(s)

        score = float(np.mean(net_scores))
        genotype = net.genotype(algorithm='best')
        print(f'[{architecture}/{nb_architectures}] \t {score:.5} \t {genotype}')
        scores[f'{genotype}'] = score

    top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
    print('---- TOP 5: ----')
    [print(f'{s} \t {g}') for g, s in top]


if __name__ == '__main__':
    fire.Fire(extract_architecture)
