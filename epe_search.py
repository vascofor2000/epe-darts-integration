from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import fire
import numpy as np
import torch
from pyswarm import pso
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from epe_darts.data import DataModule
from epe_darts.epe_nas import get_batch_jacobian, eval_score_per_class
from epe_darts.search_cnn import SearchCNNController
from epe_darts.utils import fix_random_seed

fix_random_seed(42, fix_cudnn=True)
# torch.autograd.set_detect_anomaly(True)


@dataclass
class EPESearch:
    dataset: str
    nb_architectures: int = 500
    nb_weight_samples: int = 20
    batch_size: int = 32
    init_channels: int = 16
    nb_layers: int = 8
    nb_nodes: int = 4
    stem_multiplier: int = 3
    search_space: str = 'darts'
    sparsity: float = 4
    workers: int = 4
    data_path: Path = Path('datasets')
    save_path: Path = Path('initial_alphas')

    def __post_init__(self):
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.datamodule = DataModule(dataset=self.dataset, data_dir=self.data_path, split_train=False, cutout_length=0,
                                     batch_size=self.batch_size, workers=self.workers)
        self.datamodule.setup()
        self.input_channels: int = self.datamodule.input_channels
        self.n_classes: int = self.datamodule.n_classes
        self.data_loader: DataLoader = self.datamodule.train_dataloader()

    def create_net(self, alpha_normal=None, alpha_reduce=None):
        return SearchCNNController(input_channels=self.input_channels, init_channels=self.init_channels,
                                   n_classes=self.n_classes, n_layers=self.nb_layers, n_nodes=self.nb_nodes,
                                   stem_multiplier=self.stem_multiplier,
                                   search_space=self.search_space, sparsity=self.sparsity,
                                   alpha_normal=alpha_normal, alpha_reduce=alpha_reduce).to(self.device)

    def evaluate_architecture(self, alpha_normal: nn.ParameterList, alpha_reduce: nn.ParameterList,
                              data_iterator, nb_runs: int) -> List[float]:
        """
        Initializes the architecture with random weights and the provided alphas for `nb_runs` times
        and computes the EPE-NAS score for each
        """
        scores = []
        weight_runs = trange(nb_runs, desc='')
        for _ in weight_runs:
            # print('alpha_normal:')
            # [print(alpha) for alpha in alpha_normal]
            # print('alpha_reduce:')
            # [print(alpha) for alpha in alpha_reduce]
            network = self.create_net(alpha_normal, alpha_reduce)
            x, target = next(data_iterator)
            x = x.to(self.device)

            jacobs_batch = get_batch_jacobian(network, x)
            # print('jacobs:', jacobs_batch)
            jacobs = jacobs_batch.reshape(jacobs_batch.size(0), -1).detach().cpu().numpy()
            target = target.detach().cpu().numpy()

            s = eval_score_per_class(jacobs, target, n_classes=self.n_classes)
            scores.append(s)
            weight_runs.set_description(f'mean: {np.mean(scores):.2f}  score: {s:.2f}')
        return scores

    def random_search(self):
        scores: Dict[Tuple, List] = {}
        best: float = -1
        for arch in range(self.nb_architectures):
            network = self.create_net()
            alpha_normal = network.alpha_normal
            alpha_reduce = network.alpha_reduce
            arch_scores = self.evaluate_architecture(alpha_normal, alpha_reduce, data_iterator=iter(self.data_loader),
                                                     nb_runs=self.nb_weight_samples)
            scores[(alpha_normal, alpha_reduce)] = arch_scores
            torch.save(scores, self.save_path / f'random-scores-sparsity{self.sparsity}.pt')
            if np.mean(arch_scores) > best:
                print('Best was:', best, end=' ... ')
                best = float(np.mean(arch_scores))
                print('Reached:', best)
                print('Top-k Genotype:', network.genotype(algorithm='top-k'))
                print('Best Genotype:', network.genotype(algorithm='best'))
                torch.save((alpha_normal, alpha_reduce), self.save_path / f'best-sparsity{self.sparsity}.pt')
        return scores

    def pso_search(self):
        def nb_param_list_items(l: nn.ParameterList) -> int:
            return sum([np.prod(item.shape) for item in l])

        def reshape_alphas(alphas, ref_shapes: List) -> nn.ParameterList:
            res = nn.ParameterList()
            for shape in ref_shapes:
                nb_items = np.prod(shape)
                current = alphas[:nb_items]
                current = torch.from_numpy(current)
                current = torch.reshape(current, shape)
                res.append(nn.Parameter(current))
                alphas = alphas[nb_items:]
            return res

        network = self.create_net()
        low = [0.] * (nb_param_list_items(network.alpha_normal) + nb_param_list_items(network.alpha_reduce))
        high = [1.] * (nb_param_list_items(network.alpha_normal) + nb_param_list_items(network.alpha_reduce))

        scores: Dict[Tuple, List] = {}
        best: float = -1

        def eval_alphas(alphas):
            alpha_normal = reshape_alphas(np.copy(alphas), ref_shapes=[item.shape for item in network.alpha_normal])
            alphas = alphas[nb_param_list_items(network.alpha_normal):]
            alpha_reduce = reshape_alphas(np.copy(alphas), ref_shapes=[item.shape for item in network.alpha_reduce])

            arch_scores = self.evaluate_architecture(alpha_normal, alpha_reduce, data_iterator=iter(self.data_loader),
                                                     nb_runs=self.nb_weight_samples)
            scores[(alpha_normal, alpha_reduce)] = arch_scores
            torch.save(scores, self.save_path / f'pso-scores-sparsity{self.sparsity}.pt')

            nonlocal best
            if np.mean(arch_scores) > best:
                best = float(np.mean(arch_scores))
                torch.save((alpha_normal, alpha_reduce), self.save_path / f'best-sparsity{self.sparsity}.pt')

            return -np.mean(arch_scores)

        pso(eval_alphas, lb=low, ub=high, maxiter=self.nb_architectures)
        return scores


if __name__ == '__main__':
    fire.Fire(EPESearch)
