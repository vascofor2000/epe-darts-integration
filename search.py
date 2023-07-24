""" Search cell """
from typing import Union, List, Optional

import fire
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from epe_darts.data import DataModule
from epe_darts.search_cnn import SearchCNNController
from epe_darts.controller import SearchController
from epe_darts.utils import fix_random_seed, ExperimentSetup


def main(name: str, dataset: str, data_path: str = 'datasets/', project: str = 'search-epe-darts',
         search_space: str = 'darts', single_level_optimization: bool = False,
         batch_size: int = 64, epochs: int = 50, seed: int = 42,
         print_freq: int = 50, gpus: Union[int, List[int]] = -1, workers: Optional[int] = None,
         init_channels: int = 16, n_layers: int = 8, nodes: int = 4, stem_multiplier: int = 3,
         w_lr: float = 0.025, w_lr_min: float = 0.001, w_momentum: float = 0.9, w_weight_decay: float = 3e-4,
         w_grad_clip: float = 5., nesterov: bool = False,
         sparsity: float = 1,
         alpha_lr: float = 3e-4, alpha_weight_decay: float = 1e-3, alphas_path: Optional[str] = None,
         mask_alphas: bool = False, prune_strategy: str = 'smallest', amended_hessian: bool = False,
         normal_none_penalty: float = 0, reduce_none_penalty: float = 0):
    """
    :param name: Experiment name
    :param dataset: CIFAR10 / CIFAR100 / ImageNet / MNIST / FashionMNIST
    :param data_path: Path to the dataset (download in that location if not present)
    :param project: Name of the project (to log in wandb)
    :param search_space: On which search space to perform the search: {darts, nas-bench-201, connect-nas-bench}
    :param single_level_optimization: Whether to use bi-level or single-level optimization
    :param batch_size: Batch size
    :param epochs: # of training epochs
    :param seed: Random seed
    :param print_freq: Logging frequency
    :param gpus: Lis of GPUs to use or a single GPU (will be ignored if no GPU is available)
    :param workers: # of workers for data loading if None will be os.cpu_count() - 1
    :param init_channels: Initial channels
    :param n_layers: # of layers in the network (number of cells)
    :param nodes: # of nodes in a cell
    :param stem_multiplier: Stem multiplier
    :param w_lr: Learning rate for network weights
    :param w_lr_min: Minimum learning rate for network weights
    :param w_momentum: Momentum for network weights
    :param w_weight_decay: Weight decay for network weights
    :param w_grad_clip: Gradient clipping threshold for network weights
    :param nesterov: Whether to use nesterov for SGD of weights or no
    :param sparsity: Entmax(sparisty) for alphas [1 is equivalent to Softmax]
    :param alpha_lr: Learning rate for alphas
    :param alpha_weight_decay: Weight decay for alphas
    :param alphas_path: Optional path for initial alphas (will be loaded as a torch file)
    :param mask_alphas: Whether to mask alphas and prune them or no
    :param prune_strategy: Prune the `smallest`/`zero` connection at a time
    :param amended_hessian: Whether to use amended hessian computation or no
    :param normal_none_penalty: Penalize none connections in normal cell
    :param reduce_none_penalty: Penalize none connections in reduce cell
    """
    hyperparams = locals()
    # set seed
    fix_random_seed(seed, fix_cudnn=True)
    experiment = ExperimentSetup(name=name, create_latest=True, long_description="""
    Trying out Pytorch Lightning
    """)

    data = DataModule(dataset=dataset, data_dir=data_path, split_train=True,
                      cutout_length=0, batch_size=batch_size, workers=workers)
    data.setup()

    alpha_normal, alpha_reduce = torch.load(alphas_path) if alphas_path else (None, None)
    net = SearchCNNController(data.input_channels, init_channels, data.n_classes, n_layers, nodes, stem_multiplier,
                              search_space=search_space,
                              sparsity=sparsity, prune_strategy=prune_strategy,
                              alpha_normal=alpha_normal, alpha_reduce=alpha_reduce, mask_alphas=mask_alphas)
    model = SearchController(net, experiment.log_dir / 'cell_images',
                             bi_level_optimization=not single_level_optimization,
                             w_lr=w_lr, w_momentum=w_momentum, w_weight_decay=w_weight_decay, w_lr_min=w_lr_min,
                             w_grad_clip=w_grad_clip, nesterov=nesterov,
                             alpha_lr=alpha_lr, alpha_weight_decay=alpha_weight_decay, amended_hessian=amended_hessian,
                             normal_none_penalty=normal_none_penalty, reduce_none_penalty=reduce_none_penalty,
                             max_epochs=epochs)

    # callbacks = [
    #     RankingChangeEarlyStopping(monitor_param=param, patience=10)
    #     for name, param in model.named_parameters()
    #     if 'alpha_normal' in name
    # ]

    loggers = [
        CSVLogger(experiment.log_dir, name='history'),
        TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
        WandbLogger(name=experiment.name, project=project,
                    save_dir=experiment.log_dir, save_code=True, notes=experiment.long_description),
        # AimLogger(experiment=experiment.name),
    ]
    for logger in loggers:
        logger.log_hyperparams(hyperparams)

    trainer = Trainer(logger=loggers, log_every_n_steps=print_freq,
                      gpus=-1 if torch.cuda.is_available() else None,
                      max_epochs=epochs, terminate_on_nan=True,
                      gradient_clip_val=w_grad_clip if single_level_optimization else 0,
                      callbacks=[
                          # EarlyStopping(monitor='valid_top1', patience=5, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path,
                                          filename='model-{epoch:02d}-{valid_top1:.2f}',
                                          monitor='valid_top1', mode='max', save_top_k=5,
                                          verbose=True),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    fire.Fire(main)
