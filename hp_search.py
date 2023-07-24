from typing import Union, List, Optional

import ConfigSpace
import fire
import hpbandster.core.nameserver as hpns
import torch
import wandb
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from epe_darts import genotypes as gt, ops
from epe_darts.augment_cnn import AugmentCNN
from epe_darts.data import DataModule
from epe_darts.genotypes import Genotype
from epe_darts.utils import fix_random_seed, ExperimentSetup


class Architect(Worker):
    def __init__(self, *args, project: str, dataset: str, search_space: str = 'darts',
                 data_path: str = 'datasets/', batch_size: int = 128, seed: int = 42,
                 lr: float = 0.025, momentum: float = 0.9, weight_decay: float = 3e-4, grad_clip: float = 5.,
                 print_freq: int = 200, gpus: Union[List[int], int] = -1,
                 init_channels: int = 16, n_layers: int = 8, n_nodes: int = 4, stem_multiplier: int = 3,
                 workers: Optional[int] = None,
                 aux_weight: float = 0, cutout_length: int = 0, **kwargs):
        """
        :param project: Name of the project (to log in wandb)
        :param dataset: CIFAR10 / CIFAR100 / ImageNet / MNIST / FashionMNIST
        :param search_space: On which search space to perform the search: {darts, nas-bench-201, connect-nas-bench}
        :param data_path: Path to the dataset (download in that location if not present)
        :param batch_size: Batch size
        :param seed: Random seed
        :param lr: Initial learning rate for weights
        :param momentum: SGD momentum
        :param weight_decay: SGD weight decay
        :param grad_clip: Gradient clipping threshold
        :param print_freq: Logging frequency
        :param gpus: Lis of GPUs to use or a single GPU (will be ignored if no GPU is available)
        :param init_channels: Initial channels
        :param n_layers: # of layers in the network (number of cells)
        :param n_nodes: # of nodes in a cell
        :param stem_multiplier: Stem multiplier
        :param workers: # of workers for data loading if None will be os.cpu_count() - 1
        :param aux_weight: Auxiliary loss weight
        :param cutout_length: Cutout length (for augmentation)
        """
        self.project: str = project
        self.dataset: str = dataset
        self.search_space: str = search_space
        self.data_path: str = data_path
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.lr: float = lr
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.grad_clip: float = grad_clip
        self.print_freq: int = print_freq
        self.gpus: Union[List[int], int] = gpus
        self.init_channels: int = init_channels
        self.layers: int = n_layers
        self.n_nodes: int = n_nodes
        self.stem_multiplier: int = stem_multiplier
        self.workers: Optional[int] = workers
        self.aux_weight: float = aux_weight
        self.cutout_length: int = cutout_length
        super().__init__(*args, **kwargs)

    def train(self, name: str, genotype: str, max_minutes: int):
        """
        :param name: Experiment name
        :param genotype: Cell genotype
        :param max_minutes: Number of hours to train
        """
        hyperparams = locals()

        fix_random_seed(self.seed, fix_cudnn=True)
        experiment = ExperimentSetup(name=name, create_latest=True, long_description="""
        Trying out Pytorch Lightning
        """)
        genotype: Genotype = gt.from_str(genotype)

        data = DataModule(dataset=self.dataset, data_dir=self.data_path, split_train=True, return_train_val=False,
                          cutout_length=self.cutout_length, batch_size=self.batch_size, workers=self.workers)
        data.setup()

        model = AugmentCNN(data.input_size, data.input_channels, self.init_channels, data.n_classes, self.layers,
                           self.aux_weight, genotype, stem_multiplier=self.stem_multiplier,
                           lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay,
                           max_epochs=int(2.5 * max_minutes))  # A gross empirical approximation

        loggers = [
            CSVLogger(experiment.log_dir, name='history'),
            TensorBoardLogger(experiment.log_dir, name=experiment.name, default_hp_metric=False),
            WandbLogger(name=experiment.name, project=self.project,
                        save_dir=experiment.log_dir, save_code=True, notes=experiment.long_description),
            # AimLogger(experiment=experiment.name),
        ]
        for logger in loggers:
            logger.log_hyperparams(hyperparams)

        trainer = Trainer(logger=loggers, log_every_n_steps=self.print_freq,
                          gpus=self.gpus if torch.cuda.is_available() else None,
                          max_time={'minutes': max_minutes}, terminate_on_nan=True,
                          gradient_clip_val=self.grad_clip,
                          callbacks=[
                              # DropPathCallback(max_epochs=epochs, drop_path_prob=self.drop_path_prob),
                              # EarlyStopping(monitor='valid_top1', patience=5, verbose=True, mode='max'),
                              ModelCheckpoint(dirpath=experiment.model_save_path,
                                              filename='model-{epoch:02d}-{valid_top1:.2f}',
                                              monitor='valid_top1', mode='max', save_top_k=5,
                                              verbose=True),
                              LearningRateMonitor(logging_interval='epoch'),
                          ])

        trainer.fit(model, datamodule=data)
        wandb.finish()
        return trainer.callback_metrics

    def compute(self, config, budget, **kwargs):
        normals = [[config[f'n{n1}-{n2}'] for n1 in range(n2)] for n2 in range(2, self.n_nodes + 2)]
        reduces = [[config[f'r{r1}-{r2}'] for r1 in range(r2)] for r2 in range(2, self.n_nodes + 2)]
        genotype = f'''Genotype(
            normal={normals}, 
            normal_concat=range(2, {self.n_nodes + 2}),
            reduce={reduces},
            reduce_concat=range(2, {self.n_nodes + 2})
        )'''
        print('Genotype:', genotype)
        res = self.train(genotype=genotype, name=f'exp-{int(budget)}', max_minutes=int(budget))
        print('RES:', res)
        return ({
            'loss': -float(res['valid_top1']),
            'info': {
                'valid_top1': float(res['valid_top1']),
                'valid_top5': float(res['valid_top5']),
            },
        })

    def get_configspace(self):
        """
        Genotype(normal=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],  # n2
                         [('max_pool_3x3', 2), ('max_pool_3x3', 1)],  # n3
                         [('max_pool_3x3', 2), ('max_pool_3x3', 0)],  # n4
                         [('skip_connect', 4), ('max_pool_3x3', 2)]], # n5
                 normal_concat=range(2, 6),
                 reduce=[[('skip_connect', 1), ('sep_conv_5x5', 0)],  # r2
                         [('sep_conv_5x5', 0), ('sep_conv_3x3', 2)],  # r3
                         [('max_pool_3x3', 3), ('sep_conv_5x5', 0)],  # r4
                         [('max_pool_3x3', 4), ('max_pool_3x3', 3)]], # r5
                 reduce_concat=range(2, 6))
        """
        primitives = ops.SEARCH_SPACE2OPS[self.search_space]

        config = ConfigSpace.ConfigurationSpace()
        # Normal cell configuration
        for n2 in range(2, self.n_nodes + 2):
            for n1 in range(n2):
                config.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(
                    f'n{n1}-{n2}',
                    [(op, n1) for op in primitives]
                ))

        # Reduction cell configuration
        for r2 in range(2, self.n_nodes + 2):
            for r1 in range(r2):
                config.add_hyperparameter(ConfigSpace.CategoricalHyperparameter(
                    f'r{r1}-{r2}',
                    [(op, r1) for op in primitives]
                ))

        print(config)
        return config


def main(project: str = 'hyperband', dataset: str = 'CIFAR100', nb_successive_halvings: int = 6):
    NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
    NS.start()

    w = Architect(nameserver='127.0.0.1', run_id='example1', project=project, dataset=dataset)
    w.run(background=True)

    bohb = BOHB(configspace=w.get_configspace(), run_id='example1', nameserver='127.0.0.1',
                min_budget=5, max_budget=200)
    res = bohb.run(n_iterations=nb_successive_halvings)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    nb_evals = sum([r.budget for r in res.get_all_runs()]) / 10
    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print(f'Total budget corresponds to {nb_evals:.1f} full function evaluations.')


if __name__ == '__main__':
    fire.Fire(main)
