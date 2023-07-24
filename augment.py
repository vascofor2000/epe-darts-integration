from typing import Union, List, Optional

import fire
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger, WandbLogger

from epe_darts.augment_cnn import AugmentCNN, DropPathCallback
from epe_darts.data import DataModule
from epe_darts.genotypes import Genotype
from epe_darts.utils import fix_random_seed, ExperimentSetup
from epe_darts import genotypes as gt


def train(name: str, dataset: str, genotype: str, project: str = 'augment-epe-darts',
          data_path: str = 'datasets/', batch_size: int = 96, epochs: int = 600, seed: int = 42,
          lr: float = 0.025, momentum: float = 0.9, weight_decay: float = 3e-4, grad_clip: float = 5.,
          print_freq: int = 200, gpus: Union[List[int], int] = -1,
          init_channels: int = 36, layers: int = 20, stem_multiplier: int = 3, workers: Optional[int] = None,
          aux_weight: float = 0.4, cutout_length: int = 16, drop_path_prob: float = 0.2):
    """
    Training Augmented Model

    :param name: Experiment name
    :param dataset: CIFAR10 / CIFAR100 / ImageNet / MNIST / FashionMNIST
    :param genotype: Cell genotype
    :param project: Name of the project (to log in wandb)
    :param data_path: Path to the dataset (download in that location if not present)
    :param batch_size: Batch size
    :param epochs: # of training epochs
    :param seed: Random seed
    :param lr: Initial learning rate for weights
    :param momentum: SGD momentum
    :param weight_decay: SGD weight decay
    :param grad_clip: Gradient clipping threshold
    :param print_freq: Logging frequency
    :param gpus: Lis of GPUs to use or a single GPU (will be ignored if no GPU is available)
    :param init_channels: Initial channels
    :param layers: # of layers in the network (number of cells)
    :param stem_multiplier: Stem multiplier
    :param workers: # of workers for data loading if None will be os.cpu_count() - 1
    :param aux_weight: Auxiliary loss weight
    :param cutout_length: Cutout length (for augmentation)
    :param drop_path_prob: Probability of dropping a path
    """
    hyperparams = locals()

    fix_random_seed(seed, fix_cudnn=True)
    experiment = ExperimentSetup(name=name, create_latest=True, long_description="""
    Trying out Pytorch Lightning
    """)
    genotype: Genotype = gt.from_str(genotype)

    data = DataModule(dataset, data_path, split_train=False,
                      cutout_length=cutout_length, batch_size=batch_size, workers=workers)
    data.setup()

    model = AugmentCNN(data.input_size, data.input_channels, init_channels, data.n_classes, layers,
                       aux_weight, genotype, stem_multiplier=stem_multiplier,
                       lr=lr, momentum=momentum, weight_decay=weight_decay, max_epochs=epochs)

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
                      gpus=gpus if torch.cuda.is_available() else None,
                      max_epochs=epochs, terminate_on_nan=True,
                      gradient_clip_val=grad_clip,
                      callbacks=[
                          DropPathCallback(max_epochs=epochs, drop_path_prob=drop_path_prob),
                          # EarlyStopping(monitor='valid_top1', patience=5, verbose=True, mode='max'),
                          ModelCheckpoint(dirpath=experiment.model_save_path,
                                          filename='model-{epoch:02d}-{valid_top1:.2f}',
                                          monitor='valid_top1', mode='max', save_top_k=5,
                                          verbose=True),
                          LearningRateMonitor(logging_interval='epoch'),
                      ])

    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    fire.Fire(train)
