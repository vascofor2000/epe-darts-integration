import os
import pathlib
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, TypeVar

import numpy as np
import torch


PathLike = TypeVar("PathLike", str, pathlib.Path, bytes, os.PathLike)


def fix_random_seed(seed: int = 42, fix_cudnn: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if fix_cudnn:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True


class ExperimentSetup:
    def __init__(self, name: str, long_description: Optional[str] = None, create_latest: bool = False):
        """ Keeps track of the experiment path, model save path, log directory, and sessions """
        self.name = name
        self.long_description = long_description
        self.experiment_time = datetime.now().replace(microsecond=0).isoformat()

        self.experiment_path = Path('experiments') / f'{self.experiment_time}_{self.name}'
        self.model_save_path = self.experiment_path / 'models/'
        self.log_dir = self.experiment_path / 'logs/'
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if long_description:
            with open(self.log_dir / 'description.txt', 'w') as f:
                f.write(long_description.strip())

        if create_latest:
            latest = Path('experiments/latest/').absolute()
            if latest.exists(): latest.unlink()  # missing_ok=True
            latest.symlink_to(self.experiment_path.absolute(), target_is_directory=True)

        print(f'Logging experiments at: `{self.experiment_path.absolute()}`')


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.flatten(correct[:k]).float().sum(0)
        # correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res
