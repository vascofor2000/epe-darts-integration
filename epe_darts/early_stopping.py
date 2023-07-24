import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class RankingChangeEarlyStopping(EarlyStopping):

    def __init__(
        self,
        monitor_param: nn.Parameter,
        patience: int = 10,
        verbose: bool = False,
        mode: str = 'change',
    ):
        self.mode_dict.update({'change': torch.equal})
        super().__init__(patience=patience, verbose=verbose, mode=mode)
        self.monitor_param = monitor_param
        self.latest_ranking = None

    def _run_early_stopping_check(self, trainer, pl_module: pl.LightningModule):
        """
        Keeps track of the ranking for the provided metric
        If the ranking doesn't change for `patience` epochs, tells the trainer to stop the training.
        """

        if (
                trainer.fast_dev_run            # disable early_stopping with fast_dev_run
                or self.monitor_param is None   # short circuit if metric not present
        ):
            return  # short circuit if metric not present

        # Sort the current version of param
        current_ranking = torch.argsort(self.monitor_param.detach(), dim=-1)
        if self.latest_ranking is None or not torch.equal(current_ranking, self.latest_ranking):
            self.wait_count = 0
            self.latest_ranking = current_ranking
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stopped_epoch = trainer.current_epoch
                trainer.should_stop = True

        # stop every ddp process if any world process decides to stop
        trainer.should_stop = trainer.training_type_plugin.reduce_boolean_decision(trainer.should_stop)
