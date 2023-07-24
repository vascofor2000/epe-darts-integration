import copy
from pathlib import Path
from typing import Dict

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from plotly.subplots import make_subplots

from epe_darts import genotypes as gt, utils
from epe_darts.architect import Architect


class SearchController(pl.LightningModule):
    def __init__(self, net: nn.Module, image_log_path: Path,
                 bi_level_optimization: bool = True,
                 w_lr=0.025, w_momentum=0.9, w_weight_decay: float = 3e-4, w_lr_min: float = 0.001, w_grad_clip=5.,
                 nesterov=False,
                 alpha_lr=3e-4, alpha_weight_decay=1e-3, amended_hessian: bool = False,
                 normal_none_penalty: float = 0, reduce_none_penalty: float = 0,
                 max_epochs: int = 50):
        super().__init__()
        self.save_hyperparameters('image_log_path', 'bi_level_optimization',
                                  'w_lr', 'w_momentum', 'w_weight_decay', 'w_lr_min', 'w_grad_clip', 'nesterov',
                                  'alpha_lr', 'alpha_weight_decay', 'amended_hessian',
                                  'normal_none_penalty', 'reduce_none_penalty', 'max_epochs')
        #COMENTEI A SEGUINTE LINHA PORQUE ORIGINAVA ERRO (can't set attribute)
        #self.automatic_optimization = not bi_level_optimization
        self.bi_level_optimization: bool = bi_level_optimization

        self.image_log_path: Path = image_log_path
        self.image_log_path.mkdir(parents=True, exist_ok=True)

        self.w_lr: float = w_lr
        self.w_momentum: float = w_momentum
        self.w_weight_decay: float = w_weight_decay
        self.w_lr_min: float = w_lr_min
        self.w_grad_clip: float = w_grad_clip
        self.nesterov: bool = nesterov
        self.alpha_lr: float = alpha_lr
        self.alpha_weight_decay: float = alpha_weight_decay
        self.amended_hessian: bool = amended_hessian
        self.normal_none_penalty: float = normal_none_penalty
        self.reduce_none_penalty: float = reduce_none_penalty
        self.max_epochs: int = max_epochs

        self.epoch2normal_alphas: Dict = {}
        self.epoch2reduce_alphas: Dict = {}
        self.epoch2raw_normal_alphas: Dict = {}
        self.epoch2raw_reduce_alphas: Dict = {}

        self.net: nn.Module = net
        self.net_copy: nn.Module = copy.deepcopy(net)
        self.architect = Architect(self.net, self.net_copy, self.w_momentum, self.w_weight_decay,
                                   normal_none_penalty=normal_none_penalty, reduce_none_penalty=reduce_none_penalty)

    def training_step(self, batch, batch_idx):
        (trn_X, trn_y), (val_X, val_y) = batch

        # Single level optimization => use standard feedforward + default gradient descend
        if not self.bi_level_optimization:
            logits = self.net(trn_X)
            loss = self.net.criterion(logits, trn_y)

            # Loss += SUM[ none - mean(others) ]
            normal_alphas, reduce_alphas = self.net.alpha_weights()
            loss += self.normal_none_penalty * sum([(a[:, -1] - a[:, :-1].mean()).sum() for a in normal_alphas])
            loss += self.reduce_none_penalty * sum([(a[:, -1] - a[:, :-1].mean()).sum() for a in reduce_alphas])

            prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
            self.log('train_loss', loss)
            self.log('train_top1', prec1)
            self.log('train_top5', prec5)
            return loss

        w_optim, alpha_optim = self.optimizers()

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        w_lr = self.w_scheduler['scheduler'].get_last_lr()[-1]
        self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, w_lr, w_optim, amended=self.amended_hessian)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        logits = self.net(trn_X)
        loss = self.net.criterion(logits, trn_y)
        self.manual_backward(loss)

        # gradient clipping
        nn.utils.clip_grad_norm_(self.net.weights(), self.w_grad_clip)
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        self.log('train_loss', loss)
        self.log('train_top1', prec1)
        self.log('train_top5', prec5)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits = self.net(x)
            loss = self.net.criterion(logits, y)
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

        self.log('valid_loss', loss)
        self.log('valid_top1', prec1)
        self.log('valid_top5', prec5)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        w_optim = torch.optim.SGD(self.net.weights(), self.w_lr,
                                  momentum=self.w_momentum, weight_decay=self.w_weight_decay, nesterov=self.nesterov)
        self.w_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(w_optim, self.max_epochs, eta_min=self.w_lr_min),
            'interval': 'epoch',
        }

        alpha_optim = torch.optim.Adam(self.net.alphas(), self.alpha_lr,
                                       betas=(0.5, 0.999), weight_decay=self.alpha_weight_decay)
        self.alpha_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(alpha_optim, lr_lambda=lambda x: 1),
            'interval': 'epoch',
        }
        return [w_optim, alpha_optim], [self.w_scheduler, self.alpha_scheduler]

    def on_validation_epoch_end(self):
        epoch = self.trainer.current_epoch

        # Remove the worst connection
        if epoch > 0 and self.net.mask_alphas:
            # TODO: Pass how many connections to remove
            self.net.remove_worst_connection()

        # log genotype
        self.plot_genotype(genotype=self.net.genotype(algorithm='top-k'), name=f'top2-{epoch}')
        self.plot_genotype(genotype=self.net.genotype(algorithm='best'),  name=f'best-{epoch}')

        alpha_normal, alpha_reduce = self.net.alpha_weights()
        raw_alpha_normal, raw_alpha_reduce = self.net.raw_alphas()

        self.epoch2normal_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in alpha_normal]
        self.epoch2reduce_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in alpha_reduce]
        self.epoch2raw_normal_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in raw_alpha_normal]
        self.epoch2raw_reduce_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in raw_alpha_reduce]

        normal_fig = self.plot_alphas(self.epoch2normal_alphas)
        reduce_fig = self.plot_alphas(self.epoch2reduce_alphas)
        raw_normal_fig = self.plot_alphas(self.epoch2raw_normal_alphas)
        raw_reduce_fig = self.plot_alphas(self.epoch2raw_reduce_alphas)

        wandb.log({'Normal cell alpha change throughout epochs': normal_fig})
        wandb.log({'Reduce cell alpha change throughout epochs': reduce_fig})
        wandb.log({'Normal cell raw alpha values throughout epochs': raw_normal_fig})
        wandb.log({'Reduce cell raw alpha values throughout epochs': raw_reduce_fig})
        if epoch != 0:
            normal_diff = np.sum([np.sum(np.abs(cur - prev)) for cur, prev in zip(self.epoch2normal_alphas[epoch],
                                                                                  self.epoch2normal_alphas[epoch - 1])])
            reduce_diff = np.sum([np.sum(np.abs(cur - prev)) for cur, prev in zip(self.epoch2reduce_alphas[epoch],
                                                                                  self.epoch2reduce_alphas[epoch - 1])])
            self.log('normal_diff', normal_diff)
            self.log('reduce_diff', reduce_diff)

        print(f'\nSparsity: {self.net.sparsity}')
        print("####### MASK #######")
        print("\n# Alpha - mask")
        for mask in self.net.normal_mask:
            print(mask)
        print("\n# Alpha - reduce")
        for mask in self.net.reduce_mask:
            print(mask)

        print("####### ALPHA #######")
        print("\n# Alpha - normal")
        for alpha in alpha_normal:
            print(alpha)
        print("\n# Alpha - reduce")
        for alpha in alpha_reduce:
            print(alpha)

    def plot_genotype(self, genotype: gt.Genotype, name: str):
        gt.plot(genotype.normal, self.image_log_path / f'normal-{name}')
        gt.plot(genotype.reduce, self.image_log_path / f'reduction-{name}')

        log_name = name.split('-')[0]
        wandb.log({f'normal-{log_name}-cell': wandb.Image(str(self.image_log_path / f'normal-{name}.png'))})
        wandb.log({f'reduction-{log_name}-cell': wandb.Image(str(self.image_log_path / f'reduction-{name}.png'))})
        print(f'\nGenotype {name}:', genotype)

    def plot_alphas(self, epoch2alphas: Dict):
        epochs = len(epoch2alphas)

        fig = make_subplots(rows=self.net.n_nodes, cols=self.net.n_nodes + 1,
                            subplot_titles=[f'{node1} ➜ {node2}' if node1 < node2 else ''
                                            for node2 in range(2, self.net.n_nodes + 2)
                                            for node1 in range(self.net.n_nodes + 1)])

        for node1 in range(2, self.net.n_nodes + 2):
            for node2 in range(node1):
                for connection_id, connection_name in enumerate(self.net.primitives):
                    fig.add_trace(
                        go.Scatter(x=list(range(epochs)),
                                   y=[epoch2alphas[epoch][node1 - 2][node2][connection_id] for epoch in range(epochs)],
                                   name=connection_name,
                                   legendgroup=connection_name,
                                   showlegend=node1 == 2 and node2 == 0,
                                   line=dict(color=px.colors.qualitative.Plotly[connection_id])),
                        row=node1 - 1,
                        col=node2 + 1,
                    )

        fig.update_layout(height=1000, width=1000)
        return fig
