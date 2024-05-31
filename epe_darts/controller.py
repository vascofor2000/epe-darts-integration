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
                 max_epochs: int = 50, debugging: bool = False, genotype_to_retain: str = 'top2', bdarts: bool = False,
                 topk_on_weights: bool = False, topk_on_alphas: bool = False, topk_on_virtualstep: bool = False,
                 topk_to_remove: bool = False, linear_temperature_increase: bool = False, linear_temperature_max: int = 1,
                 hd_on_weights: bool = False, hd_on_alphas: bool = False, hd_on_virtualstep: bool = False, hd_start: int = 0,
                 dal_on_alphas: bool = False, dal_on_weights: bool = False, dal_starting_epoch: int = 0, dal_factor: float = 1., dal_factor_max: int = 1, linear_dal_factor_increase: bool = False,
                 dropout_on_weights: bool = False, dropout_on_alphas: bool = False, dropout_on_virtualstep: bool = False,
                 variable_temperature: bool = False, zero_variable_temperature: bool = False, epsilon: int = 6):
        super().__init__()
        self.save_hyperparameters('image_log_path', 'bi_level_optimization',
                                  'w_lr', 'w_momentum', 'w_weight_decay', 'w_lr_min', 'w_grad_clip', 'nesterov',
                                  'alpha_lr', 'alpha_weight_decay', 'amended_hessian',
                                  'normal_none_penalty', 'reduce_none_penalty', 'max_epochs')

        self.automatic_optimization = not bi_level_optimization
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

        self.genotype_to_retain = genotype_to_retain
        self.bdarts = bdarts
        self.debugging = debugging
        self.topk_on_weights = topk_on_weights
        self.hd_start = hd_start
        if hd_start != 0:
            self.hd_on_weights = False
        else:
            self.hd_on_weights = hd_on_weights
        self.topk_to_remove = topk_to_remove
        self.linear_temperature_increase = linear_temperature_increase
        self.linear_temperature_max = linear_temperature_max
        self.dal_on_alphas = dal_on_alphas
        self.dal_on_weights = dal_on_weights
        self.dal_starting_epoch = dal_starting_epoch
        self.dal_factor = dal_factor
        self.linear_dal_factor_increase = linear_dal_factor_increase
        self.dal_factor_max = dal_factor_max
        self.dropout_on_weights = dropout_on_weights
        self.variable_temperature = variable_temperature
        self.zero_variable_temperature = zero_variable_temperature
        self.epsilon = epsilon


        self.epoch2normal_alphas: Dict = {}
        self.epoch2reduce_alphas: Dict = {}
        self.epoch2raw_normal_alphas: Dict = {}
        self.epoch2raw_reduce_alphas: Dict = {}
        self.epoch2topk_normal_alphas: Dict = {}
        self.epoch2topk_reduce_alphas: Dict = {}

        self.net: nn.Module = net
        self.net_copy: nn.Module = copy.deepcopy(net)
        self.architect = Architect(self.net, self.net_copy, self.w_momentum, self.w_weight_decay,
                                   normal_none_penalty=normal_none_penalty, reduce_none_penalty=reduce_none_penalty, 
                                   topk_on_alphas=topk_on_alphas, topk_on_virtualstep=topk_on_virtualstep, 
                                   hd_on_alphas=hd_on_alphas, hd_on_virtualstep=hd_on_virtualstep, debugging = debugging,
                                   dropout_on_alphas=dropout_on_alphas, dropout_on_virtualstep=dropout_on_virtualstep)

    def training_step(self, batch, batch_idx, optimizer_idx):
        (trn_X, trn_y), (val_X, val_y) = batch
        epoch = self.trainer.current_epoch
        n_batches = self.trainer.num_training_batches
        if self.linear_temperature_increase:
            self.net.set_temperature(self.linear_temperature_max*epoch/self.max_epochs)
        if self.linear_dal_factor_increase:
            self.dal_factor = self.dal_factor_max*epoch/self.max_epochs
        if self.hd_start != 0 and epoch == self.hd_start:
            self.hd_on_weights = True
                    
        
        # Single level optimization => use standard feedforward + default gradient descend
        if not self.bi_level_optimization:
            logits = self.net(trn_X, self.topk_on_weights, self.hd_on_weights)
            loss = self.net.criterion(logits, trn_y)

            # Loss += SUM[ none - mean(others) ]
            normal_alphas, reduce_alphas = self.net.alpha_weights()
            loss += self.normal_none_penalty * sum([(a[:, -1] - a[:, :-1].mean()).sum() for a in normal_alphas])
            loss += self.reduce_none_penalty * sum([(a[:, -1] - a[:, :-1].mean()).sum() for a in reduce_alphas])

            #adding beta_loss
            if self.bdarts:
                reg_coef = 0 + 50*epoch/100
                beta_loss = self.architect.beta_loss()
                loss += reg_coef*beta_loss

            prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
            self.log('train_loss', loss)
            self.log('train_top1', prec1)
            self.log('train_top5', prec5)
            return loss
        
        w_optim, alpha_optim = self.optimizers()

        # phase 2. architect step (alpha)
        alpha_optim.zero_grad()
        # Single level optimization => use standard feedforward + default gradient descend
        '''if not self.bi_level_optimization:
            self.architect.rolled_backward(val_X, val_y)

        else:'''
        w_lr = self.w_scheduler['scheduler'].get_last_lr()[-1]
        if self.dal_on_alphas and epoch >= self.dal_starting_epoch:
            self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, w_lr, w_optim, epoch=epoch, amended=self.amended_hessian, bdarts=self.bdarts, dal=True, dal_factor=self.dal_factor)
        else:
            self.architect.unrolled_backward(trn_X, trn_y, val_X, val_y, w_lr, w_optim, epoch=epoch, amended=self.amended_hessian, bdarts=self.bdarts, dal=False)
        alpha_optim.step()

        # phase 1. child network step (w)
        w_optim.zero_grad()
        #print("goes on weights update")
        if self.variable_temperature:
            with torch.no_grad():
                logits_without_dropout = self.net(trn_X, self.topk_on_weights, self.hd_on_weights, False)
                loss_without_dropout = self.net.criterion(logits_without_dropout, trn_y)
                logits_with_dropout = self.net(trn_X, self.topk_on_weights, self.hd_on_weights, True)
                loss_with_dropout = self.net.criterion(logits_with_dropout, trn_y)
                amount_to_update = self.linear_temperature_max/(self.max_epochs*n_batches)
            if loss_without_dropout > loss_with_dropout:
                #print("dropout não ajudou, caso normal")
                self.net.update_temperature(amount_to_update)
            else:
                #print("dropout ajudou, hora de mudar")
                if self.zero_variable_temperature:
                    self.net.update_temperature(0)
                else:
                    self.net.update_temperature(-amount_to_update)

            #print(f"temperature now is {self.net.get_temperature()}")
        self.log('temperature', self.net.get_temperature())
        #else:
        logits = self.net(trn_X, self.topk_on_weights, self.hd_on_weights, self.dropout_on_weights)
        loss = self.net.criterion(logits, trn_y)
        if self.dal_on_weights and epoch >= self.dal_starting_epoch:
            loss += self.dal_factor * self.net.discretization_additional_loss()
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
            logits = self.net(x, self.topk_on_weights, self.hd_on_weights, self.dropout_on_weights)
            loss = self.net.criterion(logits, y)
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

        self.log('valid_loss', loss)
        self.log('valid_top1', prec1)
        self.log('valid_top5', prec5)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        # Single level optimization => no need for optimizers
        #if not self.bi_level_optimization:
        #    return [], []

        # Bi-level optimization will have two distinct optimizers
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
            self.net.remove_worst_connection(self.topk_to_remove, 10 ** -self.epsilon)

        # log genotype
        self.plot_genotype(genotype=self.net.genotype(algorithm='top-k'), name=f'top2-{epoch}')
        self.plot_genotype(genotype=self.net.genotype(algorithm='top-k_with_none'), name=f'top2_with_none-{epoch}')
        self.plot_genotype(genotype=self.net.genotype(algorithm='best'),  name=f'best-{epoch}')
        self.plot_genotype(genotype=self.net.genotype(algorithm='all'),  name=f'all-{epoch}')

        alpha_normal, alpha_reduce = self.net.alpha_weights()
        raw_alpha_normal, raw_alpha_reduce = self.net.raw_alphas()
        topk_alpha_normal, topk_alpha_reduce = self.net.topk(alpha_normal, alpha_reduce)
        
        #if self.debugging:
        #    gt.parse(alpha_normal, "darts", 2, "topk-node")

        self.epoch2normal_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in alpha_normal]
        self.epoch2reduce_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in alpha_reduce]
        self.epoch2raw_normal_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in raw_alpha_normal]
        self.epoch2raw_reduce_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in raw_alpha_reduce]
        self.epoch2topk_normal_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in topk_alpha_normal]
        self.epoch2topk_reduce_alphas[epoch] = [alpha.detach().cpu().numpy() for alpha in topk_alpha_reduce]

        normal_fig = self.plot_alphas(self.epoch2normal_alphas)
        reduce_fig = self.plot_alphas(self.epoch2reduce_alphas)
        raw_normal_fig = self.plot_alphas(self.epoch2raw_normal_alphas)
        raw_reduce_fig = self.plot_alphas(self.epoch2raw_reduce_alphas)
        topk_normal_fig = self.plot_alphas(self.epoch2topk_normal_alphas)
        topk_reduce_fig = self.plot_alphas(self.epoch2topk_reduce_alphas)

        wandb.log({'Normal cell alpha change throughout epochs': normal_fig})
        wandb.log({'Reduce cell alpha change throughout epochs': reduce_fig})
        wandb.log({'Normal cell raw alpha values throughout epochs': raw_normal_fig})
        wandb.log({'Reduce cell raw alpha values throughout epochs': raw_reduce_fig})
        wandb.log({'Normal cell topk alpha values throughout epochs': topk_normal_fig})
        wandb.log({'Reduce cell topk alpha values throughout epochs': topk_reduce_fig})

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

        print("####### ALPHA topk #######")
        print(f'temperature is {self.net.get_temperature()}')
        print("\n# Alpha topk - normal")
        alpha_normal, alpha_reduce = self.net.topk(alpha_normal, alpha_reduce)
        for alpha in alpha_normal:
            print(alpha)
        print("\n# Alpha topk - reduce")
        for alpha in alpha_reduce:
            print(alpha)

    def plot_genotype(self, genotype: gt.Genotype, name: str):
        gt.plot(genotype.normal, self.image_log_path / f'normal-{name}')
        gt.plot(genotype.reduce, self.image_log_path / f'reduction-{name}')

        log_name = name.split('-')[0]
        wandb.log({f'normal-{log_name}-cell': wandb.Image(str(self.image_log_path / f'normal-{name}.png'))})
        wandb.log({f'reduction-{log_name}-cell': wandb.Image(str(self.image_log_path / f'reduction-{name}.png'))})
        if self.genotype_to_retain == "top2" and name == "top2-49":
            print(f'\nGenotype {name}:@@@@@{genotype}@@@@@')
        elif self.genotype_to_retain == "best" and name == "best-49":
            print(f'\nGenotype {name}:@@@@@{genotype}@@@@@')
        elif self.genotype_to_retain == "all" and name == "all-49":
            print(f'\nGenotype {name}:@@@@@{genotype}@@@@@')   
        elif self.genotype_to_retain == "top2_with_none" and name == "top2_with_none-49":
            print(f'\nGenotype {name}:@@@@@{genotype}@@@@@')      
        else:
            print(f'\nGenotype {name}:', genotype)
        wandb.log({f'genotype {log_name}': str(genotype)})

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
