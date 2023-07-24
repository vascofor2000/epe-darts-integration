""" CNN for network augmentation """
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Callback, Trainer

from epe_darts import genotypes as gt, ops, utils


class AugmentCell(nn.Module):
    """ Cell for augmentation
    Each edge is discrete.
    """
    def __init__(self, genotype, prev_prev_channels, prev_channels, channels, reduction_p, reduction):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = len(genotype.normal)

        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(prev_prev_channels, channels)
        else:
            self.preproc0 = ops.StdConv(prev_prev_channels, channels, 1, 1, 0)
        self.preproc1 = ops.StdConv(prev_channels, channels, 1, 1, 0)

        # generate dag
        if reduction:
            gene = genotype.reduce
            self.concat = genotype.reduce_concat
        else:
            gene = genotype.normal
            self.concat = genotype.normal_concat

        self.dag = gt.to_dag(channels, gene, reduction)

    def forward(self, s0, s1):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges in self.dag:
            s_cur = sum(op(states[op.s_idx]) for op in edges)
            states.append(s_cur)

        s_out = torch.cat([states[i] for i in self.concat], dim=1)

        return s_out


class AuxiliaryHead(nn.Module):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, channels, n_classes):
        """ assuming input size 7x7 or 8x8 """
        assert input_size in [7, 8]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=input_size-5, padding=0, count_include_pad=False), # 2x2 out
            nn.Conv2d(channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, kernel_size=2, bias=False), # 1x1 out
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1) # flatten
        logits = self.linear(out)
        return logits


class AugmentCNN(LightningModule):
    """ Augmented CNN model """
    def __init__(self, input_size, input_channels, init_channels, n_classes, n_layers, auxiliary_weight, genotype, stem_multiplier=3,
                 lr: float = 0.025, momentum: float = 0.9, weight_decay: float = 3e-4, max_epochs: int = 600):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            input_channels: # of input channels
            init_channels: # of starting model channels
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr: float = lr
        self.momentum: float = momentum
        self.weight_decay: float = weight_decay
        self.max_epochs: int = max_epochs
        self.criterion = nn.CrossEntropyLoss()

        self.input_channels = input_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.genotype = genotype
        # aux head position
        self.auxiliary_weight = auxiliary_weight
        self.aux_pos = 2*n_layers//3 if auxiliary_weight else -1

        cur_channels = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, cur_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(cur_channels)
        )

        prev_prev_channels, prev_channels, cur_channels = cur_channels, cur_channels, init_channels

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            if i in [n_layers//3, 2*n_layers//3]:
                cur_channels *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(genotype, prev_prev_channels, prev_channels, cur_channels, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            cur_out_channels = cur_channels * len(cell.concat)
            prev_prev_channels, prev_channels = prev_channels, cur_out_channels

            if i == self.aux_pos:
                # [!] this auxiliary head is ignored in computing parameter size
                #     by the name 'aux_head'
                self.aux_head = AuxiliaryHead(input_size//4, prev_channels, n_classes)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(prev_channels, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits, aux_logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, aux_logits = self(x)
        loss = self.criterion(logits, y)
        if self.auxiliary_weight:
            loss += self.auxiliary_weight * self.criterion(aux_logits, y)

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        self.log('train_loss', loss)
        self.log('train_top1', prec1)
        self.log('train_top5', prec5)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            logits, _ = self(x)
            loss = self.criterion(logits, y)
            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))

        self.log('valid_loss', loss)
        self.log('valid_top1', prec1)
        self.log('valid_top5', prec5)

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


class DropPathCallback(Callback):
    def __init__(self, max_epochs: int, drop_path_prob: float):
        self.max_epochs: int = max_epochs
        self.drop_path_prob: float = drop_path_prob

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        epoch = trainer.current_epoch
        drop_prob = self.drop_path_prob * epoch / self.max_epochs
        model = trainer.model
        assert isinstance(model, AugmentCNN)
        model.drop_path_prob(drop_prob)
