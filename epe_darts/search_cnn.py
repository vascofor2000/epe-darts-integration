""" CNN for architecture search """
import copy
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn

from epe_darts import genotypes as gt, ops
from epe_darts.functional import softmax, sigmoid, entmax


class SearchCell(nn.Module):
    """ Cell for search
    Each edge is mixed and continuous relaxed.
    """
    def __init__(self, n_nodes,
                 prev_prev_channels, prev_channels, current_channels,
                 reduction_p, reduction,
                 search_space):
        super().__init__()
        self.reduction = reduction
        self.n_nodes = n_nodes
        self.search_space = search_space

        # If previous cell is reduction cell, current input size does not match with
        # output size of cell[k-2]. So the output[k-2] should be reduced by preprocessing.
        if reduction_p:
            self.preproc0 = ops.FactorizedReduce(prev_prev_channels, current_channels, affine=False)
        else:
            self.preproc0 = ops.StdConv(prev_prev_channels, current_channels, 1, 1, 0, affine=False)
        self.preproc1 = ops.StdConv(prev_channels, current_channels, 1, 1, 0, affine=False)

        # generate dag
        self.dag = nn.ModuleList()
        for i in range(self.n_nodes):
            self.dag.append(nn.ModuleList())
            for j in range(2 + i):  # include 2 input nodes
                # reduction should be used only for input node
                stride = 2 if reduction and j < 2 else 1
                op = ops.MixedOp(current_channels, stride, self.search_space)
                self.dag[i].append(op)

    def forward(self, s0, s1, w_dag):
        s0 = self.preproc0(s0)
        s1 = self.preproc1(s1)

        states = [s0, s1]
        for edges, w_list in zip(self.dag, w_dag):
            s_cur = sum(edges[i](s, w) for i, (s, w) in enumerate(zip(states, w_list)))
            states.append(s_cur)

        s_out = torch.cat(states[2:], dim=1)
        return s_out


class SearchCNN(nn.Module):
    """ Search CNN model """
    def __init__(self, input_channels, init_channels, n_classes, n_layers, n_nodes=4, stem_multiplier=3,
                 search_space: str = 'darts'):
        """
        Args:
            input_channels: # of input channels
            init_channels: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.input_channels = input_channels
        self.init_channels = init_channels
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.search_space = search_space

        current_channels = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, current_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(current_channels)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] prev_prev_channels and prev_channels is output channel size, but current_channels is input channel size.
        prev_prev_channels, prev_channels, current_channels = current_channels, current_channels, init_channels

        self.cells = nn.ModuleList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers // 3, 2 * n_layers // 3]:
                current_channels *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, prev_prev_channels, prev_channels, current_channels, reduction_p, reduction,
                              search_space=search_space)
            reduction_p = reduction
            self.cells.append(cell)
            cur_out_channels = current_channels * n_nodes
            prev_prev_channels, prev_channels = prev_channels, cur_out_channels

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(prev_channels, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)

        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = out.view(out.size(0), -1)  # flatten
        logits = self.linear(out)
        return logits


class SearchCNNController(pl.LightningModule):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, input_channels, init_channels, n_classes, n_layers, n_nodes=4, stem_multiplier=3,
                 search_space='darts',
                 sparsity=1, prune_strategy='smallest', alpha_normal=None, alpha_reduce=None, mask_alphas=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.search_space = search_space
        self.primitives = ops.SEARCH_SPACE2OPS[self.search_space]
        self.n_nodes: int = n_nodes
        self.criterion = nn.CrossEntropyLoss()
        self.sparsity = sparsity
        self.prune_strategy = prune_strategy
        self.mask_alphas = mask_alphas

        # initialize alphas
        n_ops = len(self.primitives)
        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        self.normal_mask = nn.ParameterList()
        self.reduce_mask = nn.ParameterList()

        if alpha_normal is not None and alpha_reduce is not None:
            # print('Using provided alphas...')
            for normal, reduce in zip(alpha_normal, alpha_reduce):
                self.alpha_normal.append(nn.Parameter(normal))
                self.alpha_reduce.append(nn.Parameter(reduce))
                self.normal_mask.append(nn.Parameter(torch.ones_like(normal, dtype=torch.bool), requires_grad=False))
                self.reduce_mask.append(nn.Parameter(torch.ones_like(reduce, dtype=torch.bool), requires_grad=False))
        else:
            for i in range(n_nodes):
                normal = nn.Parameter(torch.randn(i + 2, n_ops) * 1e-3)
                reduce = nn.Parameter(torch.randn(i + 2, n_ops) * 1e-3)
                self.alpha_normal.append(normal)
                self.alpha_reduce.append(reduce)
                self.normal_mask.append(nn.Parameter(torch.ones_like(normal, dtype=torch.bool), requires_grad=False))
                self.reduce_mask.append(nn.Parameter(torch.ones_like(reduce, dtype=torch.bool), requires_grad=False))

        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))

        self.net = SearchCNN(input_channels, init_channels, n_classes, n_layers, n_nodes, stem_multiplier,
                             search_space=search_space)

    def alpha_weights(self):
        if self.sparsity == -1:
            weights_normal = [sigmoid(alpha, mask) for alpha, mask in zip(self.alpha_normal, self.normal_mask)]
            weights_reduce = [sigmoid(alpha, mask) for alpha, mask in zip(self.alpha_reduce, self.reduce_mask)]
        elif self.sparsity == 1:
            weights_normal = [softmax(alpha, mask, dim=-1) for alpha, mask in zip(self.alpha_normal, self.normal_mask)]
            weights_reduce = [softmax(alpha, mask, dim=-1) for alpha, mask in zip(self.alpha_reduce, self.reduce_mask)]
        else:
            weights_normal = [entmax(alpha, mask, dim=-1, alpha=self.sparsity) for alpha, mask in zip(self.alpha_normal, self.normal_mask)]
            weights_reduce = [entmax(alpha, mask, dim=-1, alpha=self.sparsity) for alpha, mask in zip(self.alpha_reduce, self.reduce_mask)]
        return weights_normal, weights_reduce

    def raw_alphas(self):
        normal = [alpha * mask.float() for alpha, mask in zip(self.alpha_normal, self.normal_mask)]
        reduce = [alpha * mask.float() for alpha, mask in zip(self.alpha_reduce, self.reduce_mask)]
        return normal, reduce

    def forward(self, x):
        weights_normal, weights_reduce = self.alpha_weights()
        return self.net(x, weights_normal, weights_reduce)

    def loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)

    def remove_worst_connection(self, epsilon: float = 1e-6) -> None:
        if not self.mask_alphas:
            raise ValueError('Cannot remove a connection when the alphas are set to be not masked')
        weights_normal, weights_reduce = self.alpha_weights()

        def remove(alphas: List[torch.Tensor], masks: nn.ParameterList, strategy: str):
            lowest_idx = None
            for i, (alpha, mask) in enumerate(zip(alphas, masks)):
                alpha_cp = copy.deepcopy(alpha)
                alpha_cp[~mask] = 1000
                vals, cols = alpha_cp.min(dim=-1)
                val, row = vals.min(dim=-1)
                col = cols[row]
                if lowest_idx is None or alphas[lowest_idx[0]][lowest_idx[1]] > val:
                    lowest_idx = i, (row, col)

            if strategy == 'smallest':
                masks[lowest_idx[0]][lowest_idx[1]] = False
            elif strategy == 'zero':
                if abs(alphas[lowest_idx[0]][lowest_idx[1]]) < epsilon:
                    masks[lowest_idx[0]][lowest_idx[1]] = False
                else:
                    print('Not pruning any weights as none of them is 0')
            else:
                raise ValueError(f'Pruning strategy `{strategy}` is not implemented')

        remove(weights_normal, self.normal_mask, self.prune_strategy)
        remove(weights_reduce, self.reduce_mask, self.prune_strategy)

    def genotype(self, algorithm: str = 'top-k'):
        weights_normal, weights_reduce = self.alpha_weights()

        gene_normal = gt.parse(weights_normal, search_space=self.search_space, k=2, algorithm=algorithm)
        gene_reduce = gt.parse(weights_reduce, search_space=self.search_space, k=2, algorithm=algorithm)
        concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        return self.net.parameters()

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
