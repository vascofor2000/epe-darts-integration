""" CNN for architecture search """
import copy
from copy import deepcopy
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
#from torchviz import make_dot

from epe_darts import genotypes as gt, ops
from epe_darts.functional import softmax, sigmoid, entmax, TopK
from epe_darts.ops import ResNetBasicblock, NAS201SearchCell


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

        if search_space == "nas-bench-201":
            C = 16
            N = 5
            max_nodes = 4
            affine = False
            track_running_stats = False
            self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(C)
            )

            layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
            layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

            C_prev, num_edge, edge2index = C, None, None
            self.cells = nn.ModuleList()
            for index, (C_curr, reduction) in enumerate(
                zip(layer_channels, layer_reductions)
            ):
                if reduction:
                    cell = ResNetBasicblock(C_prev, C_curr, 2)
                else:
                    cell = NAS201SearchCell(
                        C_prev,
                        C_curr,
                        1,
                        max_nodes,
                        search_space,
                        affine,
                        track_running_stats,
                    )
                    if num_edge is None:
                        num_edge, edge2index = cell.num_edges, cell.edge2index
                    else:
                        assert (
                            num_edge == cell.num_edges and edge2index == cell.edge2index
                        ), "invalid {:} vs. {:}.".format(num_edge, cell.num_edges)
                self.cells.append(cell)
                C_prev = cell.out_dim
            self.op_names = deepcopy(search_space)
            self._Layer = len(self.cells)
            self.edge2index = edge2index
            self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
            self.global_pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(C_prev, n_classes)


        else:    
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
        if self.search_space == "nas-bench-201":
            feature = self.stem(x)
            for i, cell in enumerate(self.cells):
                if isinstance(cell, NAS201SearchCell):
                    feature = cell(feature, weights_normal)
                else:
                    feature = cell(feature)

            out = self.lastact(feature)
            out = self.global_pooling(out)
            out = out.view(out.size(0), -1)
            logits = self.classifier(out)

            return logits

        else:

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
                 #topk=False, 
                 justtopk: bool = False,
                 topk_k_node:int = 2, topk_temperature_node: int = 1, 
                 topk_k_edge:int = 1, topk_temperature_edge: int = 1, option:int = 1,
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

        #for the topk
        #self.topk_flag = topk
        self.apply_topk = TopK.apply
        self.topk_k_node = topk_k_node
        self.topk_k_edge = topk_k_edge
        self.topk_temperature_node = topk_temperature_node
        self.topk_temperature_edge = topk_temperature_edge
        self.justtopk = justtopk
        self.option = option

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
        
        self.dropout = nn.Dropout(0.15)

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
        if weights_normal[0][0, 0].isnan():
            print(f"alpha_normal after applying softmax {[alphas for alphas in self.alpha_normal]}")
            print(f"alpha_reduce after applying softmax {[alphas for alphas in self.alpha_reduce]}")
            return
        
        return weights_normal, weights_reduce

    def raw_alphas(self):
        normal = [alpha * mask.float() for alpha, mask in zip(self.alpha_normal, self.normal_mask)]
        reduce = [alpha * mask.float() for alpha, mask in zip(self.alpha_reduce, self.reduce_mask)]
        return normal, reduce

    def set_temperature(self, new_temperature):
        self.topk_temperature_edge = new_temperature
        self.topk_temperature_node = new_temperature

    def update_temperature(self, amount_to_update):
        #print(amount_to_update)
        self.topk_temperature_edge += amount_to_update
        self.topk_temperature_node += amount_to_update
        if self.topk_temperature_edge < 1:
            self.topk_temperature_edge = 1.0
            self.topk_temperature_node = 1.0

    def get_temperature(self):
        return self.topk_temperature_edge

    def hard_discretization(self, weights_normal, weights_reduce):
        hd_weights_normal = [torch.zeros_like(weights) for weights in weights_normal]
        hd_weights_reduce = [torch.zeros_like(weights) for weights in weights_reduce]

        for idx, edges in enumerate(weights_normal):
            edge_max, primitive_indices = torch.topk(edges, 1)
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), 2)
            for edge_idx in topk_edge_indices:
                hd_weights_normal[idx][edge_idx, primitive_indices[edge_idx]] = 1
        
        for idx, edges in enumerate(weights_reduce):
            edge_max, primitive_indices = torch.topk(edges, 1)
            topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), 2)
            for edge_idx in topk_edge_indices:
                hd_weights_reduce[idx][edge_idx, primitive_indices[edge_idx]] = 1
            
        return hd_weights_normal, hd_weights_reduce

    def topk(self, weights_normal, weights_reduce):
        topk_weights_normal = weights_normal
        topk_weights_reduce = weights_reduce

        #first apply for edge
        if self.topk_k_edge != 0:
            topk_weights_normal = [self.apply_topk(weights, self.topk_k_edge, self.topk_temperature_edge) for weights in weights_normal]
            topk_weights_reduce = [self.apply_topk(weights, self.topk_k_edge, self.topk_temperature_edge) for weights in weights_reduce]
            #topk_weights_reduce = []
            #for weights in weights_reduce:
                #topk_weights = self.apply_topk(weights, self.topk_k_edge, self.topk_temperature_edge)
                #topk_weights_reduce.append(topk_weights)

        #then apply for node
        if self.topk_k_node != 0:
            n_edges_normal = [weights.shape[0] for weights in weights_normal]
            n_edges_reduce = [weights.shape[0] for weights in weights_reduce]
                
            if self.option == 1:
                topk_node_weights_normal = [self.apply_topk(weights.view(1, -1), self.topk_k_node, self.topk_temperature_node).view(n, -1) for weights, n in zip(weights_normal, n_edges_normal)]
                topk_node_weights_reduce = [self.apply_topk(weights.view(1, -1), self.topk_k_node, self.topk_temperature_node).view(n, -1) for weights, n in zip(weights_reduce, n_edges_reduce)]
            elif self.option == 2:
                topk_weights_normal = [self.apply_topk(weights.view(1, -1), self.topk_k_node, self.topk_temperature_node).view(n, -1) for weights, n in zip(topk_weights_normal, n_edges_normal)]
                topk_weights_reduce = [self.apply_topk(weights.view(1, -1), self.topk_k_node, self.topk_temperature_node).view(n, -1) for weights, n in zip(topk_weights_reduce, n_edges_reduce)]


        if not self.justtopk:
            if self.option == 2:
                topk_weights_normal = [weights * topk_weights for weights, topk_weights in zip(weights_normal, topk_weights_normal)]
                topk_weights_reduce = [weights * topk_weights for weights, topk_weights in zip(weights_reduce, topk_weights_reduce)]
            elif self.option == 1:
                topk_weights_normal = [weights * topk_weights * topk_node_weights for weights, topk_weights, topk_node_weights in zip(weights_normal, topk_weights_normal, topk_node_weights_normal)]
                topk_weights_reduce = [weights * topk_weights * topk_node_weights for weights, topk_weights, topk_node_weights in zip(weights_reduce, topk_weights_reduce, topk_node_weights_reduce)]
        return topk_weights_normal, topk_weights_reduce

    def forward(self, x, use_topk, use_hd, use_dropout):
        weights_normal, weights_reduce = self.alpha_weights()
        if use_dropout:
            #print(f"weights before the dropout {weights_normal}")
            weights_normal = [self.dropout(weights) for weights in weights_normal]
            #print(f"weights after the dropout {weights_normal}")
            weights_reduce = [self.dropout(weights) for weights in weights_reduce]
        if use_topk:
            weights_normal, weights_reduce = self.topk(weights_normal, weights_reduce)
        elif use_hd:
            weights_normal, weights_reduce = self.hard_discretization(weights_normal, weights_reduce)
        #print(f"dos weights no fowrdard da net maior {weights_normal} e agr do reduce {weights_reduce}")
        return self.net(x, weights_normal, weights_reduce)

    def discretization_additional_loss(self):
        weights_normal, weights_reduce = self.alpha_weights()
        discrete_weights_normal, discrete_weights_reduce = self.hard_discretization(weights_normal, weights_reduce)
        extra_loss = nn.MSELoss()
        losses = []
        for node, discrete_node in zip(weights_normal, discrete_weights_normal):
            losses.append(extra_loss(node, discrete_node))
        return sum(losses)/len(losses)

    def loss(self, x, y, use_topk, use_hd, use_dropout):
        logits = self.forward(x, use_topk, use_hd, use_dropout)
        #print("here goes the graph")
        #graph = make_dot(logits, params=dict(self.named_weights()))
        #graph.view()
        #return
        return self.criterion(logits, y)

    def remove_worst_connection(self, use_topk, epsilon: float = 1e-6) -> None:
        #print("---------------------------------------------------------------remove_worst_connection is called")
        if not self.mask_alphas:
            #print("and exception will be raised")
            raise ValueError('Cannot remove a connection when the alphas are set to be not masked')
        #print("-------------------------------------------------------------------------------------------------------------------------------exception was not raised")
        weights_normal, weights_reduce = self.alpha_weights()
        if use_topk:
            weights_normal, weights_reduce = self.topk(weights_normal, weights_reduce)
        print(f"epsilon is {epsilon}")
        def remove(alphas: List[torch.Tensor], masks: nn.ParameterList, strategy: str):
            if strategy == 'smallest':
                lowest_idx = None
                for node, (alpha, mask) in enumerate(zip(alphas, masks)):
                    alpha_cp = copy.deepcopy(alpha)
                    alpha_cp[~mask] = 1000
                    vals, cols = alpha_cp.min(dim=-1)
                    val, row = vals.min(dim=-1)
                    col = cols[row]
                    if lowest_idx is None or alphas[lowest_idx[0]][lowest_idx[1]] > val:
                        lowest_idx = node, (row, col)

                masks[lowest_idx[0]][lowest_idx[1]] = False
            elif strategy == 'zero':
                for node, alpha in enumerate(alphas):
                    n_rows, n_cols = alpha.shape
                    for row in range(n_rows):
                        for col in range(n_cols):
                            if abs(alphas[node][(row, col)]) < epsilon:
                                masks[node][(row, col)] = False

                #if abs(alphas[lowest_idx[0]][lowest_idx[1]]) < epsilon:
                #    masks[lowest_idx[0]][lowest_idx[1]] = False
                else:
                    print('Not pruning any weights as none of them is 0')
            else:
                raise ValueError(f'Pruning strategy `{strategy}` is not implemented')

        remove(weights_normal, self.normal_mask, self.prune_strategy)
        remove(weights_reduce, self.reduce_mask, self.prune_strategy)

    def genotype(self, algorithm: str = 'top-k'):
        if algorithm == 'all':
            weights_normal, weights_reduce = self.raw_alphas()
        else:
            weights_normal, weights_reduce = self.alpha_weights()

        gene_normal, concat_normal = gt.parse(weights_normal, search_space=self.search_space, k=2, algorithm=algorithm)
        #concat_normal = gt.concat_for_gene(gene_normal)
        gene_reduce, concat_reduce = gt.parse(weights_reduce, search_space=self.search_space, k=2, algorithm=algorithm)
        #concat_reduce = gt.concat_for_gene(gene_reduce)

        #concat = range(2, 2 + self.n_nodes)  # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat_normal,
                           reduce=gene_reduce, reduce_concat=concat_reduce)

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
