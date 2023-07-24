""" Genotypes
    - Genotype: normal/reduce gene + normal/reduce cell output connection (concat)
    - gene: discrete ops information (w/o output connection)
    - dag: real ops (can be mixed or discrete, but Genotype has only discrete information itself)
"""
from collections import namedtuple
from typing import List, Tuple, Optional, Iterable

import torch
import torch.nn as nn
from graphviz import Digraph

from epe_darts import ops
from epe_darts.utils import PathLike

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def to_dag(in_channels: int, gene: List[List[Tuple[str, int]]], reduction):
    """ generate discrete ops from gene """
    dag = nn.ModuleList()
    for edges in gene:
        row = nn.ModuleList()
        for op_name, s_idx in edges:
            # reduction cell & from input nodes => stride = 2
            stride = 2 if reduction and s_idx < 2 else 1
            op = ops.OPS[op_name](in_channels, stride, True)
            if not isinstance(op, ops.Identity):  # Identity does not use drop path
                op = nn.Sequential(
                    op,
                    ops.DropPath_()
                )
            op.s_idx = s_idx
            row.append(op)
        dag.append(row)

    return dag


def from_str(s: str) -> Genotype:
    """ generate genotype from string
    e.g. "Genotype(
            normal=[[('sep_conv_3x3', 0), ('sep_conv_3x3', 1)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('sep_conv_3x3', 2)],
                    [('sep_conv_3x3', 1), ('dil_conv_3x3', 4)]],
            normal_concat=range(2, 6),
            reduce=[[('max_pool_3x3', 0), ('max_pool_3x3', 1)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)],
                    [('max_pool_3x3', 0), ('skip_connect', 2)]],
            reduce_concat=range(2, 6))"
    """
    return eval(s)


def parse(alpha: Iterable[nn.Parameter], search_space: str, k: int = 2,
          algorithm: str = 'top-k') -> List[List[Tuple[str, int]]]:
    """
    parse continuous alpha to discrete gene.
    alpha is ParameterList:
    ParameterList [
        Parameter(n_edges1, n_ops),
        Parameter(n_edges2, n_ops),
        ...
    ]

    gene is list:
    [
        [('node1_ops_1', node_idx), ..., ('node1_ops_k', node_idx)],
        [('node2_ops_1', node_idx), ..., ('node2_ops_k', node_idx)],
        ...
    ]

    If algorithm = 'top-k' =>  each node has two edges (k=2) in CNN.
    If algorithm = 'best' => each node as as many connections as there are not-none top nodes connected to it
    """

    gene = []
    primitives = ops.SEARCH_SPACE2OPS[search_space]
    assert primitives[-1] == 'none'  # assume last PRIMITIVE is 'none'

    # 1) Convert the mixed op to discrete edge (single op) by choosing top-1 weight edge
    # 2) Choose top-k edges per node by edge score (top-1 weight in edge)
    for edges in alpha:
        # edges: Tensor(n_edges, n_ops)
        if algorithm == 'top-k':  # ignore 'none'
            edges = edges[:, :-1]

        edge_max, primitive_indices = torch.topk(edges, 1)
        topk_edge_values, topk_edge_indices = torch.topk(edge_max.view(-1), k)
        node_gene = []

        if algorithm == 'top-k':
            for edge_idx in topk_edge_indices:
                prim_idx = primitive_indices[edge_idx]
                prim = primitives[prim_idx]
                node_gene.append((prim, edge_idx.item()))
        elif algorithm == 'best':
            for edge_idx, prim_idx in enumerate(primitive_indices):
                prim = primitives[prim_idx]
                if prim != 'none':
                    node_gene.append((prim, edge_idx))
        else:
            raise ValueError(f'Algorithm {algorithm} not supported')

        gene.append(node_gene)

    return gene


def plot(genotype: Genotype, file_path: PathLike, caption: Optional[str] = None):
    """ make DAG plot and save to file_path as .png """
    edge_attr = {
        'fontsize': '20',
        'fontname': 'times'
    }
    node_attr = {
        'style': 'filled',
        'shape': 'rect',
        'align': 'center',
        'fontsize': '20',
        'height': '0.5',
        'width': '0.5',
        'penwidth': '2',
        'fontname': 'times'
    }
    g = Digraph(
        format='png',
        edge_attr=edge_attr,
        node_attr=node_attr,
        engine='dot')
    g.body.extend(['rankdir=LR'])

    # input nodes
    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')

    # intermediate nodes
    n_nodes = len(genotype)
    for i in range(n_nodes):
        g.node(str(i), fillcolor='lightblue')

    for i, edges in enumerate(genotype):
        for op, j in edges:
            if j == 0:
                u = "c_{k-2}"
            elif j == 1:
                u = "c_{k-1}"
            else:
                u = str(j-2)

            v = str(i)
            g.edge(u, v, label=op, fillcolor="gray")

    # output node
    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(n_nodes):
        g.edge(str(i), "c_{k}", fillcolor="gray")

    # add image caption
    if caption:
        g.attr(label=caption, overlap='false', fontsize='20', fontname='times')

    g.render(file_path, view=False)
