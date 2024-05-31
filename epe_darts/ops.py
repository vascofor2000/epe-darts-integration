""" Operations """
import torch
import torch.nn as nn
from copy import deepcopy

OPS = {
    'none': lambda channels, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda channels, stride, affine: PoolBN('avg', channels, 3, stride, 1, affine=affine),
    'max_pool_3x3': lambda channels, stride, affine: PoolBN('max', channels, 3, stride, 1, affine=affine),
    'skip_connect': lambda channels, stride, affine: Identity() if stride == 1 else FactorizedReduce(channels, channels, affine=affine),
    'sep_conv_3x3': lambda channels, stride, affine: SepConv(channels, channels, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda channels, stride, affine: SepConv(channels, channels, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda channels, stride, affine: SepConv(channels, channels, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda channels, stride, affine: DilConv(channels, channels, 3, stride, 2, 2, affine=affine),  # 5x5
    'dil_conv_5x5': lambda channels, stride, affine: DilConv(channels, channels, 5, stride, 4, 2, affine=affine),  # 9x9
    'conv_7x1_1x7': lambda channels, stride, affine: FacConv(channels, channels, 7, stride, 3, affine=affine),
    'nor_conv_7x7': lambda channels, stride, affine: ReLUConvBN(channels, channels, 7, stride, 3, affine=affine),
    'nor_conv_3x3': lambda channels, stride, affine: ReLUConvBN(channels, channels, 3, stride, 1, affine=affine),
    'nor_conv_1x1': lambda channels, stride, affine: ReLUConvBN(channels, channels, 1, stride, 0, affine=affine),
    #all from nats
    "nats-nor_conv_1x1": lambda C_in, C_out, stride, affine, track_running_stats: NATSReLUConvBN(C_in,C_out,(1, 1),(stride, stride),(0, 0),(1, 1),affine,track_running_stats),
    "nats-nor_conv_3x3": lambda C_in, C_out, stride, affine, track_running_stats: NATSReLUConvBN(C_in,C_out,(3, 3),(stride, stride),(1, 1),(1, 1),affine,track_running_stats),
    "nats-avg_pool_3x3": lambda C_in, C_out, stride, affine, track_running_stats: POOLING(C_in, C_out, stride, "avg", affine, track_running_stats),
    "nats-skip_connect": lambda C_in, C_out, stride, affine, track_running_stats: Identity(),
    "nats-none": lambda C_in, C_out, stride, affine, track_running_stats: NATSZero(C_in, C_out, stride),
}


CONNECT_NAS_BENCHMARK = ['nor_conv_3x3', 'skip_connect', 'none']
NAS_BENCH_201         = ['nats-nor_conv_1x1', 'nats-nor_conv_3x3', 'nats-avg_pool_3x3', 'nats-skip_connect', 'nats-none']
DARTS                 = ['sep_conv_3x3', 'sep_conv_5x5', 'dil_conv_3x3', 'dil_conv_5x5',
                         'avg_pool_3x3', 'max_pool_3x3', 'skip_connect', 'none']
S2                    = ['sep_conv_3x3', 'skip_connect']
S3                    = ['sep_conv_3x3', 'skip_connect', 'none']
SEARCH_SPACE2OPS = {
    'connect-nas-bench': CONNECT_NAS_BENCHMARK,
    'nas-bench-201': NAS_BENCH_201,
    'darts': DARTS,
    'S2': S2,
    'S3': S3,
}


def drop_path_(x, drop_prob, training):
    if training and drop_prob > 0.:
        keep_prob = 1. - drop_prob
        # per data point mask;
        mask = torch.Tensor(x.size(0), 1, 1, 1).to(x.device).bernoulli_(keep_prob)
        x.div_(keep_prob).mul_(mask)

    return x


class DropPath_(nn.Module):
    def __init__(self, p=0.):
        """ [!] DropPath is inplace module
        Args:
            p: probability of an path to be zeroed.
        """
        super().__init__()
        self.p = p

    def extra_repr(self):
        return 'p={}, inplace'.format(self.p)

    def forward(self, x):
        drop_path_(x, self.p, self.training)

        return x


class PoolBN(nn.Module):
    """
    AvgPool or MaxPool - BN
    """
    def __init__(self, pool_type, channels, kernel_size, stride, padding, affine=True):
        """
        Args:
            pool_type: 'max' or 'avg'
        """
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

        self.bn = nn.BatchNorm2d(channels, affine=affine)

    def forward(self, x):
        out = self.pool(x)
        out = self.bn(out)
        return out


class StdConv(nn.Module):
    """ Standard conv
    ReLU - Conv - BN
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FacConv(nn.Module):
    """ Factorized conv
    ReLU - Conv(Kx1) - Conv(1xK) - BN
    """
    def __init__(self, in_channels, out_channels, kernel_length, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, (kernel_length, 1), stride, padding, bias=False),
            nn.Conv2d(in_channels, out_channels, (1, kernel_length), stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class DilConv(nn.Module):
    """ (Dilated) depthwise separable conv
    ReLU - (Dilated) depthwise separable - Pointwise - BN

    If dilation == 2, 3x3 conv => 5x5 receptive field
                      5x5 conv => 9x9 receptive field
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    """ Depthwise separable conv
    DilConv(dilation=1) * 2
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(in_channels, in_channels, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(in_channels, out_channels, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        # re-sizing by stride
        return x[:, :, ::self.stride, ::self.stride] * 0.


class ReLUConvBN(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, dilation=1, bias=not affine),
            nn.BatchNorm2d(out_channels, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    """
    Reduce feature map size by factorized pointwise(stride=2).
    """
    def __init__(self, in_channels, out_channels, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

class namingOP(nn.Module):
    """little trick to have the name of the operations when calling named_parameters()"""
    def __init__(self, primitive, op):
        super().__init__()
        self.primitive = primitive
        if primitive == "none":
            self.none = op
        elif primitive == 'avg_pool_3x3':
            self.avg_pool_3x3 = op
        elif primitive == 'max_pool_3x3':
            self.max_pool_3x3 = op
        elif primitive == 'skip_connect':
            self.skip_connect = op
        elif primitive == 'sep_conv_3x3':
            self.sep_conv_3x3 = op
        elif primitive == 'sep_conv_5x5':
            self.sep_conv_5x5 = op
        elif primitive == 'sep_conv_7x7':
            self.sep_conv_7x7 = op
        elif primitive == 'dil_conv_3x3':
            self.dil_conv_3x3 = op
        elif primitive == 'dil_conv_5x5':
            self.dil_conv_5x5 = op
        elif primitive == 'conv_7x1_1x7':
            self.conv_7x1_1x7 = op
        elif primitive == 'nor_conv_7x7':
            self.nor_conv_7x7 = op
        elif primitive == 'nor_conv_3x3':
            self.nor_conv_3x3 = op
        elif primitive == 'nor_conv_1x1':
            self.nor_conv_1x1 = op
    
    def forward(self, x):
        if self.primitive == "none":
            return self.none(x)
        elif self.primitive == 'avg_pool_3x3':
            return self.avg_pool_3x3(x)
        elif self.primitive == 'max_pool_3x3':
            return self.max_pool_3x3(x)
        elif self.primitive == 'skip_connect':
            return self.skip_connect(x)
        elif self.primitive == 'sep_conv_3x3':
            return self.sep_conv_3x3(x)
        elif self.primitive == 'sep_conv_5x5':
            return self.sep_conv_5x5(x)
        elif self.primitive == 'sep_conv_7x7':
            return self.sep_conv_7x7(x)
        elif self.primitive == 'dil_conv_3x3':
            return self.dil_conv_3x3(x)
        elif self.primitive == 'dil_conv_5x5':
            return self.dil_conv_5x5(x)
        elif self.primitive == 'conv_7x1_1x7':
            return self.conv_7x1_1x7(x)
        elif self.primitive == 'nor_conv_7x7':
            return self.nor_conv_7x7(x)
        elif self.primitive == 'nor_conv_3x3':
            return self.nor_conv_3x3(x)
        elif self.primitive == 'nor_conv_1x1':
            return self.nor_conv_1x1(x)
        
    def print_primitive(self):
        print(f"primitive is {self.primitive}")


class MixedOp(nn.Module):
    """ Mixed operation """
    def __init__(self, channels, stride, search_space):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in SEARCH_SPACE2OPS[search_space]:
            op = OPS[primitive](channels, stride, affine=False)
            #self._ops.append(op)
            self._ops.append(namingOP(primitive, op))

    def forward(self, x, weights):
        """
        Args:
            x: input
            weights: weight for each operation
        """
        #for w, op in zip(weights, self._ops):
            #if w == 0:
            #    op.print_primitive()
        return sum(w * op(x) for w, op in zip(weights, self._ops) if w != 0)
        #return sum(w * op(x) for w, op in zip(weights, self._ops))


#FOR NATS-BENCH
class NATSReLUConvBN(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation,
        affine,
        track_running_stats=True,
    ):
        super(NATSReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                C_in,
                C_out,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=not affine,
            ),
            nn.BatchNorm2d(
                C_out, affine=affine, track_running_stats=track_running_stats
            ),
        )

    def forward(self, x):
        return self.op(x)

class POOLING(nn.Module):
    def __init__(
        self, C_in, C_out, stride, mode, affine=True, track_running_stats=True
    ):
        super(POOLING, self).__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(
                C_in, C_out, 1, 1, 0, 1, affine, track_running_stats
            )
        if mode == "avg":
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == "max":
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError("Invalid mode={:} in POOLING".format(mode))

    def forward(self, inputs):
        if self.preprocess:
            x = self.preprocess(inputs)
        else:
            x = inputs
        return self.op(x)


class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride, affine=True, track_running_stats=True):
        super(ResNetBasicblock, self).__init__()
        assert stride == 1 or stride == 2, "invalid stride {:}".format(stride)
        self.conv_a = NATSReLUConvBN(
            inplanes, planes, 3, stride, 1, 1, affine, track_running_stats
        )
        self.conv_b = NATSReLUConvBN(
            planes, planes, 3, 1, 1, 1, affine, track_running_stats
        )
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(
                    inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False
                ),
            )
        elif inplanes != planes:
            self.downsample = NATSReLUConvBN(
                inplanes, planes, 1, 1, 0, 1, affine, track_running_stats
            )
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def extra_repr(self):
        string = "{name}(inC={in_dim}, outC={out_dim}, stride={stride})".format(
            name=self.__class__.__name__, **self.__dict__
        )
        return string

    def forward(self, inputs):

        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(inputs)
        else:
            residual = inputs
        return residual + basicblock

class NATSZero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super(Zero, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.0)
            else:
                return x[:, :, :: self.stride, :: self.stride].mul(0.0)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros

    def extra_repr(self):
        return "C_in={C_in}, C_out={C_out}, stride={stride}".format(**self.__dict__)


# This module is used for NAS-Bench-201, represents a small search space with a complete DAG
class NAS201SearchCell(nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        stride,
        max_nodes,
        search_space,
        affine=False,
        track_running_stats=True,
    ):
        super(NAS201SearchCell, self).__init__()

        op_names = SEARCH_SPACE2OPS[search_space]
        self.op_names = deepcopy(op_names)
        self.edges = nn.ModuleDict()
        self.max_nodes = max_nodes
        self.in_dim = C_in
        self.out_dim = C_out
        for i in range(1, max_nodes):
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                if j == 0:
                    xlists = [
                        OPS[op_name](C_in, C_out, stride, affine, track_running_stats)
                        for op_name in op_names
                    ]
                else:
                    xlists = [
                        OPS[op_name](C_in, C_out, 1, affine, track_running_stats)
                        for op_name in op_names
                    ]
                self.edges[node_str] = nn.ModuleList(xlists)
        self.edge_keys = sorted(list(self.edges.keys()))
        self.edge2index = {key: i for i, key in enumerate(self.edge_keys)}
        self.num_edges = len(self.edges)

    def extra_repr(self):
        string = "info :: {max_nodes} nodes, inC={in_dim}, outC={out_dim}".format(
            **self.__dict__
        )
        return string

    def forward(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    sum(
                        layer(nodes[j]) * w
                        for layer, w in zip(self.edges[node_str], weights)
                    )
                )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # GDAS
    def forward_gdas(self, inputs, hardwts, index):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = hardwts[self.edge2index[node_str]]
                argmaxs = index[self.edge2index[node_str]].item()
                weigsum = sum(
                    weights[_ie] * edge(nodes[j]) if _ie == argmaxs else weights[_ie]
                    for _ie, edge in enumerate(self.edges[node_str])
                )
                inter_nodes.append(weigsum)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # GDAS Variant: https://github.com/D-X-Y/AutoDL-Projects/issues/119
    def forward_gdas_v1(self, inputs, hardwts, index):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = hardwts[self.edge2index[node_str]]
                argmaxs = index[self.edge2index[node_str]].item()
                weigsum = weights[argmaxs] * self.edges[node_str](nodes[j])
                inter_nodes.append(weigsum)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # joint
    def forward_joint(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                # aggregation = sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) / weights.numel()
                aggregation = sum(
                    layer(nodes[j]) * w
                    for layer, w in zip(self.edges[node_str], weights)
                )
                inter_nodes.append(aggregation)
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # uniform random sampling per iteration, SETN
    def forward_urs(self, inputs):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            while True:  # to avoid select zero for all ops
                sops, has_non_zero = [], False
                for j in range(i):
                    node_str = "{:}<-{:}".format(i, j)
                    candidates = self.edges[node_str]
                    select_op = random.choice(candidates)
                    sops.append(select_op)
                    if not hasattr(select_op, "is_zero") or select_op.is_zero is False:
                        has_non_zero = True
                if has_non_zero:
                    break
            inter_nodes = []
            for j, select_op in enumerate(sops):
                inter_nodes.append(select_op(nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # select the argmax
    def forward_select(self, inputs, weightss):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            inter_nodes = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                weights = weightss[self.edge2index[node_str]]
                inter_nodes.append(
                    self.edges[node_str][weights.argmax().item()](nodes[j])
                )
                # inter_nodes.append( sum( layer(nodes[j]) * w for layer, w in zip(self.edges[node_str], weights) ) )
            nodes.append(sum(inter_nodes))
        return nodes[-1]

    # forward with a specific structure
    def forward_dynamic(self, inputs, structure):
        nodes = [inputs]
        for i in range(1, self.max_nodes):
            cur_op_node = structure.nodes[i - 1]
            inter_nodes = []
            for op_name, j in cur_op_node:
                node_str = "{:}<-{:}".format(i, j)
                op_index = self.op_names.index(op_name)
                inter_nodes.append(self.edges[node_str][op_index](nodes[j]))
            nodes.append(sum(inter_nodes))
        return nodes[-1]


