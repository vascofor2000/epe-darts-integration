import torch
from entmax import entmax_bisect


def softmax(inputs: torch.Tensor, mask: torch.BoolTensor, dim: int, epsilon: float = 1e-5):
    exps = torch.exp(inputs)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return masked_exps / masked_sums


def sigmoid(inputs: torch.Tensor, mask: torch.BoolTensor):
    res = torch.sigmoid(inputs)
    return res * mask.float()


def entmax(inputs: torch.Tensor, mask: torch.BoolTensor, dim: int, alpha: float, epsilon: float = 1e-5):
    res = entmax_bisect(inputs, dim=dim, alpha=alpha, ensure_sum_one=False)
    masked_res = res * mask.float()
    masked_sums = masked_res.sum(dim, keepdim=True) + epsilon
    normalized = masked_res / masked_sums

    # Make sure the result is nonzero if the mask has only one True and its value is negative
    mask_sum = mask.sum(dim=-1)
    mask_sum = mask_sum == 1
    sz = list(mask.size())
    sz[0] = 1
    mask_sum = mask_sum.unsqueeze(1).repeat(*sz)

    nonzero = normalized + (mask & mask_sum).float()
    nonzero_sum = nonzero.sum(dim, keepdim=True)
    return nonzero / nonzero_sum
