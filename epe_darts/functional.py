import torch
from entmax import entmax_bisect

from functorch import vmap, grad
from torch.autograd import Function


def softmax(inputs: torch.Tensor, mask: torch.BoolTensor, dim: int, epsilon: float = 1e-5):
    exps = torch.exp(inputs)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    #print(f"inputs is {inputs}")
    #print(f"exps is {exps}")
    #print(f"masked_exps is {masked_exps}")
    #print(f"masked_sums is {masked_sums}")
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


# The rest is for topk

sigmoid = torch.sigmoid
sigmoid_grad = vmap(vmap(grad(sigmoid)))

class TopK(Function):
    @staticmethod
    def forward(ctx, xs, k, t:int = 1):
        ts, ps = _find_ts(xs, k, t)
        ctx.save_for_backward(xs, ts)
        ctx.t = t
        return ps

    @staticmethod
    def backward(ctx, grad_output):
        # Compute vjp, that is grad_output.T @ J.
        xs, ts= ctx.saved_tensors
        # Let v = sigmoid'(x + t)
        v = sigmoid_grad((xs + ts)*ctx.t)
        s = v.sum(dim=1, keepdims=True)
        # Jacobian is -vv.T/s + diag(v)
        uv = grad_output * v
        t1 = - uv.sum(dim=1, keepdims=True) * v / s
        #verifying if s has 0s, if it does, it will cause an error and is necessary to replace the nans for 0s
        if torch.any(s == 0):
            #print(f"s tem zeros so, é assim {s}")
            #print(f"(xs + ts)*ctx.t é {(xs + ts)*ctx.t}")
            #print(f"the v was {v}")
            #print(f"xs é {xs}")
            #print(f"ts é {ts}")
            #return
            t1 = torch.nan_to_num(t1, nan=0.0)
        return t1 + uv, None, None

@torch.no_grad()
def _find_ts(xs, k, t):
    b, n = xs.shape
    assert 0 < k < n
    # Lo should be small enough that all sigmoids are in the 0 area.
    # Similarly Hi is large enough that all are in their 1 area.
    lo = -xs.max(dim=1, keepdims=True).values - 10
    hi = -xs.min(dim=1, keepdims=True).values + 10
    for _ in range(64):
        mid = (hi + lo)/2
        mask = sigmoid((xs + mid)*t).sum(dim=1) < k
        lo[mask] = mid[mask]
        hi[~mask] = mid[~mask]
    ts = (lo + hi)/2
    return ts, sigmoid((xs + ts)*t)