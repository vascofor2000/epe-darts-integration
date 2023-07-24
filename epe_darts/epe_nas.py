import numpy as np
import torch
from torch import nn


def get_batch_jacobian(net: nn.Module, x: torch.Tensor):
    net.zero_grad()
    x.requires_grad = True

    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()

    return jacob


def eval_score_per_class(jacobs: np.ndarray, labels: np.ndarray, n_classes: int):
    per_class = {}
    for jacob, label in zip(jacobs, labels):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label], jacob))
        else:
            per_class[label] = jacob

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        corrs = np.corrcoef(per_class[c])

        s = np.sum(np.log(abs(corrs) + np.finfo(np.float32).eps))  # /len(corrs)
        if n_classes > 100:
            s /= len(corrs)
        ind_corr_matrix_score[c] = s

    # per class-corr matrix A and B
    score = 0
    ind_corr_matrix_score_keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:
        for c in ind_corr_matrix_score_keys:
            # B)
            score += np.absolute(ind_corr_matrix_score[c])
    else:
        for c in ind_corr_matrix_score_keys:
            # A)
            for cj in ind_corr_matrix_score_keys:
                score += np.absolute(ind_corr_matrix_score[c] - ind_corr_matrix_score[cj])

        # should divide by number of classes seen
        score /= len(ind_corr_matrix_score_keys)
    return score

