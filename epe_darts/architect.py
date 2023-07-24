""" Architect controls architecture of cell by computing gradients of alphas """
import torch


class Architect:
    """ Compute gradients of alphas """

    def __init__(self, net,  v_net, w_momentum, w_weight_decay,
                 normal_none_penalty: float = 0, reduce_none_penalty: float = 0):
        super().__init__()
        self.net = net
        self.v_net = v_net
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.normal_none_penalty: float = normal_none_penalty
        self.reduce_none_penalty: float = reduce_none_penalty

    def virtual_step(self, trn_X, trn_y, lr, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        :param lr: learning rate for virtual gradient step (same as weights lr)
        :param w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y)  # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - lr * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, lr, w_optim, amended: bool = False):
        """ Compute unrolled loss and backward its gradients

        :param lr: learning rate for virtual gradient step (same as net lr)
        :param w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, lr, w_optim)

        # calc unrolled loss
        normal_alphas, reduce_alphas = self.v_net.alpha_weights()
        loss = self.v_net.loss(val_X, val_y)  # L_val(w`)

        # Loss += SUM[ none - mean(others) ]
        loss += self.normal_none_penalty * sum([(alpha[:, -1] - alpha[:, :-1].mean()).sum() for alpha in normal_alphas])
        loss += self.reduce_none_penalty * sum([(alpha[:, -1] - alpha[:, :-1].mean()).sum() for alpha in reduce_alphas])

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.amended_gradient(dw, trn_X, trn_y) if amended else self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - lr*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - lr * h

    def finite_difference(self, dw, inputs, labels, eps: float, wrt: str = 'alpha'):
        """
        Computes finite difference approximation with respect to `wrt` parameter: f'(x) = [f(x+eps) - f(x)] / eps
        :ref: https://en.wikipedia.org/wiki/Finite_difference
        In our context
            * f is the self.net and
            * x is the w (self.net.weights())
            * gradient is computed with respect to alphas (self.net.alphas())

        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        :returns: gradient with respect to `wrt` of f(x + eps) = self.net( self.net.weights() + eps )
        """
        # w+ = w + eps*dw`
        with torch.no_grad():
            for w, d in zip(self.net.weights(), dw):
                w += eps * d

        # compute loss and the gradient with respect to alphas
        loss = self.net.loss(inputs, labels)
        if wrt == 'alpha':      res = torch.autograd.grad(loss, self.net.alphas())   # grad { Loss(w+, alpha) }
        elif wrt == 'weights':  res = torch.autograd.grad(loss, self.net.weights())  # grad { Loss(w+, weights) }
        else:
            raise ValueError(f'Computing gradient with respect to {wrt} is not supported')

        # recover w
        with torch.no_grad():
            for w, d in zip(self.net.weights(), dw):
                w -= eps * d

        return res

    def amended_gradient(self, dw, trn_X, trn_y, epsilon: float = 0.01, amend: float = 0.1):
        """
        dw = dw` { L_val(w`, alpha) }
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = epsilon / norm

        dw_pos = self.finite_difference(dw, trn_X, trn_y, eps, wrt='weights')
        dw_neg = self.finite_difference(dw, trn_X, trn_y, -eps, wrt='weights')
        dalpha_pos = self.finite_difference([(wp - wn) / 2 for wp, wn in zip(dw_pos, dw_neg)], trn_X, trn_y, 1, wrt='alpha')
        dalpha_neg = self.finite_difference([(wp - wn) / 2 for wp, wn in zip(dw_pos, dw_neg)], trn_X, trn_y, -1, wrt='alpha')
        hessian = [-amend * (p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian

    def compute_hessian(self, dw, trn_X, trn_y, epsilon: float = 0.01):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = epsilon / norm

        dalpha_pos = self.finite_difference(dw, trn_X, trn_y, eps, wrt='alpha')
        dalpha_neg = self.finite_difference(dw, trn_X, trn_y, -eps, wrt='alpha')
        hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
