""" Architect controls architecture of cell by computing gradients of alphas """
import torch



class Architect:
    """ Compute gradients of alphas """

    def __init__(self, net,  v_net, w_momentum, w_weight_decay,
                 normal_none_penalty: float = 0, reduce_none_penalty: float = 0,
                 topk_on_alphas: bool = False, topk_on_virtualstep: bool = False, 
                 hd_on_alphas: bool = False, hd_on_virtualstep: bool = False, debugging: bool = False,
                 dropout_on_alphas: bool = False, dropout_on_virtualstep: bool = False):
        super().__init__()
        self.net = net
        self.v_net = v_net
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.normal_none_penalty: float = normal_none_penalty
        self.reduce_none_penalty: float = reduce_none_penalty
        self.topk_on_alphas = topk_on_alphas
        self.topk_on_virtualstep = topk_on_virtualstep
        self.hd_on_alphas = hd_on_alphas
        self.hd_on_virtualstep = hd_on_virtualstep
        self.debugging: bool = debugging
        self.dropout_on_alphas = dropout_on_alphas
        self.dropout_on_virtualstep = dropout_on_virtualstep

    def rolled_backward(self, val_X, val_y):
        loss = self.net.loss(val_X, val_y, self.topk_on_alphas, self.hd_on_alphas, self.dropout_on_alphas)

        normal_alphas, reduce_alphas = self.net.alpha_weights()
        # Loss += SUM[ none - mean(others) ]
        loss += self.normal_none_penalty * sum([(alpha[:, -1] - alpha[:, :-1].mean()).sum() for alpha in normal_alphas])
        loss += self.reduce_none_penalty * sum([(alpha[:, -1] - alpha[:, :-1].mean()).sum() for alpha in reduce_alphas])

        loss.backward()

    def beta_loss(self):
        y_pred_neg = self.net.alpha_weights()
        concatenated_tensor = torch.cat([torch.cat(sublist, dim=0) for sublist in y_pred_neg], dim=0)
        #print(f"concaten: {concatenated_tensor}")
        neg_loss = torch.logsumexp(concatenated_tensor, dim=-1)
        aux_loss = torch.mean(torch.Tensor(neg_loss))
        return aux_loss
        
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
        #print("goes on virtualstep")
        loss = self.net.loss(trn_X, trn_y, self.topk_on_virtualstep, self.hd_on_virtualstep, self.dropout_on_virtualstep)  # L_trn(w)

        # compute gradient
        #if self.debugging:
        #    gradients = torch.autograd.grad(loss, self.net.weights(), allow_unused=True)
            #models = self.net.named_modules()
            #for (name, model) in models:
            #    print(f"model name: {name}")
        #    parameters = self.net.named_weights()
        #    for (name, param) in parameters:
        #        print(f"Parameter Name: {name}")
        #gradients = self.none_2_0s(self.net.weights(), gradients)
            #print("H\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\nH\n")
        #else:
        gradients = torch.autograd.grad(loss, self.net.weights(), allow_unused=True)
        gradients = self.none_2_0s(self.net.weights(), gradients)
        
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for (name, w), vw, g in zip(self.net.named_weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                '''if self.debugging:
                    debuging
                    
                    print(f"o g é {g.size()}")
                    print(f"o do w é {w.size()}")
                    print(f"w is {type(w)}")
                    print(f"m is {type(m)}")
                    print(f"g is {type(g)}")
                    if not ((isinstance(m, torch.Tensor) or isinstance(m, float)) and isinstance(g, torch.Tensor) and isinstance(w, torch.nn.parameter.Parameter)):
                        
                        print(f"w is {type(w)} and lenght {len(w)} e conteudo é {w}")
                        if isinstance(m, float):
                            print(f"m is {type(m)} and possa conteudo é {m}")
                        else:                        
                            print(f"m is {type(m)} and size {len(m)} e conteudo é {m}")
                        print(f"g is {type(gradients)} and lenght {len(gradients)} ")
                        if isinstance(gradients, torch.Tensor):
                            print(f"e conteudo é {gradients.shape}")

                        w_none_indices = [[index for index, value in enumerate(self.net.weights()) if value is None]]
                        vw_none_indices = [[index for index, value in enumerate(self.v_net.weights()) if value is None]]
                        print(f"indices with None in w {w_none_indices}")
                        print(f"indices with None in vw {vw_none_indices}")
                        g_none_indices = [[index for index, value in enumerate(gradients) if value is None]]
                        print(f"indices with None in ggggggg {g_none_indices}")
                        
                        print(f"o g é {g}")
                        print(f"o size do w é {w.size()} e w mesmo é {w}")
                        g = torch.zeros(w.size())
                        print(f"o size do g é {g.size()} e g mesmo é {g}")
                        raise
                        
                        parameters = self.net.named_weights()
                        num_weights = 0
                        for (name, param) in parameters:
                            print(f"Parameter Name: {name}")
                            numel = param.numel()
                            num_weights += numel
                            print(f"Parameter number of weights: {numel}")
                            sizel = param.size()
                            print(f"Parameter size: {sizel}")
                            #print(f"Parameter Value: {param.data}")
                        print(f"Total number of weights: {num_weights}")
                        models = self.net.named_modules()
                        for (name, model) in models:
                            print(f"model name: {name}")
                        
                        
                    '''
                #if isinstance(g, torch.Tensor):
                vw.copy_(w - lr * (m + g + self.w_weight_decay * w))
                #else:
                #    print(f"Parameter Name that had none: {name}")
                #    '''end debuging'''
                #else:
                #vw.copy_(w - lr * (m + g + self.w_weight_decay * w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, lr, w_optim, epoch: int, amended: bool = False, bdarts: bool = False, dal: bool = False, dal_factor: int = 1):
        """ Compute unrolled loss and backward its gradients

        :param lr: learning rate for virtual gradient step (same as net lr)
        :param w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, lr, w_optim)

        # calc unrolled loss
        normal_alphas, reduce_alphas = self.v_net.alpha_weights()
        #print("goes for update alphas")
        loss = self.v_net.loss(val_X, val_y, self.topk_on_alphas, self.hd_on_alphas, self.dropout_on_alphas)  # L_val(w`)
        #print(f"normal loss is {loss}")

        if dal:
            loss += dal_factor * self.v_net.discretization_additional_loss()
            #print(f"loss after additional loss {loss}")

        # Loss += SUM[ none - mean(others) ]
        loss += self.normal_none_penalty * sum([(alpha[:, -1] - alpha[:, :-1].mean()).sum() for alpha in normal_alphas])
        loss += self.reduce_none_penalty * sum([(alpha[:, -1] - alpha[:, :-1].mean()).sum() for alpha in reduce_alphas])

        #adding beta_loss
        if bdarts:
            reg_coef = 0 + 50*epoch/100
            beta_loss = self.beta_loss()
            loss += reg_coef*beta_loss

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        #if self.debugging:
        #print(f"v_alphas is {v_alphas}")
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights, allow_unused=True)

        #print("calling printPosNans after autograd")
        #self.print_position_and_weight_of_nan(v_grads[len(v_alphas):], self.net.named_weights())
        v_grads = self.none_2_0s(v_alphas + v_weights, v_grads)

        #print("calling printPosNans after none20")
        #self.print_position_and_weight_of_nan(v_grads[len(v_alphas):], self.net.named_weights())
        #else:
        #v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.amended_gradient(dw, trn_X, trn_y) if amended else self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - lr*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - lr * h
                #print(f"is grad weird anytime? {alpha.grad}")

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

        #print("calling printPosNans after adding gradients on finite difference")
        #self.print_position_and_weight_of_nan(self.net.weights(), self.net.named_weights())

        torch._C._debug_only_display_vmap_fallback_warnings(True)
        #with torch.autograd.detect_anomaly(check_nan=True):
        
        # compute loss and the gradient with respect to alphas
        loss = self.net.loss(inputs, labels, self.topk_on_alphas, self.hd_on_alphas, self.dropout_on_alphas)
        #print("calling printPosNans after adding the loss calculation on finite difference")
        #self.print_position_and_weight_of_nan(self.net.weights(), self.net.named_weights())
        #print(f"loss is {loss}")
        aux = ~torch.isfinite(loss)
        if aux.any():
            print("loss is not finite")
        passe_flag = 1
        #while passe_flag:
        if wrt == 'alpha':          res = torch.autograd.grad(loss, self.net.alphas(), retain_graph=True)   # grad { Loss(w+, alpha) }
        elif wrt == 'weights':      res = torch.autograd.grad(loss, self.net.weights())  # grad { Loss(w+, weights) }
        else:
            raise ValueError(f'Computing gradient with respect to {wrt} is not supported')

        '''
            passe_flag -= 1
            weird_flag = False
            for tensore in res:
                
                nan_mask = torch.isnan(tensore)
                # Get indices of NaN values
                nan_indices = torch.nonzero(nan_mask, as_tuple=False)
                if nan_indices.numel() > 0:
                    weird_flag = True
            if weird_flag:
                passe_flag += 2
                print(f"res is {res}")
                print(f"lets make the autograd again for the {passe_flag} time")
            if passe_flag == 4:
                print("tryied 3 times probably occurs allways")
                return
        '''
        # recover w
        with torch.no_grad():
            for w, d in zip(self.net.weights(), dw):
                w -= eps * d
        
        #print("calling printPosNans after removing gradients on finite difference")
        #self.print_position_and_weight_of_nan(self.net.weights(), self.net.named_weights())
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
        #print(f"norm is {norm}")
        eps = epsilon / norm

        dalpha_pos = self.finite_difference(dw, trn_X, trn_y, eps, wrt='alpha')
        #print(f"dalpha_pos is {dalpha_pos}")
        dalpha_neg = self.finite_difference(dw, trn_X, trn_y, -eps, wrt='alpha')
        #print(f"dalpha_neg is {dalpha_neg}")
        hessian = [(p - n) / (2. * eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        #print(f"hessian is {hessian}")
        return hessian

    def none_2_0s(self, weights, gradients):
        new_gradients = []
        for w, g in zip(weights, gradients):
            if not isinstance(g, torch.Tensor):
                new_gradients.append(torch.zeros(w.size()).cuda())
            else:
                #print(f"size of gradients {w.size()}")
                new_gradients.append(g)
        return tuple(new_gradients)

    def print_position_and_weight_of_nan(self, gradients, named_weights):
        names = []
        for (name, weight) in named_weights:
            names.append(name)
        for d, name in zip(gradients, names):
            # Check for NaN values
            nan_mask = torch.isnan(d)
            # Get indices of NaN values
            nan_indices = torch.nonzero(nan_mask, as_tuple=False)
            if nan_indices.numel() > 0:
                print(f"found nans on {name}, on posiitons {nan_indices}")