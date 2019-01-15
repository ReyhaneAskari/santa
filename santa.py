import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class Santa(Optimizer):
    """Implements Santa algorithm.
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, decay_grad=0.1, anne_rate=0.5,
                 burnin=200, N=50000):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay,
                        decay_grad=decay_grad, anne_rate=anne_rate,
                        burnin=burnin, N=N)
        super(Santa, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Santa, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'Santa does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 1
                    state['square_avg'] = torch.zeros_like(p.data) + grad ** 2
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)

                    # import pdb; pdb.set_trace()
                    state['u'] = torch.randn(
                        grad.size()) * np.sqrt(group['lr'])
                    state['D'] = 1000 * np.sqrt(group['lr'])
                    state['gamma'] = state['D'] * torch.ones_like(p.data)
                    state['grad'] = grad + 0.0

                grad = group['decay_grad'] * state['grad'] + (
                    1 - group['decay_grad']) * grad
                state['grad'] = grad + 0.0

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                group['lr'] = (state['step'] ** (-0.3)) / 10.0

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if state['step'] < group['burnin']:
                    pcder = (1e-4 + square_avg.sqrt()).sqrt()
                else:
                    pcder = (group['eps'] + square_avg.sqrt()).sqrt()

                factor = state['step'] ** group['anne_rate']
                p.data = p.data + state['u'] / pcder / 2.0

                if state['step'] < group['burnin']:
                    state['gamma'] = state['gamma'] + (
                        state['u'] ** 2 - group['lr'] / factor) / 2.0

                state['u'] = (torch.exp(-state['gamma'] / 2) * state['u'] -
                              group['N'] * grad * group['lr'] / pcder / 2.0)

                if state['step'] < group['burnin']:
                    state['u'] = (state['u'] + np.sqrt(2 * group['lr'] **
                                  1.5 * 100.0 / factor) * torch.randn(
                                  grad.size()))

                state['u'] = (torch.exp(-state['gamma'] / 2) * state['u'] -
                              group['N'] * grad * group['lr'] / pcder / 2.0)

                if state['step'] < group['burnin']:
                    state['gamma'] = (state['gamma'] + (state['u'] **
                                      2 - group['lr'] / factor) / 2.0)

                p.data = p.data + state['u'] / pcder / 2.0

        return loss
