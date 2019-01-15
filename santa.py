import torch
from torch.optim.optimizer import Optimizer
import numpy as np


class Santa(Optimizer):
    """Implements Santa algorithm.
    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered Santa,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False, decay_grad=0.1, anne_rate=0.5, burnin=200, N=50000):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay, decay_grad=decay_grad, anne_rate=anne_rate, burnin=burnin, N=N)
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
                    raise RuntimeError('Santa does not support sparse gradients')
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
                    state['u'] = torch.randn(grad.size()) * np.sqrt(group['lr']) ## Added by us
                    state['D'] = 1000 * np.sqrt(group['lr']) ## Added by us
                    state['gamma'] = state['D'] * torch.ones_like(p.data) ## Added by us
                    state['grad'] = grad + 0.0

                grad = group['decay_grad'] * state['grad'] + (1 - group['decay_grad']) * grad ## Added by us
                state['grad'] = grad + 0.0

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                group['lr'] = (state['step'] ** (-0.3)) / 10.0

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)

                if state['step'] < group['burnin']: ## Added by us
                    pcder = (1e-4 + square_avg.sqrt()).sqrt() ## Added by us
                else: ## Added by us
                    pcder = (group['eps'] + square_avg.sqrt()).sqrt() ## Added by us

                factor = state['step'] ** group['anne_rate'] ## Added by us
                p.data = p.data + state['u'] / pcder / 2.0 ## Added by us

                if state['step'] < group['burnin']: ## Added by us
                    state['gamma'] = state['gamma'] + (state['u'] ** 2 - group['lr'] / factor) / 2.0; ## Added by us

                state['u'] = torch.exp(-state['gamma'] / 2) * state['u'] - group['N'] * grad * group['lr'] / pcder / 2.0

                if state['step'] < group['burnin']:
                    state['u'] = state['u'] + np.sqrt(2 * group['lr'] ** 1.5 * 100.0 / factor) * torch.randn(grad.size())

                state['u'] = torch.exp(-state['gamma'] / 2) * state['u'] - group['N'] * grad * group['lr'] / pcder / 2.0

                if state['step'] < group['burnin']: ## Added by us
                    state['gamma'] = state['gamma'] + (state['u'] ** 2 - group['lr'] / factor) / 2.0

                p.data = p.data + state['u'] / pcder / 2.0

                # if group['centered']:
                #     grad_avg = state['grad_avg']
                #     grad_avg.mul_(alpha).add_(1 - alpha, grad)
                #     avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                # else:
                #     avg = square_avg.sqrt().add_(group['eps'])

                # if group['momentum'] > 0:
                #     buf = state['momentum_buffer']
                #     buf.mul_(group['momentum']).addcdiv_(grad, avg)
                #     p.data.add_(-group['lr'], buf)
                # else:
                #     p.data.addcdiv_(-group['lr'], grad, avg)

        return loss
