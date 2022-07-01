import math
import torch
import torch.optim as optim


__all__ = ['Adam']


class Adam(optim.Adam):

    @torch.no_grad()
#    def step(self, closure=None):
#        """Performs a single optimization step.
#        Arguments:
#            closure (callable, optional): A closure that reevaluates the model
#                and returns the loss.
#        """
#        loss = None
#        if closure is not None:
#            with torch.enable_grad():
#                loss = closure()
#
#        for group in self.param_groups:
#            for p in group['params']:
#                if p.grad is None or not p.requires_grad:
#                    continue
#                grad = p.grad
#                if grad.is_sparse:
#                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
#                amsgrad = group['amsgrad']
#
#                state = self.state[p]
#
#                # State initialization
#                if len(state) == 0:
#                    state['step'] = 0
#                    # Exponential moving average of gradient values
#                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                    # Exponential moving average of squared gradient values
#                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#                    if amsgrad:
#                        # Maintains max of all exp. moving avg. of sq. grad. values
#                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
#
#                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                if amsgrad:
#                    max_exp_avg_sq = state['max_exp_avg_sq']
#                beta1, beta2 = group['betas']
#
#                state['step'] += 1
#                bias_correction1 = 1 - beta1 ** state['step']
#                bias_correction2 = 1 - beta2 ** state['step']
#
#                if group['weight_decay'] != 0:
#                    grad = grad.add(p, alpha=group['weight_decay'])
#
#                # Decay the first and second moment running average coefficient
#                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
#                if amsgrad:
#                    # Maintains the maximum of all 2nd moment running avg. till now
#                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
#                    # Use the max. for normalizing running avg. of gradient
#                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
#                else:
#                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
#
#                step_size = group['lr'] / bias_correction1
#
#                p.addcdiv_(exp_avg, denom, value=-step_size)
#
#        return loss



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
                if p.grad is None or not p.requires_grad: #if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss
