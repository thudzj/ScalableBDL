import torch

class MulExpAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, psi, mu):
        ctx.mark_dirty(input)
        output = input.mul_(psi.exp()).add_(mu)
        ctx.save_for_backward(mu, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mu, output = ctx.saved_tensors
        grad_psi = (grad_output*(output - mu)).sum(0)
        grad_mu = grad_output.sum(0)
        return None, grad_psi, grad_mu