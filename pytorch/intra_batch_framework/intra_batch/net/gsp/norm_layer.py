import torch


class LipschitzL2Normalization(torch.autograd.Function):
    """ torch layer to compute L2 normalization.
    Arguments:
        axis:
    Returns:
      A torch op with custom gradient.
    """

    @staticmethod
    def forward(ctx, input, axis=-1):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        x = input
        x_norm = torch.sqrt(torch.sum(torch.square(x), dim=axis, keepdim=True))
        normalize_mask = torch.greater(x_norm, 1.0).float()
        identity_mask = 1.0 - normalize_mask
        normalizer = normalize_mask * x_norm + identity_mask
        x_normalized = x / normalizer

        ctx.save_for_backward(x_normalized, normalize_mask, normalizer)
        ctx.axis = axis

        return x_normalized

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        x_normalized, normalize_mask, normalizer = ctx.saved_tensors
        axis = ctx.axis

        dy = grad_output
        x_normalized = x_normalized * normalize_mask  # make unnormalized vectors 0
        dx = (dy - torch.sum(x_normalized * dy, dim=axis, keepdim=True) * x_normalized) / normalizer

        # Gradients of non-Tensor arguments to forward must be None.
        return dx, None

'''from torch.autograd import gradcheck

input = (torch.randn(20, 20, 3, dtype=torch.double, requires_grad=True),
         torch.randn(30, 20, 3, dtype=torch.double, requires_grad=True))

l2normalize =LipschitzL2Normalization.apply

test = gradcheck(l2normalize, input, eps=1e-6, atol=1e-4)
print(test)'''