import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import gradcheck

class PartialTransportPlan(torch.autograd.Function):
    """ torch layer to compute partial optimal transport.
    Arguments:
    # binary_costs: cost matrix tensor of shape = (batch_size, num_bins_in_target, num_bins_in_source)
    #               or (batch_size, num_bins_in_target, source_width, source_height)

    # mu: masses to be transported
    # 1 - mu amount of mass will be residual mass
    Returns:
      A torch op with custom gradient.
    """

    @staticmethod
    def forward(ctx, binary_costs, mu, gamma, max_it):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        cost_shape = binary_costs.shape
        if len(cost_shape) > 3:
            batch_size, target_dim = cost_shape[:2]
            source_dim = cost_shape[2] * cost_shape[3]
            source_axes = [-2, -1]
        else:
            batch_size, target_dim, source_dim = cost_shape
            source_axes = -1

        Kb = torch.exp(- gamma * binary_costs)
        Ku = 1.0 #torch.exp(- gamma * unary_costs)

        q = torch.tensor(1. / source_dim, dtype=torch.float32, device=Kb.device)

        Kb_T_1 = torch.sum(Kb, dim=1, keepdim=True)

        mu_ = torch.full(
            fill_value=mu,
            size=[cost_shape[0]] + [1] * (len(cost_shape) - 1),
            dtype=torch.float32,
            device=Kb.device)

        for k in range(max_it):
            p_ = q / (Ku + mu_ * Kb_T_1)
            mu_ = mu / torch.sum(p_ * Kb_T_1, dim=source_axes, keepdim=True)

        p = q / (Ku + mu_ * Kb_T_1)
        mu_ = mu / torch.sum(p * Kb_T_1, dim=source_axes, keepdim=True)
        X = torch.cat((Ku * p, mu_ * Kb * p), dim=1)

        ctx.save_for_backward(X)
        ctx.mu = mu
        ctx.gamma = gamma

        return X


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.

        ctx.X is the solution to the constrained optimal transport problem with
        X.shape = [batch_size, target_dim+1, source_dim] or [batch_size, target_dim+1, source_width, source_height]
        where X[:, 0] is the residual mass vector and X[:, 1:] is the transport plan to the target
        dy is the gradient wrt X
        retuns dC which is the gradient wrt cost matrix
        """
        X, = ctx.saved_tensors
        mu = ctx.mu
        gamma = ctx.gamma

        dy = grad_output

        # jacobian computation

        # !TODO: explain computation logic (clearly write expressions and steps)
        # dx/dc = -gamma*I_bar@(D(x)-D(x)A^T@inv(H)@A@D(x)), I_bar = [0 I] for flattened x and c

        solution_shape = X.shape
        if len(solution_shape) > 3:
            batch_size, target_dim = solution_shape[:2]
            source_dim = solution_shape[2] * solution_shape[3]
            source_axes = [-2, -1]
        else:
            batch_size, target_dim, source_dim = solution_shape
            source_axes = -1

        X_by_dLdX = X * dy  # D(x) @ dL/dx term

        r_by_dLdr = torch.unsqueeze(X_by_dLdX[:, 0], dim=1)

        # precomputing some coeff.s and vectors
        r = torch.unsqueeze(X[:, 0], dim=1)
        k_1 = 1.0 / (1.0 - mu - source_dim * torch.sum(torch.square(r), dim=source_axes, keepdim=True))

        row_sum = torch.sum(X_by_dLdX, dim=1, keepdim=True)  # g_r + \sum_i g_pi ; where g = X_by_dLdX
        coeff = torch.subtract(
            torch.sum(r_by_dLdr, dim=source_axes, keepdim=True),
            source_dim * torch.sum(row_sum * r, dim=source_axes, keepdim=True))

        # gradient wrt beta
        dLdmu = - coeff * k_1

        common_vec = source_dim * (row_sum + dLdmu * r)

        # dLdC
        pre_dLdC = - gamma * (X_by_dLdX - common_vec * X)

        # gradient wrt binary cost
        dLdCb = pre_dLdC[:, 1:]
        dx = dLdCb

        # gradient wrt unary cost
        #dLdCu = tf.expand_dims(pre_dLdC[:, 0], axis=1) - gamma * dLdmu * r

        # Gradients of non-Tensor arguments to forward must be None.
        return dx, None, None, None

def test_grad():
    binary_costs = torch.randn(8, 4, 3, 3, dtype=torch.double, requires_grad=True)

    mu = 0.3
    gamma = 5.0
    max_it = 100


    l2normalize = PartialTransportPlan.apply

    test = gradcheck(l2normalize, (binary_costs, mu, gamma, max_it), eps=1e-6, atol=1e-4)
    print('\n gradient check succeeded: {}\n'.format(test))