import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    a = inputs @ weights + bias
    y = torch.tanh(a)
    delta = upstream_gradient * (1 - y) * (1 + y)
    dL_dx = delta * weights
    dL_dw = delta * inputs
    dL_db = delta
    return y, dL_dx, dL_dw, dL_db
