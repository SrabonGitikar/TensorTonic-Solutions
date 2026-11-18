import torch

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    d_k = K.shape[-1]
    z = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype = Q.dtype))
    return torch.softmax(z, dim=-1) @ V