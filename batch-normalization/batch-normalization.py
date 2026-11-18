import numpy as np

def batch_norm_forward(x: list, gamma: list, beta: list, eps: float = 1e-5) -> np.ndarray:
    """
    Returns a NumPy array with the same shape as x.
    """
    # Write code here
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    
    axes = (0,) + tuple(range(2, x.ndim))
    
    mu = np.mean(x, axis=axes, keepdims=True)
    var_x = np.var(x, axis=axes, keepdims=True)
    
    param_shape = (1, -1) + (1,) * (x.ndim - 2)
    gamma = gamma.reshape(param_shape)
    beta = beta.reshape(param_shape)
    
    x_norm = (x - mu) / np.sqrt(var_x + eps)
    y = gamma * x_norm + beta
    
    return y