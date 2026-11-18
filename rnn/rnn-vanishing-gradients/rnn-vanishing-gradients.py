import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> np.ndarray:
    """
    Returns T float64 spectral-norm powers.
    """
    ans = [1]
    nrm = np.linalg.norm(W_hh, ord = 2)
    for k in range(T-1):
        ans.append(nrm ** (k+1))

    ans = np.array(ans, dtype = np.float64)
    return ans