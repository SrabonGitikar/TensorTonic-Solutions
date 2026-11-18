import numpy as np

def min_max_scaling(data: list) -> list:
    """
    Returns each data column scaled to the range from 0 through 1.
    """
    # Write code here
    x = np.asarray(data, dtype = float)
    m, n = x.shape

    col_min = x.min(axis=0)
    col_max = x.max(axis=0)

    for i in range(m):
        for j in range(n):
            if col_max[j] == col_min[j]:
                x[i, j] = 0
                continue
            x[i, j] = (x[i, j] - col_min[j]) / (col_max[j] - col_min[j])

    return x.tolist()