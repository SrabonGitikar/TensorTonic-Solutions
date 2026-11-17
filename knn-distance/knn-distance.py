import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array with shape (n_test, k).
    """
    # Write code here
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    N = X_train.shape[0]
    n = X_test.shape[0]
    dist = np.zeros((n, N))
    nbrs = []

    if n==0:
        return np.empty((0, k), dtype=int)
    
    for i in range(n):
        for j in range(N):
            dist[i][j] = np.linalg.norm(X_test[i] - X_train[j])
        idx = np.argsort(dist[i])[:k]
        if len(idx) < k:
            idx = np.pad(idx, (0, k - len(idx)), constant_values = -1)
        nbrs.append(idx)

    return np.array(nbrs)