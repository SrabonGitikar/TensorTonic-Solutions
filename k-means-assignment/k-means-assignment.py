import numpy as np

def k_means_assignment(points: list, centroids: list) -> list:
    """
    Returns the nearest-centroid index for every point.
    """
    # Write code here
    X = np.array(points)
    Mu = np.array(centroids)
    N = X.shape[0]
    n = Mu.shape[0]
    assgn = [0] * N
    for i in range(N):
        dist = []
        for j in range(n):
            dist.append((np.linalg.norm(X[i] - Mu[j])) ** 2)
        assgn[i] = np.argmin(dist)

    return assgn