import numpy as np

def decision_tree_split(X: list, y: list) -> list:
    """
    Returns the best feature index and threshold.
    """
    # Write code here
    X = np.array(X)
    y = np.array(y)
    count = np.bincount(y)
    N, D = X.shape

    best_gini = float('inf')
    best_feature = -1
    best_threshold = None
    
    for j in range(D):
        vals = np.unique(X[:, j])
        thresholds = (vals[:-1] + vals[1:]) / 2

        for t in thresholds:
            left = X[:, j] <= t
            right = X[:, j] > t

            NL = np.bincount(y[left], minlength = 2)
            NR = np.bincount(y[right], minlength = 2)

            GL = 1 - np.sum((NL / len(y[left])) ** 2)
            GR = 1 - np.sum((NR / len(y[right])) ** 2)

            weighted_gini = (len(y[left]) * GL + len(y[right]) * GR) / N

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = j
                best_threshold = t

    return [best_feature, best_threshold]