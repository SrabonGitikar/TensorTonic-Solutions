import numpy as np

def cosine_similarity(a: list, b: list) -> float:
    """
    Returns the cosine similarity as a Python float.
    """
    # Write code here
    a = np.asarray(a)
    b = np.asarray(b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    dot = np.dot(a, b)

    if dot == 0:
        return 0.0
    
    return float(dot / (norm_a * norm_b))