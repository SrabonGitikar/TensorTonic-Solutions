import numpy as np

def bag_of_words_vector(tokens: list, vocab: list) -> np.ndarray:
    """
    Returns a NumPy array with length len(vocab).
    """
    # Write code here
    tokens = np.asarray(tokens)
    vocab = np.asarray(vocab)

    count = [0] * len(vocab)

    for token in tokens:
        idx = np.where(vocab == token)[0]
        if len(idx) == 0:
            continue
        count[idx[0]] += 1
    
    return np.asarray(count)