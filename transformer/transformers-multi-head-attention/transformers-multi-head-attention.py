
import numpy as np

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, seq_length, d_model = Q.shape

    assert d_model % num_heads == 0

    head_width = d_model // num_heads

    Q = (Q @ W_q).reshape(
        batch_size, seq_length, num_heads, head_width
    ).transpose(0, 2, 1, 3)

    K = (K @ W_k).reshape(
        batch_size, seq_length, num_heads, head_width
    ).transpose(0, 2, 1, 3)

    V = (V @ W_v).reshape(
        batch_size, seq_length, num_heads, head_width
    ).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_width)

    attention = softmax(scores)

    output = attention @ V

    output = output.transpose(0, 2, 1, 3).reshape(
        batch_size, seq_length, d_model
    )

    return output @ W_o