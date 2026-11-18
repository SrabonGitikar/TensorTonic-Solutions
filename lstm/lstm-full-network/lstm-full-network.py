import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(np.clip(-x, -500, 500)))

def lstm_forward(X: np.ndarray, W_f: np.ndarray, W_i: np.ndarray,
                 W_c: np.ndarray, W_o: np.ndarray, b_f: np.ndarray,
                 b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray,
                 W_y: np.ndarray, b_y: np.ndarray) -> dict:
    """
    Returns outputs, final_hidden_state, and final_cell_state.
    """
    N, T, D = X.shape
    O, H = W_y.shape

    h_prev = np.zeros((N, H))
    C_prev = np.zeros((N, H))
    yt = []

    for t in range(T):
        x_t = X[:, t, :]
        hx = np.concatenate([h_prev, x_t], axis=-1)
        
        f_t = sigmoid(hx @ W_f.T + b_f)
        i_t = sigmoid(hx @ W_i.T + b_i)
        C_tilde = np.tanh(hx @ W_c.T + b_c)
        C_t = f_t * C_prev + i_t * C_tilde
        o_t = sigmoid(hx @ W_o.T + b_o)
        h_t = o_t * np.tanh(C_t)
        y_t = h_t @ W_y.T + b_y

        yt.append(y_t)
        
        h_prev = h_t
        C_prev = C_t
        
    return {
        "outputs": np.stack(yt, axis=1),
        "final_hidden_state": h_t,
        "final_cell_state": C_t
    }