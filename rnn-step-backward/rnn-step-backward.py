import numpy as np

def rnn_step_backward(dh: list, cache: list) -> dict:
    """
    Returns a dictionary with dx_t, dh_prev, dW, dU, and db.
    """
    # Write code here
    dh = np.asarray(dh)

    x_t = np.asarray(cache[0])
    h_prev = np.asarray(cache[1])
    h_t = np.asarray(cache[2])
    W = np.asarray(cache[3])
    U = np.asarray(cache[4])
    b = np.asarray(cache[5])

    dz = dh * (1 - h_t * h_t)
    
    dx_t = W.T @ dz
    dh_prev = U.T @ dz

    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz

    return {
        "dx_t": dx_t,
        "dh_prev": dh_prev,
        "dW": dW,
        "dU": dU,
        "db": db
    }