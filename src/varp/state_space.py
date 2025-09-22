from .estimate_var import estimate_reduced_form_VAR
import numpy as np


def state_space_representation(df, p):
    """
    Create the state-space representation of VAR(p).

    Parameters
    ----------
    df (numpy.ndarray): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    Phi (numpy.ndarray): the (kp x kp) state transition matrix. Where k is the number of variables.
    Gamma (numpy.ndarray): the (kp x k) matrix which maps shock into state.
    C (numpy.ndarray): the (kp x 1) intercept vector, estimated from the reduced-form VAR(p).
    Theta (numpy.ndarray): the (k x kp) matrix which maps state into (observed) variables.
    """
    Beta_hat = estimate_reduced_form_VAR(df, p)[0]

    c_hat = Beta_hat[0, :]
    k = df.shape[1]
    Phi = np.zeros((k * p, k * p))

    for r in range(p):
        j = r + 1
        A_hat_j = Beta_hat[(1 + k * r) : (1 + k * (r + 1)), :].T
        Phi[0:k, (k * r) : (k * (r + 1))] = A_hat_j
        if (k * (r + 1)) < (k * p):
            Phi[(k * (r + 1)) : (k * (r + 2)), (k * r) : (k * (r + 1))] = np.eye(k)

    Gamma = np.zeros((k * p, k))
    Gamma[0:k, :] = np.eye(k)

    C = np.zeros(k * p)
    C[0:k] = c_hat.T

    Theta = np.zeros((k, k * p))
    Theta[:, 0:k] = np.eye(k)

    return Phi, Gamma, C, Theta


def state_variables(df, p):
    """
    Find the latest observed state of VAR(p).

    Parameters
    ----------
    df (numpy.ndarray): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    s_T (numpy.ndarray): the (kp x 1) vector is associated with the latest observed state.
    """
    T, k = df.shape
    s_T = np.zeros(k * p)
    for r in range(p):
        s_T[(k * r) : (k * (r + 1))] = df[T - (r + 1), :]
    return s_T
