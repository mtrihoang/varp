from .estimate_var import estimate_reduced_form_VAR
import numpy as np


def state_space_representation(df, p):
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
    T, k = df.shape
    s_T = np.zeros(k * p)
    for r in range(p):
        s_T[(k * r) : (k * (r + 1))] = df[T - (r + 1), :]
    return s_T
