from .stacking_matrices import stacked_Y, stacked_X
from numpy.linalg import inv, eigvals, cholesky


def estimate_reduced_form_VAR(df, p):
    Y, X = stacked_Y(df, p), stacked_X(df, p)
    Beta_hat = inv(X.T @ X) @ (X.T @ Y)
    U_hat = Y - X @ Beta_hat
    T = X.shape[0] - p
    k = X.shape[1]
    Sigma_U_hat = 1 / (T - (1 + k * p)) * U_hat.T @ U_hat
    return Beta_hat, Sigma_U_hat
