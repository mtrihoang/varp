from .stacked_matrices import stacked_Y, stacked_X
from numpy.linalg import inv


def estimate_reduced_form_VAR(df, p):
    """
    Create the coefficient matrix of VAR(p) and the covariance matrix of residuals.

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    Beta_hat (numpy.ndarray): A (kp + 1) x k matrix of coefficients, including intercept.
    Sigma_U_hat (numpy.ndarray): A (k x k) covariance matrix of residuals.
    """
    Y, X = stacked_Y(df, p), stacked_X(df, p)
    Beta_hat = inv(X.T @ X) @ (X.T @ Y)
    U_hat = Y - X @ Beta_hat
    T = X.shape[0] - p
    k = X.shape[1]
    Sigma_U_hat = 1 / (T - (1 + k * p)) * U_hat.T @ U_hat
    return Beta_hat, Sigma_U_hat
