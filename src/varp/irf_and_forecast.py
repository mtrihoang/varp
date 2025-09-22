from .state_space import state_space_representation, state_variables
from .estimate_var import estimate_reduced_form_VAR
from numpy.linalg import cholesky
import numpy as np


def irf_and_forecast(df, p, h):
    """
    Create Impulse Response Function (IRF) and forecast for the next periods of VAR(p).

    Parameters
    ----------
    df (numpy.ndarray): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
    h (int): the number of periods ahead.

    Returns
    -------
    Psi, (numpy.ndarray): the (h + 1, k, k) array of IRF coefficients.
    y_hat (numpy.ndarray): the (h + 1) x k array of predicted values.
    """
    Phi, Gamma, C, Theta = state_space_representation(df, p)

    s_forecast = state_variables(df, p)
    Psi = []
    y_hat = []

    for j in range(h + 1):
        Psi_j = Theta @ np.linalg.matrix_power(Phi, j) @ Gamma
        Psi.append(Psi_j)
        s_forecast = C + Phi @ s_forecast
        y_hat.append((Theta @ s_forecast).flatten())

    y_hat = np.array(y_hat)

    Sigma_u_hat = estimate_reduced_form_VAR(df, p)[1]
    P = np.linalg.cholesky(Sigma_u_hat)
    Psi = Psi @ P

    return Psi, y_hat
