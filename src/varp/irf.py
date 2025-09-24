from .state_space import state_space_representation, state_variables
from .estimate_var import estimate_reduced_form_VAR
from numpy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt


def irf_coefs(df, p, h):
    """
    Create Impulse Response Function (IRF) and forecast for the next periods of VAR(p).

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
    h (int): the number of periods ahead.

    Returns
    -------
    Psi (numpy.ndarray): the (h + 1, k, k) array of IRF coefficients.
    """
    Phi, Gamma, _, Theta = state_space_representation(df, p)
    Sigma_u_hat = estimate_reduced_form_VAR(df, p)[1]
    Psi = [Theta @ np.linalg.matrix_power(Phi, j) @ Gamma for j in range(h + 1)]
    P = np.linalg.cholesky(Sigma_u_hat)
    Psi = np.stack(Psi @ P)

    return Psi


def irf_plots(df, p, h):
    """
    Create plots of Impulse Response Function (IRF).

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
    h (int): the number of periods ahead.

    Returns
    -------
    IRF plots showing responses of all variables to shocks.
    """
    n_vars = df.shape[1]
    irf = irf_coefs(df, p, h)
    var_names = list(df.columns)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(20, 10))
    for i in range(n_vars):
        for j in range(n_vars):
            axes[i, j].plot(irf[:, i, j])
            axes[i, j].axhline(0, color="red", linewidth=1, linestyle="--")
            axes[i, j].set_title(f"{var_names[i]} responds to shock {var_names[j]}")

    plt.tight_layout()
    plt.show()
