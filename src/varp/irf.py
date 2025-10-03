from .state_space import state_space_representation, state_variables
from .estimate_var import estimate_reduced_form_VAR
from numpy.linalg import cholesky
import numpy as np
import matplotlib.pyplot as plt


def irf_coefs(df, p, h):
    """
    Compute impulse response functions (IRFs) from VAR(p).

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, associated with lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
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
    Generate IRF plots from VAR(p).

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, associated with lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
    h (int): the number of periods ahead.

    Returns
    -------
    IRF plots showing responses of all variables to shocks.
    """
    n_vars = df.shape[1]
    irf = irf_coefs(df, p, h)
    var_names = list(df.columns)
    min_irf = np.min(irf)
    max_irf = np.max(irf)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(3 * n_vars, 2.5 * n_vars))

    plt.rcParams["font.family"] = "Source Code Pro"
    plt.rcParams["font.size"] = 9

    for i in range(n_vars):
        for j in range(n_vars):
            axes[i, j].plot(irf[:, i, j], color="black", linewidth=2)
            axes[i, j].axhline(0, color="red", linewidth=2, linestyle="--")
            axes[i, j].set_title(f"{var_names[i]} responds to shock {var_names[j]}")
            axes[i, j].set_xlim(0, h)
            axes[i, j].set_ylim(min_irf, max_irf)

    plt.tight_layout()
    plt.show()
