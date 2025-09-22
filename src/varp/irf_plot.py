import matplotlib.pyplot as plt
from .irf_and_forecast import irf_and_forecast


def irf_plot(df, p, h):
    """
    Create plots of Impulse Response Function (IRF).

    Parameters
    ----------
    df (numpy.ndarray): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
    h (int): the number of periods ahead.

    Returns
    -------
    IRF plots showing responses of all variables to shocks.
    """
    n_vars = df.shape[1]
    output = irf_and_forecast(df, p, h)[0]
    var_names = [f"x{i+1}" for i in range(n_vars)]
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(20, 10))
    for i in range(n_vars):
        for j in range(n_vars):
            axes[i, j].plot(output[:, i, j])
            axes[i, j].axhline(0, color="red", linewidth=1, linestyle="--")
            axes[i, j].set_title(f"{var_names[i]} response to shock {var_names[j]}")

    plt.tight_layout()
    plt.show()
