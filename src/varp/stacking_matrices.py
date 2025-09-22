import numpy as np


def stacked_Y(df, p):
    """
    Create matrix Y.

    Parameters
    ----------
    df (numpy.ndarray): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    Y (numpy.ndarray): An array which contains stacked endogenous variables.
    """
    Y = df[p:, :]
    return Y


def stacked_X(df, p):
    """
    Create matrix X.

    Parameters
    ----------
    df (numpy.ndarray): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    X (numpy.ndarray): An array which contains stacked regressor variables (with intercept).
    """
    T, k = df.shape
    lag_list = []
    for r in range(1, p + 1):
        lag_list.append(df[(p - r) : (T - r), :])

    column_ones = np.ones((T - p, 1))
    lag_X = np.hstack(lag_list)
    X = np.hstack([column_ones, lag_X])

    return X
