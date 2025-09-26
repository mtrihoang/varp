import numpy as np


def stacked_Y(df, p):
    """
    Create matrix Y.

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, associated with lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    Y (numpy.ndarray): an array contains stacked endogenous variables.
    """
    df_tmp = df.copy()
    df_tmp = df_tmp.to_numpy()
    Y = df_tmp[p:, :]
    return Y


def stacked_X(df, p):
    """
    Create matrix X.

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, associated with lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.

    Returns
    -------
    X (numpy.ndarray): an array contains stacked regressors (with intercepts).
    """
    T, k = df.shape
    df_tmp = df.copy()
    df_tmp = df_tmp.to_numpy()

    lag_list = []
    for r in range(1, p + 1):
        lag_list.append(df_tmp[(p - r) : (T - r), :])

    column_ones = np.ones((T - p, 1))
    lag_X = np.hstack(lag_list)
    X = np.hstack([column_ones, lag_X])

    return X
