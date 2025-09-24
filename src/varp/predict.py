from .state_space import state_space_representation, state_variables
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def pred_vars(df, p, h):
    """
    Create Impulse Response Function (IRF) and forecast for the next periods of VAR(p).

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
    h (int): the number of periods ahead.

    Returns
    -------
    y_hat (pandas.core.frame.DataFrame): the h x k array of predicted values.
    """
    Phi, _, C, Theta = state_space_representation(df, p)

    s_t = state_variables(df, p)
    y_t = []

    for j in range(h):
        s_t = C + Phi @ s_t
        y_t.append(Theta @ s_t)

    y_hat = pd.DataFrame(y_t)

    return y_hat
