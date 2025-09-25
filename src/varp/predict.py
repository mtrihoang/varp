from .state_space import state_space_representation, state_variables
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


def pred_plots(df, varname, p, h, num_obs):
    """
    Create Impulse Response Function (IRF) and forecast for the next periods of VAR(p).

    Parameters
    ----------
    df (pandas.core.frame.DataFrame): input time series data.
    p (int): the number of lags, which will create lagged values x_{t-1}, x_{t-2}, ..., x_{t-p}.
    h (int): the number of periods ahead.
    num_obs (int): the last actual observations.

    Returns
    -------
    The VAR(p) forecast plot.
    """
    df_actual = df.tail(num_obs)
    df_pred = pred_vars(df, p, h)
    df_pred.columns = df_actual.columns
    df_actual["type"] = "actual"
    df_pred["type"] = "predicted"
    df_combined = pd.concat([df_actual, df_pred], ignore_index=True)
    df_combined["period"] = range(1, df_combined.shape[0] + 1)

    sns.lineplot(
        data=df_combined,
        x="period",
        y=varname,
        hue="type",
        style="type",
        markers=False,
        dashes={"actual": "", "predicted": (4, 2)},
        palette={"actual": "#277CA0", "predicted": "#277CA0"},
        linewidth=2,
    )
    sns.despine()

    plt.title(f"VAR({p}) forecast: next {h} steps", fontsize=14)
    plt.show()
