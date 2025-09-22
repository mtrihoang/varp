import numpy as np
import pandas as pd


def gen_data(n_obs, n_vars, mean, sd, A, seed):
    """
    Create sample data for testing the varp package.

    Parameters
    ----------
    n_obs (int): the sample size.
    n_vars (int): the number of variables.
    mean (float): the mean of error term.
    sd (float): the standard deviation of error term.
    A (numpy.ndarray): a given coefficient matrix.
    seed (int): a given seed value.

    Returns
    -------
    df (pandas.core.frame.DataFrame): input time series data.
    """
    var_names = [f"x{i+1}" for i in range(n_vars)]
    rng = np.random.default_rng(seed)
    data = rng.random((n_obs, n_vars))
    for t in range(1, n_obs):
        data[t] = A @ data[t - 1] + np.random.normal(mean, sd, n_vars)
    df = pd.DataFrame(data, columns=var_names)
    return df
