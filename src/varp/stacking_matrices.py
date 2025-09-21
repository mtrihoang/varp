import numpy as np


def stacked_Y(df, p):
    Y = df[p:, :]
    return Y


def stacked_X(df, p):
    T, k = df.shape
    lag_list = []
    for r in range(1, p + 1):
        lag_list.append(df[(p - r) : (T - r), :])

    column_ones = np.ones((T - p, 1))
    lag_X = np.hstack(lag_list)
    X = np.hstack([column_ones, lag_X])

    return X
