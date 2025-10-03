# varp

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-work--in--progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.12.4-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Contributors](https://img.shields.io/github/contributors/mtrihoang/varp)
![GitHub last commit](https://img.shields.io/github/last-commit/mtrihoang/varp)
![GitHub Repo stars](https://img.shields.io/github/stars/mtrihoang/varp)
![CI Status](https://github.com/mtrihoang/varp/actions/workflows/tests.yml/badge.svg)

**Author:** Tri Hoang

A Python package designed to explore Vector Autoregressive Models (VAR)

## Installation

From GitHub
```
pip install git+https://github.com/mtrihoang/varp.git
```

## Usage
Parameters

    df (pandas.core.frame.DataFrame): input time series data.
    varname (str): the selected variable to predict.
    p (int): the number of lags.
    h (int): the number of prediction steps.
    num_obs (int): the number of historical data points to display in forecast plots.

To compute impulse response functions (IRF) in VAR(p)
```
varp.irf_coefs(df, p, h)
```
To show IRF plots
```
varp.irf_plots(df, p, h)
```
To forecast h steps ahead
```
varp.pred_vars(df, p, h)
```
To show forecast plots
```
varp.pred_plots(df, varname, p, h, num_obs)
```

## Example
```
p = 3
h = 12

type(df)
<class 'pandas.core.frame.DataFrame'>

df.shape
(100, 4)

df.columns
Index(['x1', 'x2', 'x3', 'x4'], dtype='object')
```

```
varp.irf_plots(df, p = 3, h = 12)
```

![alt text](irf_fig.png)