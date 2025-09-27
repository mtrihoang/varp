# varp

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-work--in--progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.12.4-blue)
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
<pre>
>>> df
            x1        x2        x3        x4        x5
t0    0.636514  0.384812  0.047445  0.955253  0.906051
t1    0.332441 -1.077303 -0.242069 -1.632524 -0.289735
t2    0.285713 -0.497562 -1.059906  1.763713 -0.897922
t3    0.434833 -2.832117  0.465437  1.283341 -0.219628
t4   -0.792584 -0.977940  1.980122  2.071441 -0.868831
...        ...       ...       ...       ...       ...
t995  1.522618 -0.978108  1.618740  1.228262  0.463357
t996 -1.454285 -0.373361  0.344117  0.804287 -1.162873
t997  0.462740  0.963316  1.355203  1.119711 -2.495599
t998  3.237160  0.182914 -2.293840  0.648967 -0.586916
t999 -2.399679 -2.033391 -0.406719 -0.037344  0.567880
</pre>
