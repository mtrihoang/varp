# varp

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Contributors](https://img.shields.io/github/contributors/mtrihoang/varp)
![GitHub last commit](https://img.shields.io/github/last-commit/mtrihoang/varp)
![GitHub Repo stars](https://img.shields.io/github/stars/mtrihoang/varp)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Python Version](https://img.shields.io/badge/python-3.12.4-blue)

**Author:** Tri Hoang 

A package for Vector Autoregressive Models (VAR)

## Installation

From GitHub
```
pip install git+https://github.com/mtrihoang/varp.git
```

## Usage
To compute impulse response functions (IRF) in VAR(p)
```
 varp.irf_and_forecast(df, p, h)
```
To show IRF plots
```
 varp.irf_plot(df, p, h)
```