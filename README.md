# varp

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Status](https://img.shields.io/badge/status-work--in--progress-yellow)
![Python Version](https://img.shields.io/badge/python-3.12.4-blue)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
![Contributors](https://img.shields.io/github/contributors/mtrihoang/varp)
![GitHub last commit](https://img.shields.io/github/last-commit/mtrihoang/varp)
![GitHub Repo stars](https://img.shields.io/github/stars/mtrihoang/varp)

**Author:** [Tri Hoang](https://github.com/mtrihoang)

A Python package designed to explore Vector Autoregressive Models (VAR)

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