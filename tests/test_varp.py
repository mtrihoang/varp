import numpy as np
import pytest
from tests.gen_data import gen_data
from varp.estimate_var import estimate_reduced_form_VAR
from varp.irf import irf_coefs, irf_plots
from varp.predict import pred_vars

seed = 123456
n_obs = 1000
n_vars = 5
mean = 0
sd = 1
p = 2
h = 50

np.random.seed(seed)

A = np.random.uniform(-0.5, 0.5, size=(n_vars, n_vars))
expected_irf_coefs = np.array(
    [-0.17638068, -0.09145083, 0.30423662, -0.40200738, 0.07473255]
)

df = gen_data(n_obs, n_vars, mean, sd, A, seed)


def test_symmetric_matrix():
    """
    Test the symmetry of the covariance matrix of residuals.
    """
    Sigma_U_hat = estimate_reduced_form_VAR(df, p)[1]
    assert np.allclose(Sigma_U_hat, Sigma_U_hat.T), "The matrix must be symmetric"


def test_positive_definite_matrix():
    """
    Test if the residual covariance matrix is positive definite.
    """
    Sigma_U_hat = estimate_reduced_form_VAR(df, p)[1]
    assert (
        np.linalg.eigvals(Sigma_U_hat) > 0
    ).all(), "Matrix must be positive definite"


def test_irf():
    """
    Check the IRF calculation.
    """
    estimated_irf_coefs = irf_coefs(df, p, h)[1][1]
    irf_diff = np.abs(estimated_irf_coefs - expected_irf_coefs)
    assert np.all(irf_diff < 1e-6), "IRF calculation test failed!"
