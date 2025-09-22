import numpy as np
import pytest
from tests.gen_data import gen_data
from varp.estimate_var import estimate_reduced_form_VAR

n_obs = 1000
n_vars = 5
mean = 0
sd = 1
p = 3
h = 50
A = np.random.uniform(-0.5, 0.5, size=(n_vars, n_vars))
seed = 123456

df = gen_data(n_obs, n_vars, mean, sd, A, seed).to_numpy()


def test_irf():
    """
    Test the symmetry of the covariance matrix of residuals.
    """
    output = estimate_reduced_form_VAR(df, p)[1]
    assert np.allclose(output, output.T)
