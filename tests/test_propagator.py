import numpy as np
from arhmm.propagator import Propagator, create_sample_dataset

def test_propagator():
    phi = np.eye(3) * 2
    cov = np.eye(3) * 0.1
    drift = np.ones(3)
    prop = Propagator(phi, cov, drift)
    x_seq_list, ws_list = create_sample_dataset(prop)

    prop = Propagator.fit_parameter(x_seq_list, ws_list)
    np.testing.assert_almost_equal(prop._cov, cov, decimal=1e-2)
    np.testing.assert_almost_equal(prop._drift, drift, decimal=1e-2)
    np.testing.assert_almost_equal(prop._phi, phi, decimal=1e-2)
