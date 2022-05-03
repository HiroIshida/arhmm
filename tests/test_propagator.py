import numpy as np
from arhmm.propagator import Propagator


def create_sample_dataset(prop: Propagator):
    x_seq_list = []
    N = 300
    for i in range(N):
        x = np.random.randn(2)
        x_list = [x]
        for j in range(30):
            x = prop(x)
            x_list.append(x)
        x_seq_list.append(np.array(x_list))
    ws_list = [np.ones(30) for _ in range(N)]
    return x_seq_list, ws_list


def test_propagator():
    phi = np.array([[1.0, 0.3], [0.0, 1.0]])
    cov = np.eye(2) * 0.3
    drift = np.array([-0.01, 0.01])
    prop = Propagator(phi, cov, drift)
    x_seq_list, ws_list = create_sample_dataset(prop)

    prop_fit = Propagator.fit_parameter(x_seq_list, ws_list)
    np.testing.assert_almost_equal(prop_fit._phi, phi, decimal=2)
    np.testing.assert_almost_equal(prop_fit._drift, drift, decimal=2)
    np.testing.assert_almost_equal(prop_fit._cov, cov, decimal=2)
