import numpy as np
from test_propagator import create_sample_dataset

from arhmm.propagator import Propagator
from arhmm.utils import (
    create_init_propagators_irreversible_case,
    create_irreversible_markov_matrix,
)


def test_create_init_propagators_irreversible_case():
    phi = np.array([[1.0, 0.3], [0.0, 1.0]])
    cov = np.eye(2) * 0.3
    drift = np.array([-0.01, 0.01])
    prop = Propagator(phi, cov, drift)
    xs_list, _ = create_sample_dataset(prop, 300)
    props_fit = create_init_propagators_irreversible_case(xs_list, 2)

    assert len(props_fit) == 2
    for prop_fit in props_fit:
        np.testing.assert_almost_equal(prop_fit._phi, phi, decimal=2)
        np.testing.assert_almost_equal(prop_fit._drift, drift, decimal=2)
        np.testing.assert_almost_equal(prop_fit._cov, cov, decimal=2)


def test_create_markov_matrix():
    A = create_irreversible_markov_matrix(3, 0.99)
    A_desired = np.array([[0.99, 0.00, 0.00], [0.01, 0.99, 0.00], [0.00, 0.01, 1.00]])
    np.testing.assert_almost_equal(A, A_desired)
