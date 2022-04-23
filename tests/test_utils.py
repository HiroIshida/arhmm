import numpy as np

from arhmm.utils import create_markov_matrix


def test_create_markov_matrix():
    A = create_markov_matrix(3, 0.99)
    A_desired = np.array([
        [0.99, 0.00, 0.00],
        [0.01, 0.99, 0.00],
        [0.00, 0.01, 1.00]])
    np.testing.assert_almost_equal(A, A_desired)
