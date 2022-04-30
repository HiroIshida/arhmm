import numpy as np


def create_irreversible_markov_matrix(n_phase, p_trans):
    A = np.zeros((n_phase, n_phase))
    for i in range(n_phase - 1):
        A[i, i] = p_trans
        A[i + 1, i] = (1 - p_trans)
    A[-1, -1] = 1.0
    return A
