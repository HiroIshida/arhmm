from typing import List

import numpy as np

from arhmm.propagator import Propagator


def create_init_propagators_irreversible_case(
    xs_list: List[np.ndarray], n_phase
) -> List[Propagator]:
    xs_list_partial_list: List[List[np.ndarray]] = [[] for _ in range(n_phase)]
    for xs in xs_list:
        n_seq_len = len(xs)
        indices_list = np.array_split(list(range(n_seq_len - 1)), n_phase)
        for phase, indices in enumerate(indices_list):
            xs_list_partial_list[phase].append(xs[indices])
    prop_each_phase_list = [Propagator.fit_parameter(e) for e in xs_list_partial_list]
    return prop_each_phase_list


def create_irreversible_markov_matrix(n_phase, p_trans):
    A = np.zeros((n_phase, n_phase))
    for i in range(n_phase - 1):
        A[i, i] = p_trans
        A[i + 1, i] = 1 - p_trans
    A[-1, -1] = 1.0
    return A
