import pytest
import numpy as np
# np.random.seed(seed=0)

from arhmm.dataset import generate_distinct_randomwalks
from arhmm.dataset import irreversible_random_walk_dataset
from arhmm.core import ARHMM, HiddenStates
from arhmm.core import _expectation_step
from arhmm.core import expectation_step
from arhmm.core import maximization_step

np.random.seed(0)


def test_expectation_step():
    # Because we use real model parameter, only expectation step can predict
    # hidden state with a certain accuracy
    n_dim = 3
    n_phase = 3
    n_data = 30
    seqs, model = irreversible_random_walk_dataset(n_data, n_dim, n_phase, 10)
    for seq in seqs:
        xs = np.array([s.x for s in seq])
        zs = np.array([s.phase for s in seq])
        hs = HiddenStates.construct(n_phase, len(seq))
        _expectation_step(hs, model, xs)
        pred_phases = np.array([np.argmax(z) for z in hs.z_ests])
        error = sum(np.abs(pred_phases - zs[:-1])) / len(pred_phases)
        assert error < 0.1


def test_maximization_step():
    n_dim = 3
    n_phase = 3
    n_data = 30

    hs_list = []
    xs_list = []
    seqs, model = irreversible_random_walk_dataset(n_data, n_dim, n_phase, 10)
    for seq in seqs:
        xs = np.array([s.x for s in seq])
        zs = np.array([s.phase for s in seq])
        hs = HiddenStates.construct(n_phase, len(seq))
        print(hs.zz_ests)
        hs_list.append(hs)
        xs_list.append(xs)
    assert False

    import copy
    model_new = copy.deepcopy(model)
    maximization_step(hs_list, model_new, xs_list)
    #print(model.A)
    #print(model_new.A)
    assert False


"""
def test_em_algorithm(data_2d_randomwalk):
    xs_list, zs_list, mp_real, mp_est = data_2d_randomwalk

    hs_list = [HiddenStates.construct(2, len(xs)) for xs in xs_list]
    loglikeli_seq = []
    for i in range(3):
        loglikeli = expectation_step(hs_list, mp_est, xs_list)
        maximization_step(hs_list, mp_est, xs_list)
        loglikeli_seq.append(loglikeli)
    is_loglikeli_ascending = sorted(loglikeli_seq) == loglikeli_seq
    assert is_loglikeli_ascending
"""
