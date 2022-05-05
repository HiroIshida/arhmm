import pytest
import numpy as np
# np.random.seed(seed=0)

from arhmm.dataset import generate_distinct_randomwalks
from arhmm.core import ARHMM, HiddenStates
from arhmm.core import _expectation_step
from arhmm.core import expectation_step

np.random.seed(0)


@pytest.fixture(scope='session')
def data_2d_randomwalk():
    return generate_distinct_randomwalks()


def test_expectation_step(data_2d_randomwalk):
    # Because we use real model parameter, only expectation step can predict
    # hidden state with a certain accuracy
    xs_list, zs_list, mp_real, _ = data_2d_randomwalk

    for i in range(len(xs_list)):
        xs, zs = xs_list[i], zs_list[i]
        hs, _ = _expectation_step(mp_real, xs)
        pred_phases = np.array([np.argmax(z) for z in hs.z_ests])
        error = sum(np.abs(pred_phases - zs[:-1])) / len(pred_phases)
        assert error < 0.1


def test_em_algorithm(data_2d_randomwalk):
    xs_list, zs_list, mp_real, mp_est = data_2d_randomwalk

    hs_list = [HiddenStates.construct(2, len(xs)) for xs in xs_list]
    loglikeli_seq = []
    for i in range(3):
        hs_list, loglikeli = expectation_step(mp_est, xs_list)
        mp_est = ARHMM.construct_by_maximization(hs_list, xs_list)
        loglikeli_seq.append(loglikeli)
    is_loglikeli_ascending = sorted(loglikeli_seq) == loglikeli_seq
    assert is_loglikeli_ascending
