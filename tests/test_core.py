import pytest
import numpy as np
#np.random.seed(seed=0)

from arhmm.dataset import generate_distinct_randomwalks
from arhmm.propagator import Propagator
from arhmm.core import ModelParameter, beta_forward
from arhmm.core import HiddenStates
from arhmm.core import alpha_forward
from arhmm.core import _expectation_step
from arhmm.core import expectation_step
from arhmm.core import maximization_step

np.random.seed(0)

@pytest.fixture(scope='session')
def data_2d_randomwalk(): return generate_distinct_randomwalks()

def test_expectation_step(data_2d_randomwalk):
    # Because we use real model parameter, only expectation step can predict
    # hidden state with a certain accuracy
    xs_stack, zs_stack, mp_real, _ = data_2d_randomwalk

    for i in range(len(xs_stack)):
        xs, zs = xs_stack[i], zs_stack[i]
        hs = HiddenStates.construct(2, len(xs))
        _expectation_step(hs, mp_real, xs)
        pred_phases = np.array([np.argmax(z) for z in hs.z_ests])
        error = sum(np.abs(pred_phases - zs[:-1]))/len(pred_phases)
        assert error < 0.1

def test_em_algorithm(data_2d_randomwalk):
    xs_stack, zs_stack, mp_real, mp_est = data_2d_randomwalk

    hs_list = [HiddenStates.construct(2, len(xs)) for xs in xs_stack]
    loglikeli_seq = []
    for i in range(3):
        loglikeli = expectation_step(hs_list, mp_est, xs_stack)
        maximization_step(hs_list, mp_est, xs_stack)
        loglikeli_seq.append(loglikeli)
    is_loglikeli_ascending = sorted(loglikeli_seq) == loglikeli_seq
    assert is_loglikeli_ascending
