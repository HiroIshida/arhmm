import numpy as np
import pytest

from arhmm.core import ARHMM, HiddenStates
from arhmm.dataset import generate_distinct_randomwalks
from arhmm.utils import (
    create_init_propagators_irreversible_case,
    create_irreversible_markov_matrix,
)

# np.random.seed(seed=0)


np.random.seed(0)


@pytest.fixture(scope="session")
def data_2d_randomwalk():
    return generate_distinct_randomwalks()


def test_arhmm_serialization(data_2d_randomwalk):
    xs_list, zs_list, model_real, _ = data_2d_randomwalk

    props_init = create_init_propagators_irreversible_case(xs_list, 2)
    A_init = create_irreversible_markov_matrix(2, 0.98)
    arhmm = ARHMM(A_init, props_init, None)
    assert ARHMM.loads(arhmm.dumps()) == arhmm


def test_expectation_step(data_2d_randomwalk):
    # Because we use real model parameter, only expectation step can predict
    # hidden state with a certain accuracy
    xs_list, zs_list, model_real, _ = data_2d_randomwalk

    for i in range(len(xs_list)):
        xs, zs = xs_list[i], zs_list[i]
        hs, _ = model_real.expect_hs(xs)
        pred_phases = np.array([np.argmax(z) for z in hs.z_ests])
        error = sum(np.abs(pred_phases - zs[:-1])) / len(pred_phases)
        assert error < 0.1


def test_em_algorithm(data_2d_randomwalk):
    xs_list, zs_list, model_real, model_est = data_2d_randomwalk

    hs_list = [HiddenStates.construct(2, len(xs)) for xs in xs_list]
    loglikeli_seq = []
    for i in range(3):
        hs_list, loglikeli = model_est.expect_hs_list(xs_list)
        model_est = ARHMM.construct_by_maximization(hs_list, xs_list)
        loglikeli_seq.append(loglikeli)
    is_loglikeli_ascending = sorted(loglikeli_seq) == loglikeli_seq
    assert is_loglikeli_ascending
