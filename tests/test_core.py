import pytest
import numpy as np
from arhmm.dataset import generate_2d_randomwalk
from arhmm.propagator import Propagator
from arhmm.core import ModelParameter, beta_forward
from arhmm.core import HiddenStates
from arhmm.core import alpha_forward

@pytest.fixture(scope='session')
def data_2d_randomwalk(): return generate_2d_randomwalk()

def test_alpha_beta(data_2d_randomwalk):
    xs_stack, zs_stack = data_2d_randomwalk
    hs = HiddenStates.construct(2, len(xs_stack[0]))

    noise_std = 1.2e-1
    prop1 = Propagator(np.ones((1, 1)), 1.2 * np.ones((1, 1)) * noise_std**2, np.array([0.1]))
    prop2 = Propagator(np.ones((1, 1)), 2.0 * np.ones((1, 1)) * noise_std**2, np.array([-0.6]))
    A_init = np.array([[0.99, 0.0], [0.01, 1.0]])

    mp_real = ModelParameter(A_init, props=[prop1, prop2])
    mp = ModelParameter(A=np.array([[0.99, 0.0], [0.01, 1.0]]), props=[prop1, prop2])
    alpha_forward(hs, mp, xs_stack[0])
    beta_forward(hs, mp, xs_stack[0])
