import numpy as np
from arhmm.propagator import Propagator
from arhmm.core import HiddenStates
from arhmm.core import ModelParameter

def generate_swtiching_linear_seq(n_time, mp: ModelParameter):
    x = np.array([0.0])
    z = 0
    xs = [x]
    zs = [z]
    for i in range(n_time):
        x_next = mp._props[z](x)
        z_next = np.random.choice(2, 1, replace=False, p=mp._A[:, z])[0]
        x, z = x_next, z_next
        xs.append(x)
        zs.append(z)
    return np.array(xs), np.array(zs)

def generate_2d_randomwalk(N=50):
    noise_std = 1e-1
    prop1 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.4]))
    prop2 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.4]))
    A_init = np.array([[0.95, 0.0], [0.05, 1.0]])
    mp_real = ModelParameter(A_init, props=[prop1, prop2])
    xs_stack = []
    zs_stack = []
    for i in range(N):
        xs, zs = generate_swtiching_linear_seq(100, mp_real)
        xs_stack.append(xs)
        zs_stack.append(zs)
    return xs_stack, zs_stack, mp_real
