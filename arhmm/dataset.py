import numpy as np

from arhmm.core import ARHMM
from arhmm.propagator import Propagator


def generate_swtiching_linear_seq(n_time, mp: ARHMM):
    x = np.array([0.0])
    z = 0
    xs = [x]
    zs = [z]
    for i in range(n_time):
        x_next = mp.props[z](x)
        z_next = np.random.choice(2, 1, replace=False, p=mp.A[:, z])[0]
        x, z = x_next, z_next
        xs.append(x)
        zs.append(z)
    return np.array(xs), np.array(zs)


def generate_distinct_randomwalks(N=10):
    noise_std = 1e-1
    prop1 = Propagator(2, np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.4]))
    prop2 = Propagator(2, np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.4]))
    A_init = np.array([[0.85, 0.15], [0.15, 0.85]])
    mp_real = ARHMM(A_init, props=[prop1, prop2])

    prop1_est = Propagator(2, np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.3]))
    prop2_est = Propagator(2, np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.2]))
    A_init_est = np.array([[0.9, 0.1], [0.1, 0.9]])
    mp_est = ARHMM(A_init_est, props=[prop1_est, prop2_est])

    xs_list = []
    zs_list = []
    for i in range(N):
        xs, zs = generate_swtiching_linear_seq(100, mp_real)
        xs_list.append(xs)
        zs_list.append(zs)
    return xs_list, zs_list, mp_real, mp_est
