from typing import List, Tuple
import numpy as np
from numpy.lib.function_base import append
from arhmm.propagator import Propagator
from arhmm.core import ARHMM, PhasedState
from arhmm.utils import create_irreversible_markov_matrix


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
    prop1 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.4]))
    prop2 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.4]))
    A_init = np.array([[0.85, 0.15], [0.15, 0.85]])
    mp_real = ARHMM(A_init, props=[prop1, prop2])

    prop1_est = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.3]))
    prop2_est = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.2]))
    A_init_est = np.array([[0.9, 0.1], [0.1, 0.9]])
    mp_est = ARHMM(A_init_est, props=[prop1_est, prop2_est])

    xs_list = []
    zs_list = []
    for i in range(N):
        xs, zs = generate_swtiching_linear_seq(100, mp_real)
        xs_list.append(xs)
        zs_list.append(zs)
    return xs_list, zs_list, mp_real, mp_est


def irreversible_random_walk_dataset(n_data: int, n_dim: int, n_phase: int) -> Tuple[List[List[PhasedState]], ARHMM]:
    props = []
    for i in range(n_phase):
        phi = np.random.randn(n_dim, n_dim)
        bias = np.random.randn(n_dim)
        tmp = np.random.rand(n_dim, n_dim)
        cov = np.dot(tmp, tmp.transpose()) * 0.1
        prop = Propagator(phi, cov, bias)
        props.append(prop)

    A_init = create_irreversible_markov_matrix(n_phase, 0.95)
    model = ARHMM(A_init, props=props)

    seqs: List[List[PhasedState]] = []
    for i in range(n_data):
        seq = [PhasedState(np.random.randn(n_dim), 0)]
        for j in range(100):
            seq.append(model(seq[-1]))
        seqs.append(seq)
    return seqs, model
