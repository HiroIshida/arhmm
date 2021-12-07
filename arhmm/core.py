from dataclasses import dataclass
from typing import List
import numpy as np

from arhmm.propagator import Propagator

class ModelParameter:
    _A: np.ndarray
    _props: List[Propagator]
    _pmf_z1: np.ndarray
    def __init__(self, A: np.ndarray, props: List[Propagator]):
        n_phase = A.shape[0]
        pmf_z1 = np.zeros(n_phase); pmf_z1[0] = 1.0 # because initial phase must be phase 1
        self._A = A
        self._props = props
        self._pmf_z1 = pmf_z1

    @property
    def n_phase(self): return self._A.shape[0]

@dataclass
class HiddenStates:
    z_ests: List[np.ndarray]
    zz_ests: List[np.ndarray]
    alphas: List[np.ndarray]
    betas: List[np.ndarray]
    c_seq: np.ndarray

    @classmethod
    def construct(cls, n_phase: int, n_seq: int):
        z_ests = [np.zeros(n_phase) for _ in range(n_seq-1)]
        zz_ests = [np.zeros((n_phase, n_phase)) for _ in range(n_seq-2)]
        alphas = [np.zeros(n_phase) for _ in range(n_seq-1)]
        betas = [np.zeros(n_phase) for _ in range(n_seq-1)]
        c_seq = np.zeros(n_seq)
        return cls(z_ests, zz_ests, alphas, betas, c_seq)

    @property
    def n_phase(self): return len(self.z_ests[0])

def alpha_forward(hs: HiddenStates, mp: ModelParameter, xs: np.ndarray):
    hs.c_seq[0] = 1.0
    x1, x2 = xs[0], xs[1]
    px1 = 1.0 # deterministic x 
    trans_probs = lambda x, x_next: np.array([prop.transition_prob(x, x_next) for prop in mp._props])

    tmp = trans_probs(x1, x2) * mp._pmf_z1 * px1
    hs.c_seq[1] = sum(tmp)
    hs.alphas[0] = tmp/hs.c_seq[1]

    n_time = len(xs)
    for t in range(1, n_time-1):
        x_t, x_tp1 = xs[t], xs[t+1]
        for i in range(mp.n_phase):
            integral_term = sum(mp._A[i, j] * hs.alphas[t-1][j] for j in range(mp.n_phase))
            hs.alphas[t][i] = mp._props[i].transition_prob(x_t, x_tp1) * integral_term

        hs.c_seq[t+1] = sum(hs.alphas[t])
        hs.alphas[t] /= hs.c_seq[t+1]
