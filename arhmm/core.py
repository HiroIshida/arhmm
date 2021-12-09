from dataclasses import dataclass
from typing import List
import numpy as np
import math

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

def expectation_step(hs_list: List[HiddenStates], mp: ModelParameter, xs_list: List[np.ndarray]):
    for hs, xs in zip(hs_list, xs_list):
        _expectation_step(hs, mp, xs)

def _expectation_step(hs: HiddenStates, mp: ModelParameter, xs: np.ndarray):
    alpha_forward(hs, mp, xs)
    beta_forward(hs, mp, xs)

    for t in range(len(xs)-1):
        hs.z_ests[t] = hs.alphas[t] * hs.betas[t]

    for t in range(len(xs)-2):
        for i in range(mp.n_phase):
            for j in range(mp.n_phase):
                x_t, x_tt = xs[t+1], xs[t+1]
                prob_trans = mp._props[j].transition_prob(x_t, x_tt)
                tmp = hs.alphas[t][i] * mp._A[j, i] * prob_trans * hs.betas[t+1][j]
                hs.zz_ests[t][i, j] = tmp/hs.c_seq[t+2]

    # compute log_likelihood
    log_likeli = sum(math.log(c) for c in hs.c_seq)
    return log_likeli

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

def beta_forward(hs: HiddenStates, mp: ModelParameter, xs: np.ndarray):
    n_seq, n_dim = xs.shape
    hs.betas[n_seq - 2] = np.ones(mp.n_phase)
    for t in range(n_seq-3, -1, -1):
        x_tp1 = xs[t+1]
        x_tp2 = xs[t+2]
        for j in range(mp.n_phase): # phase at t
            sum = 0.0
            for i in range(mp.n_phase): # phase at t+1
                sum +=mp._A[i, j] * mp._props[i].transition_prob(x_tp1, x_tp2) * hs.betas[t+1][i]
            hs.betas[t][j] = sum
            hs.betas[t][j] /= hs.c_seq[t+2]

def maximization_step(hs_list: List[HiddenStates], mp: ModelParameter, xs_list: List[np.ndarray]):
    assert len(hs_list) == len(xs_list)

    # update pmf_z1
    mp._pmf_z1 = sum(hs.z_ests[0] for hs in hs_list) / len(hs_list)

    # update A
    A_new = np.zeros((mp.n_phase, mp.n_phase))
    for hs in hs_list:
        n_seq = len(hs.z_ests) + 1
        for t in range(n_seq - 2):
            for i in range(mp.n_phase):
                for j in range(mp.n_phase):
                    # Note that our stochastic matrix is different (transposed) from
                    # the one in PRML
                    A_new[j, i] += hs.zz_ests[t][i, j]
    for j in range(mp.n_phase): # normalize
        A_new[:, j] /= sum(A_new[:, j])
    mp.A = A_new

    # update prop list
    for i in range(mp.n_phase):
        ws_list = [np.array([z_est[i] for z_est in hs.z_ests]) for hs in hs_list]
        mp._props[i] = Propagator.fit_parameter(xs_list, ws_list)
