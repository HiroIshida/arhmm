import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from arhmm.propagator import Propagator


@dataclass
class HiddenStates:
    z_ests: List[np.ndarray]
    zz_ests: List[np.ndarray]
    alphas: List[np.ndarray]
    betas: List[np.ndarray]
    c_seq: np.ndarray

    def to_phase_seq(self) -> List[int]:
        return [int(np.argmax(z_est)) for z_est in self.z_ests]

    @classmethod
    def construct(cls, n_phase: int, n_seq: int):
        indices_list = np.array_split(list(range(n_seq - 1)), n_phase)
        z_ests = []
        for phase, indices in enumerate(indices_list):
            for _ in range(len(indices)):
                e = np.zeros(n_phase)
                e[phase] = 1.0
                z_ests.append(e)
        assert len(z_ests) == n_seq - 1

        zz_ests = [np.zeros((n_phase, n_phase)) for _ in range(n_seq - 2)]
        alphas = [np.zeros(n_phase) for _ in range(n_seq - 1)]
        betas = [np.zeros(n_phase) for _ in range(n_seq - 1)]
        c_seq = np.zeros(n_seq)
        return cls(z_ests, zz_ests, alphas, betas, c_seq)

    @property
    def n_phase(self):
        return len(self.z_ests[0])


@dataclass
class ARHMM:
    A: np.ndarray
    props: List[Propagator]

    def __post_init__(self):
        assert self.A.shape[0] == len(self.props)

    @property
    def pmf_z1(self) -> np.ndarray:
        arr = np.zeros(self.n_phase)
        arr[0] = 1.0
        return arr

    @property
    def n_phase(self):
        return self.A.shape[0]

    @classmethod
    def construct_by_maximization(
        cls, hs_list: List[HiddenStates], xs_list: List[np.ndarray]
    ) -> "ARHMM":
        assert len(hs_list) == len(xs_list)
        n_phase = hs_list[0].n_phase

        # update A
        A = np.zeros((n_phase, n_phase))
        for hs in hs_list:
            n_seq = len(hs.z_ests) + 1
            for t in range(n_seq - 2):
                for i in range(n_phase):
                    for j in range(n_phase):
                        # Note that our stochastic matrix is different (transposed) from
                        # the one in PRML
                        A[j, i] += hs.zz_ests[t][i, j]
        for j in range(n_phase):  # normalize
            A[:, j] /= sum(A[:, j])

        # update prop list
        props: List[Propagator] = []
        for i in range(n_phase):
            ws_list = [np.array([z_est[i] for z_est in hs.z_ests]) for hs in hs_list]
            props.append(Propagator.fit_parameter(xs_list, ws_list))
        return cls(A, props)

    def expect_hs_list(self, xs_list: List[np.ndarray]) -> Tuple[List[HiddenStates], List[float]]:
        hs_list = []
        loglikeli_list = []
        for xs in xs_list:
            hs, log_likeli = self.expect_hs(xs)
            hs_list.append(hs)
            loglikeli_list.append(log_likeli)
        return hs_list, loglikeli_list

    def expect_hs(self, xs: np.ndarray) -> Tuple[HiddenStates, float]:
        hs = HiddenStates.construct(self.n_phase, len(xs))
        self.alpha_forward(hs, xs)
        self.beta_forward(hs, xs)

        for t in range(len(xs) - 1):
            hs.z_ests[t] = hs.alphas[t] * hs.betas[t]

        for t in range(len(xs) - 2):
            for i in range(self.n_phase):
                for j in range(self.n_phase):
                    x_t, x_tt = xs[t + 1], xs[t + 1]
                    prob_trans = self.props[j].transition_prob(x_t, x_tt)
                    tmp = hs.alphas[t][i] * self.A[j, i] * prob_trans * hs.betas[t + 1][j]
                    hs.zz_ests[t][i, j] = tmp / hs.c_seq[t + 2]

        # compute log_likelihood
        log_likeli = sum(math.log(c) for c in hs.c_seq)
        return hs, log_likeli

    def alpha_forward(self, hs: HiddenStates, xs: np.ndarray):
        hs.c_seq[0] = 1.0
        x1, x2 = xs[0], xs[1]  # noqa
        px1 = 1.0  # deterministic x

        def trans_probs(x, x_next):
            return np.array([prop.transition_prob(x, x_next) for prop in self.props])

        tmp = trans_probs(x1, x2) * self.pmf_z1 * px1
        hs.c_seq[1] = sum(tmp)
        hs.alphas[0] = tmp / hs.c_seq[1]

        n_time = len(xs)
        for t in range(1, n_time - 1):
            x_t, x_tp1 = xs[t], xs[t + 1]
            for i in range(self.n_phase):
                integral_term = sum(self.A[i, j] * hs.alphas[t - 1][j] for j in range(self.n_phase))
                hs.alphas[t][i] = self.props[i].transition_prob(x_t, x_tp1) * integral_term

            hs.c_seq[t + 1] = sum(hs.alphas[t])
            hs.alphas[t] /= hs.c_seq[t + 1]

    def beta_forward(self, hs: HiddenStates, xs: np.ndarray):
        n_seq, n_dim = xs.shape
        hs.betas[n_seq - 2] = np.ones(self.n_phase)
        for t in range(n_seq - 3, -1, -1):
            x_tp1 = xs[t + 1]
            x_tp2 = xs[t + 2]
            for j in range(self.n_phase):  # phase at t
                s = 0.0
                for i in range(self.n_phase):  # phase at t+1
                    s += (
                        self.A[i, j]
                        * self.props[i].transition_prob(x_tp1, x_tp2)
                        * hs.betas[t + 1][i]
                    )
                hs.betas[t][j] = s
                hs.betas[t][j] /= hs.c_seq[t + 2]

    def dumps(self) -> str:
        d = {}
        d["A"] = self.A.tolist()
        d["props"] = [prop.to_dict() for prop in self.props]
        return json.dumps(d, indent=2)

    def dump(self, path: Path) -> None:
        with path.open(mode="w") as f:
            f.write(self.dumps())

    @classmethod
    def loads(cls, jsonda_data: str) -> "ARHMM":
        d = json.loads(jsonda_data)
        kwargs: Dict[str, Any] = {}
        kwargs["A"] = np.array(d["A"], dtype=np.float64)
        kwargs["props"] = [Propagator.from_dict(dd) for dd in d["props"]]
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path) -> "ARHMM":
        with path.open(mode="r") as f:
            json_data = f.read()
        return cls.loads(json_data)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ARHMM):
            return NotImplemented
        assert type(self) == type(other)

        if not np.allclose(self.A, other.A, atol=1e-6):
            return False
        for prop_self, prop_other in zip(self.props, other.props):
            if prop_self != prop_other:
                return False
        return True


def train_arhmm(
    arhmm_init: ARHMM, xs_list: List[np.ndarray], f_tol=1e-3, n_max_iter=10, verbose=False
) -> Tuple[ARHMM, List[HiddenStates], List[float]]:
    arhmm = arhmm_init
    loglikeli_list_seq = []
    for i in range(n_max_iter):
        print(i)
        hs_list, loglikeli_list = arhmm.expect_hs_list(xs_list)
        loglikeli_sum = sum(loglikeli_list)
        arhmm = ARHMM.construct_by_maximization(hs_list, xs_list)
        if verbose:
            print("iter: {}, loglikeli {}".format(i, loglikeli_sum))
        loglikeli_list_seq.append(loglikeli_list)

        if i < 1:
            continue
        if (sum(loglikeli_list_seq[-1]) - sum(loglikeli_list_seq[-2])) < f_tol:
            break

    return arhmm, hs_list, loglikeli_list_seq  # type: ignore
