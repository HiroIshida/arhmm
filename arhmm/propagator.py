from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scipy.stats


@dataclass
class Propagator:
    _dim: int
    _phi: np.ndarray
    _cov: np.ndarray
    _drift: np.ndarray

    def __call__(self, x: np.ndarray):
        mean = self._phi.dot(x) + self._drift
        x = np.random.multivariate_normal(mean, self._cov)
        return x

    def transition_prob(self, x: np.ndarray, x_next: np.ndarray) -> float:
        mean = self._phi.dot(x) + self._drift
        prob = scipy.stats.multivariate_normal.pdf(x_next, mean, self._cov)
        return prob

    @classmethod
    def fit_parameter(
        cls, xs_list: List[np.ndarray], ws_list: Optional[List[np.ndarray]] = None
    ) -> "Propagator":
        """ws_list: weigt of regression. In this context, ws_list is phase probability"""

        if ws_list is None:
            ws_list = [np.ones(len(xs) - 1) for xs in xs_list]

        assert len(xs_list) == len(ws_list)
        n_dim = xs_list[0].shape[1]

        x_sum = np.zeros(n_dim)
        y_sum = np.zeros(n_dim)
        xx_sum = np.zeros((n_dim, n_dim))
        xy_sum = np.zeros((n_dim, n_dim))
        w_sum = 0.0

        for xs, ws in zip(xs_list, ws_list):
            assert len(xs) == len(ws) + 1, "xs: {}, ws {}".format(len(xs), len(ws))
            X = xs[0:-1]
            Y = xs[1:]
            x_sum += np.sum(X * ws[:, None], axis=0)
            y_sum += np.sum(Y * ws[:, None], axis=0)
            xx_sum += X.T.dot(np.diag(ws)).dot(X)
            xy_sum += X.T.dot(np.diag(ws)).dot(Y)
            w_sum += sum(ws)

        # Thanks to Gauss-markov theorem, we can separate fitting processes into
        # first, non probabilistic term
        tmp = np.linalg.inv(w_sum * xx_sum - np.outer(x_sum, x_sum))
        phi_est = tmp.dot(w_sum * xy_sum - np.outer(x_sum, y_sum)).T
        b_est = (y_sum - phi_est.dot(x_sum)) * (1.0 / w_sum)

        cov_est = np.zeros((n_dim, n_dim))
        for xs, ws in zip(xs_list, ws_list):
            X = xs[0:-1]
            Xp = xs[1:]
            Xp_hat = (phi_est.dot(X.T)).T + b_est
            diff = Xp - Xp_hat
            cov_est += diff.T.dot(np.diag(ws)).dot(diff) / w_sum

        dim = len(phi_est)
        return cls(dim, phi_est, cov_est, b_est)

    def to_dict(self) -> Dict:
        d = {}
        for k in self.__dataclass_fields__.keys():  # type: ignore
            v = self.__dict__[k]
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Propagator":
        kwargs = {}
        for k in cls.__dataclass_fields__.keys():  # type: ignore
            v = d[k]
            if isinstance(v, list):
                v = np.array(v, dtype=np.float64)
            kwargs[k] = v
        return cls(**kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Propagator):
            return NotImplemented
        assert type(self) == type(other)
        if not np.allclose(self._phi, other._phi, atol=1e-6):
            return False
        if not np.allclose(self._cov, other._cov, atol=1e-6):
            return False
        if not np.allclose(self._drift, other._drift, atol=1e-6):
            return False
        return True


def create_sample_dataset(prop: Propagator):
    x_seq_list = []
    for i in range(30):
        x = np.random.randn(3)
        x_list = [x]
        for j in range(30):
            x = prop(x)
            x_list.append(x)
        x_seq_list.append(np.array(x_list))
    ws_list = [np.random.rand(30) for _ in range(30)]
    return x_seq_list, ws_list
