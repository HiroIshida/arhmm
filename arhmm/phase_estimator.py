from dataclasses import dataclass
from typing import Optional

import numpy as np

from arhmm.core import ARHMM


@dataclass
class OnlinePhaseEstimator:
    arhmm: ARHMM
    z_est: np.ndarray
    x_current: Optional[np.ndarray]

    @classmethod
    def construct(cls, arhmm: ARHMM) -> "OnlinePhaseEstimator":
        latent = np.zeros(arhmm.n_phase)
        latent[0] = 1.0
        return cls(arhmm, latent, None)

    @property
    def n_phase(self) -> int:
        return self.arhmm.n_phase

    def update(self, x: np.ndarray):
        initialized = self.x_current is not None

        x_previous = self.x_current
        self.x_current = x

        if initialized:
            assert x_previous is not None

            z_pre_est = np.zeros(self.n_phase)
            for i in range(self.n_phase):
                prop = self.arhmm.props[i]
                z_pre_est[i] = prop.transition_prob(x_previous, x)
            z_pre_est = z_pre_est / sum(z_pre_est)
            z_est = self.arhmm.A.dot(z_pre_est)
            self.z_est = z_est
