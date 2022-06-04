import argparse
import numpy as np
import matplotlib.pyplot as plt

from arhmm.dataset import generate_swtiching_linear_seq
from arhmm.propagator import Propagator
from arhmm.core import ARHMM
from arhmm.phase_estimator import OnlinePhaseEstimator

np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='visualize result')
    args = parser.parse_args()
    visualize = args.visualize

    # preparing real data
    noise_std = 0.2
    prop1 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.3]))
    prop2 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.3]))
    alpha = 0.1
    A_init = np.array([[1 - alpha, alpha], [alpha, 1 - alpha]])
    mp_real = ARHMM(A_init, props=[prop1, prop2])

    x_seq, z_seq = generate_swtiching_linear_seq(60, mp_real)

    z_est_seq = []
    estimator = OnlinePhaseEstimator.construct(mp_real)
    for x, z in zip(x_seq, z_seq):
        estimator.update(x)
        z_est_seq.append(np.argmax(estimator.z_est))

    plt.plot(z_seq, c='b')
    plt.plot(z_est_seq, 'r')
    plt.plot(np.array(x_seq).flatten())
    plt.show()
