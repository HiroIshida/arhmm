import argparse
import numpy as np

from arhmm.dataset import generate_swtiching_linear_seq
from arhmm.propagator import Propagator
from arhmm.core import ModelParameter, HiddenStates
from arhmm.core import expectation_step, maximization_step

np.random.seed(0)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--visualize', action='store_true', help='visualize result')
    args = parser.parse_args()
    visualize = args.visualize

    # preparing real data
    noise_std = 3e-1
    prop1 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.4]))
    prop2 = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.4]))
    A_init = np.array([[0.85, 0.15], [0.15, 0.85]])
    mp_real = ModelParameter(A_init, props=[prop1, prop2])

    xs_stack = []
    zs_stack = []
    for i in range(10):
        xs, zs = generate_swtiching_linear_seq(100, mp_real)
        xs_stack.append(xs)
        zs_stack.append(zs)

    # prepare initial estimate of model parameter
    prop1_est = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([0.3]))
    prop2_est = Propagator(np.ones((1, 1)), np.ones((1, 1)) * noise_std**2, np.array([-0.2]))
    A_init_est = np.array([[0.9, 0.1], [0.1, 0.9]])
    mp_est = ModelParameter(A_init_est, props=[prop1_est, prop2_est])

    # run algorithm
    hs_list = [HiddenStates.construct(2, len(xs)) for xs in xs_stack]
    loglikeli_seq = []
    for i in range(3):
        loglikeli = expectation_step(hs_list, mp_est, xs_stack)
        maximization_step(hs_list, mp_est, xs_stack)
        loglikeli_seq.append(loglikeli)

    if visualize:
        import matplotlib.pyplot as plt
        index = 0
        hs, xs = hs_list[index], xs_stack[index][:-1]
        horizons = np.arange(len(hs.z_ests))
        phases = np.array([np.argmax(z_est) for z_est in hs.z_ests])
        fig, ax = plt.subplots()
        ax.plot(horizons, xs, 'k')
        ax.plot(horizons[phases==0], xs[phases==0], 'ro')
        ax.plot(horizons[phases==1], xs[phases==1], 'bo')
        ax.legend(['sequence', 'phase 0', 'phase 1'])
        plt.show()
