import numpy as np
import pylab as plt
import networkx as nx
from os.path import join
from numpy.random import rand
from jitcsim.models.montbrio_sde import MontbrioNetwork

# from numpy import pi, sqrt
# from scipy.ndimage import gaussian_filter1d


if __name__ == "__main__":

    seed = 2
    np.random.seed(seed)

    adj = np.loadtxt(join("data", "weights.txt"))
    adj = adj/np.max(adj)
    assert(abs(np.trace(adj)) < 1e-8)

    # adj = np.array([[0, 1],
    #                 [1, 0]])
    N = adj.shape[0]
    initial_state = np.asarray([rand(), rand()] * N)
    initial_state[:N] = initial_state[:N] * 1.5
    initial_state[N:2*N] = initial_state[N:2*N] * 3 - 2.8

    parameters = {
        "N": N,
        't_initial': 0.,                    # initial time of integration
        "t_final": 1000.0,                  # final time of integration
        't_transition': 0.0,                # transition time
        "interval": 0.1,                    # time interval for sampling

        "adj": adj,
        "dimension": 2,
        "I_app": 0,
        "J": 14.5,
        "tau": 1.0,
        "eta": -4.6,
        "Delta": 0.7,
        "coupling": 0.0,
        'initial_state': initial_state,         # initial phase of oscillators
        "sigma_r": 0.018,                       # noise amplitude on firing rate
        "sigma_v": 0.0255,                      # noise amplitude on membraine potenrial
        'control': ["coupling"],                # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "output",                   # output directory
    }

    sol = MontbrioNetwork(parameters)
    sol.compile()

    couplings = np.arange(0, 3, 0.1)
    for itr in range(len(couplings)):
        sol = MontbrioNetwork(parameters)

        sol.set_integrator_parameters(
            atol=1e-4,
            rtol=1e-4,
            min_step=1e-10,
            max_step=1.0)
        sol.set_seed(seed)

        data = sol.simulate([couplings[itr]])
        r = data['r']
        v = data['v']
        t = data['t']

        fig, ax = plt.subplots(nrows=2, sharex=True)

        step = 1
        for i in range(N):
            ax[0].plot(t[::step], r[::step, i], alpha=0.4, lw=0.3)
            ax[1].plot(t[::step], v[::step, i], alpha=0.4, lw=0.3)
            # ax[1].plot(t, filter1d[:, i], alpha=0.4, label="f")
        ax[0].set_ylabel("r")
        ax[1].set_ylabel("v")

        plt.savefig("output/01_sde_montbrio_net_{:.6f}.png".format(couplings[itr]), dpi=150)
        plt.close()


# "E0": 0.4,                              #!! review the parameters value
# "V0": 4.0,
# "k1": 2.8,
# "k2": 0.8,
# "k3": 0.5,
# "alpha": 0.32,
# "tau_s": 1.54,
# "tau_f": 1.44,
# "tau_o": 0.98,
# "epsilon": 0.5,
# filter1d = gaussian_filter1d(v, sigma=4, axis=0)

# # Baloon model parameters
# "E0": 0.4,                              #!! review the parameters value
# "V0": 4.0,
# "k1": 2.77264,
# "k2": 0.572,
# "k3": -0.43,
# "alpha": 0.5,
# "tau_s": 1.25,
# "tau_f": 2.5,
# "tau_o": 1.02040816327,
# "epsilon": 0.5,
