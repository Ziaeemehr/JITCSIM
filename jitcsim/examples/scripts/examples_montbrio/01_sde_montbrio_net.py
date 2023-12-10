import numpy as np
import pylab as plt
from time import time
import networkx as nx
from numpy import pi, sqrt
from jitcsim.utility import display_time
from scipy.ndimage import gaussian_filter1d
from numpy.random import uniform, normal, rand
from jitcsim.models.montbrio_sde import MontbrioNetwork


if __name__ == "__main__":

    np.random.seed(2)

    N = 68
    
    
    coupling = 0.9621
    eta = -4.6  # * Delta
    J = 14.5  # * sqrt(Delta)
    # adj = nx.to_numpy_array(nx.complete_graph(N), dtype=int)
    adj = np.loadtxt("data/weights.txt")
    adj_normal = adj / np.max(adj)
    # initial_state = [0.01, -2.0, 0.05, -1] # for 2 nodes
    initial_state = rand(2 * N)
    initial_state[1::2] = initial_state[1::2] * 3 - 1.5
    initial_state[::2]= initial_state[::2]* 0.1

    parameters = {
        "N": N,
        't_initial': 0.,                    # initial time of integration
        "t_final": 10_000.0,                  # final time of integration
        't_transition': 0.0,                # transition time
        "interval": 10,                    # time interval for sampling

        "adj": adj_normal,
        "dimension": 2,
        "I_app": 0,
        "J": 14.5,
        "tau": 1.0,
        "eta": -4.6,
        "Delta": 0.7,
        # "coupling": coupling,
        'initial_state': initial_state,     # initial phase of oscillators
        "sigma_r": 0.05,                        # noise amplitude on firing rate
        "sigma_v": 0.1,                         # noise amplitude on membraine potenrial

        'integration_method': 'dopri5',     # integration method
        'control': ["coupling"],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "output",                   # output directory
    }
    
    sol = MontbrioNetwork(parameters)
    sol.compile()

    couplings = np.arange(.1, 3.71, 0.1)
    # couplings = [2.69]
    for itr in range(len(couplings)):
        
        tic = time()
        sol = MontbrioNetwork(parameters)
        sol.set_integrator_parameters(
                atol=1e-3,
                rtol=1e-2,
                min_step=1e-5,
                max_step=10.0)

        data = sol.simulate([couplings[itr]])
        display_time(time() - tic)
        
        r = data['r']
        v = data['v']
        t = data['t']

        fig, ax = plt.subplots(2, sharex=True)
        st = 1
        for i in range(0, N, 1):
            ax[0].plot(t[::st], r[::st, i], lw=0.2, alpha=0.2)
            ax[1].plot(t[::st], v[::st, i], alpha=0.2, ls="-", lw=0.2)
        
        ax[0].set_ylabel("r")
        ax[1].set_ylabel("v")
        ax[0].set_ylim(-.1, 2)
        ax[1].set_ylim(-2.2, 2)
        ax[0].set_title("G = {:.3f}".format(couplings[itr]))

        plt.savefig("output/01_sde_montbrio_{:.3f}.png".format(couplings[itr]), dpi=150)
        plt.close()
