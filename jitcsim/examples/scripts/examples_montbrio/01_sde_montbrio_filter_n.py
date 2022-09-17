import numpy as np
import networkx as nx
from numpy import pi, sqrt
from scipy.ndimage import gaussian_filter1d
from numpy.random import uniform, normal, rand
from jitcsim.models.montbrio_sde import MontbrioNetwork


if __name__ == "__main__":

    np.random.seed(2)

    N = 2
    tau = 1
    Delta = 2.0
    coupling = 1.0
    eta = -5 * Delta
    J = 15  * sqrt(Delta)
    adj = nx.to_numpy_array(nx.complete_graph(N), dtype=int) 
    adj = rand(N, N) - .5
    # initial_state = [0.01, -2.0, 0.05, -1] # for 2 nodes
    initial_state = [rand(), rand()] * N
    
    parameters = {
        "N" : N,
        't_initial': 0.,                    # initial time of integration
        "t_final": 40.0,                     # final time of integration
        't_transition': 0.0,                # transition time
        "interval": 0.1,                    # time interval for sampling

        "adj" : adj,
        "dimension": 2,
        "I_app": 2,
        "J": J,
        "tau": tau,
        "eta": eta,
        "Delta": Delta,
        "coupling" : coupling,
        'initial_state': initial_state,     # initial phase of oscillators
        "sigma_r": 0.01,                        # noise amplitude on firing rate
        "sigma_v": 0.01,                        # noise amplitude on membraine potenrial
        
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

        'integration_method': 'dopri5',     # integration method
        'control': [],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }
    sol = MontbrioNetwork(parameters)
    sol.compile()
    data = sol.simulate()
    r = data['r']
    v = data['v']
    t = data['t']
    filter1d = gaussian_filter1d(v, sigma=4, axis=0)

    import pylab as plt 
    fig, ax = plt.subplots(2, sharex=True)

    for i in range(1):
        ax[0].plot(t, r[:, i])
        ax[1].plot(t, v[:, i], alpha=0.4, label="v", ls="--")
        ax[1].plot(t, filter1d[:, i], alpha=0.4, label="f")
    ax[0].set_ylabel("r")
    ax[1].set_ylabel("v")
    ax[1].legend()

    plt.savefig("data/01_sde_montbrio_filter_n.png", dpi=150)
    plt.show()




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