import numpy as np
import networkx as nx
from numpy import pi, sqrt
from numpy.random import uniform, normal
from jitcsim.models.montbrio import Montbrio_n

if __name__ == "__main__":

    np.random.seed(2)

    N = 2
    tau = 1
    Delta = 2.0
    coupling = 1.0
    eta = -5 * Delta
    J = 15  * sqrt(Delta)
    adj = nx.to_numpy_array(nx.complete_graph(N), dtype=int) 
    adj = np.random.rand(N, N) - .5
    # initial_state = [0.01, -2.0, 0.05, -1] # for 2 nodes
    initial_state = np.random.rand(2 * N) * 2
    
    parameters = {
        "N" : N,
        't_initial': 0.,                    # initial time of integration
        "t_final": 40.0,                     # final time of integration
        't_transition': 0.0,                # transition time
        "interval": 0.1,                    # time interval for sampling

        "adj" : adj,
        "dimension": 2,
        "I_app": 1,
        "J": J,
        "tau": tau,
        "eta": eta,
        "Delta": Delta,
        "coupling" : coupling,
        'initial_state': initial_state,     # initial phase of oscillators
        "sigma_r": 0.005,                       # noise amplitude on firing rate
        "sigma_v": 0.01,                        # noise amplitude on membraine potenrial
        
        # Baloon model parameters
        "E0": 0.4,                              #!! review the parameters value
        "V0": 4.0,
        "k1": 2.77264,
        "k2": 0.572,
        "k3": -0.43,
        "alpha": 5.0,
        "taus": 1.25,
        "tauf": 2.5,
        "tauo": 1.02040816327,
        "epsilon": 0.5,

        'integration_method': 'dopri5',     # integration method
        'control': [],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    sol = Montbrio_n(parameters)
    sol.compile()

    data = sol.simulate([])
    r = data['r']
    v = data['v']
    t = data['t']

    import pylab as plt 
    fig, ax = plt.subplots(2, sharex=True)

    for i in range(N):
        ax[0].plot(t, r[:, i], label="r")
        ax[1].plot(t, v[:, i], label="v")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("data/01_montbrio_fast_n.png", dpi=150)

    plt.show()

