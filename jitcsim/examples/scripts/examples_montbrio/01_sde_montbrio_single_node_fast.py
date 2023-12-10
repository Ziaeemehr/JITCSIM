import numpy as np
from numpy import pi, sqrt
from numpy.random import uniform, normal
from jitcsim.models.montbrio_sde import MontbrioSingleNode

if __name__ == "__main__":

    N = 1
    tau = 1
    Delta = 1.0
    eta = -5
    J = 15
    initial_state = [0.01, .0]
    
    parameters = {
        "N":1,
        't_initial': 0.,                    # initial time of integration
        "t_final": 100.0,                     # final time of integration
        't_transition': 0.0,                # transition time
        "interval": 0.1,                    # time interval for sampling

        "dimension": 2,
        "I_app" : 4,
        "J": J,
        "tau": tau,
        "eta": eta,
        "sigma_r": 0.05,
        "sigma_v": 0.1,
        "Delta": Delta,
        'initial_state': initial_state,     # initial phase of oscillators
        # 'integration_method': 'dopri5',     # integration method
        'control': [],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    sol = MontbrioSingleNode(parameters)
    sol.compile()

    data = sol.simulate()
    x = data['x']
    t = data['t']

    import pylab as plt 
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(t, x[:, 0], label="r", lw=1)
    ax[1].plot(t, x[:, 1], label="v", lw=1)
    ax[0].legend()
    ax[1].legend()
    plt.savefig("data/01_sde_montbrio_single_node.png", dpi=150)

    plt.show()

