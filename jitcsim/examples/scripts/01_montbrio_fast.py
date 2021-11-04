import numpy as np
from numpy import pi, sqrt
from numpy.random import uniform, normal
from jitcsim.models.montbrio import Montbrio_fast

if __name__ == "__main__":

    # np.random.seed(2)

    Delta = 2.0
    tau = 1
    eta = -5 * Delta
    J = 15  * sqrt(Delta)
    initial_state = [0.01, -2.0]
    
    parameters = {
        't_initial': 0.,                    # initial time of integration
        "t_final": 40.0,                     # final time of integration
        't_transition': 0.0,                # transition time
        "interval": 0.1,                    # time interval for sampling

        "dimension": 2,
        "J": J,
        "tau": tau,
        "eta": eta,
        "Delta": Delta,
        'initial_state': initial_state,     # initial phase of oscillators

        'integration_method': 'dopri5',     # integration method
        'control': [],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    sol = Montbrio_fast(parameters)
    sol.set_step_current(t_start=20, amplitude=3)
    sol.compile()

    data = sol.simulate([])
    x = data['x']
    t = data['t']


    import pylab as plt 
    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(t, x[:, 0], label="r")
    ax[1].plot(t, x[:, 1], label="v")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("data/01_montbrio.png", dpi=150)

    plt.show()

