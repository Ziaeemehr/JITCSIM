"""
Calculate the Lyapunov exponents of the Kuramoto model.
The output is Lyapunov exponents vs time.
"""


import numpy as np
from numpy import pi
import pylab as plt
from numpy.random import uniform, normal
from jitcsim.models.kuramoto import Lyap_Kuramoto_II
from jitcsim.visualization import plot_lyaps


if __name__ == "__main__":

    np.random.seed(2)

    N = 3
    alpha0 = 0.0
    coupling0 = 1.0 / (N - 1)
    omega0 = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)
    FeedBackLoop = np.asarray([[0, 0, 1],
                               [1, 0, 0],
                               [0, 1, 0]])
    FeedForwardLoop = np.asarray([[0, 0, 0],
                                  [1, 0, 0],
                                  [1, 1, 0]])

    parameters = {
        'N': N,                             # number of nodes
        # 'adj': FeedBackLoop,                # adjacency matrix
        "adj": FeedForwardLoop,
        't_initial': 0.,                    # initial time of integration
        "t_final": 1000,                     # final time of integration
        't_transition': 0.0,                # transition time
        "interval": 0.5,                    # time interval for sampling

        "alpha": alpha0,                    # frustration
        "omega": omega0,                    # initial angular frequencies
        'initial_state': initial_state,     # initial phase of oscillators

        'n_lyap': 3,                        # number of the Lyapunov exponents to calculate

        'integration_method': 'RK45',       # integration method
        'control': ['coupling'],            # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    # make an instance of the model
    sol = Lyap_Kuramoto_II(parameters)
    # compile the model
    sol.compile()

    # run the simulation by setting the control parameters
    controls = [coupling0]
    data = sol.simulate(controls)
    x = data['lyap']
    t = data['t']

    fig, ax = plt.subplots(1)
    colors = ["r", "b", "g"]
    for i in range(3):
        plot_lyaps(t,
                   x[:, i],
                   ax=ax,
                   xlabel="time",
                   ylabel="LE",
                   color=colors[i],
                   xlog=True)
    plt.savefig("data/lyap.png")
