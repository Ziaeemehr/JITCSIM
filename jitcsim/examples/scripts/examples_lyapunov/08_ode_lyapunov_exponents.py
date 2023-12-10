"""

In this example we are going to calculate the Lyapunov exponent spectrum for the Kuramoto model on feed forward loop.

The output is Lyapunov exponent (LE) spectrum vs time.

Start with importing modules

.. literalinclude:: ../../jitcsim/examples/scripts/08_ode_lyapunov_exponents.py
        :start-after: example-st\u0061rt
        :lines: 1-6
        :caption:

define feed forward loop adjacency matrix and set the parameters:

.. literalinclude:: ../jitcsim/examples/scripts/08_ode_lyapunov_exponents.py
        :start-after: example-st\u0061rt
        :lines: 21-44
        :dedent: 4

`n_lyap` is a new parameter which determine how many of the LEs need to be calculate.
Usually we need 2 or 3 largets LE. Increasing the number of LEs make the compilation time extremly long for large networks and the simulation time also will be much longer.

`Lyap_Kuramoto_II` class is defined for calculation of LEs:

.. literalinclude:: ../jitcsim/examples/scripts/08_ode_lyapunov_exponents.py
        :start-after: example-st\u0061rt
        :lines: 46-52
        :dedent: 4

.. figure:: ../jitcsim/examples/scripts/data/lyap.png
    :scale: 80 %

    The LEs of the Kuramoto oscillators on feed forward loop.

"""

# example-start
import numpy as np
from numpy import pi
import pylab as plt
from numpy.random import uniform, normal
from jitcsim.models.kuramoto_ode import Lyap_Kuramoto_II
from jitcsim.visualization import plot_lyaps


if __name__ == "__main__":

    np.random.seed(2)

    N = 3
    alpha0 = 0.0
    coupling0 = 1.0 / (N - 1)
    omega0 = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)
    FeedBackLoop = np.asarray([[0, 1, 0],
                               [0, 0, 1],
                               [1, 0, 0]])
    FeedForwardLoop = np.asarray([[0, 1, 1],
                                  [0, 0, 1],
                                  [0, 0, 0]])

    parameters = {
        'N': N,                             # number of nodes
        "adj": FeedForwardLoop,             # adjacency matrix
        't_initial': 0.,                    # initial time of integration
        "t_final": 10000,                     # final time of integration
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

    sol = Lyap_Kuramoto_II(parameters)
    sol.compile()

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
