"""
**Simulation of the Kuramoto model with noise.**

.. math::
        \\frac{d\\theta_i}{dt} = \\omega_i + \\xi_i + \\sum_{j=0}^{N-1} a_{i,j} \\sin(y_j - y_i - \\alpha)  

where :math:`\\xi_i` is white noise for each osillator.
Here it's possible to set integration absolute and relative error from 
`set_integrator_parameters`.
"""

# example-start
import numpy as np
from numpy import pi
from numpy.random import uniform, normal
from jitcsim.visualization import plot_order
from jitcsim.models.kuramoto_sde import Kuramoto_II
from jitcsim.networks import make_network

if __name__ == "__main__":

    np.random.seed(2)

    N = 30
    alpha0 = 0.0
    sigma0 = 0.05
    coupling0 = 0.5 / (N - 1)
    omega0 = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)

    net = make_network()
    adj = net.complete(N)

    parameters = {
        'N': N,                             # number of nodes
        'adj': adj,                         # adjacency matrix
        't_initial': 0.,                    # initial time of integration
        "t_final": 100,                     # final time of integration
        't_transition': 2.0,                # transition time
        "interval": 1.0,                    # time interval for sampling

        "sigma": sigma0,                    # noise amplitude (normal distribution)
        "alpha": alpha0,                    # frustration
        "omega": omega0,                    # initial angular frequencies
        'initial_state': initial_state,     # initial phase of oscillators

        'control': ['coupling'],            # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    sol = Kuramoto_II(parameters)
    sol.compile()
    sol.set_integrator_parameters(atol=1e-6, rtol=1e-3)

    controls = [coupling0]
    data = sol.simulate(controls)
    x = data['x']
    t = data['t']

    # calculate the Kuramoto order parameter
    order = sol.order_parameter(x)

    # plot order parameter vs time
    plot_order(t,
               order,
               filename="data/02_sde.png",
               xlabel="time", 
               ylabel="r(t)")
