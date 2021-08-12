"""
**Simulation of the Kuramoto model on complete network.**

where :math:`\\theta_i` is phase of oscillator i, :math:`\\omega_i` the angular frequency of oscillator i, :math:`\\alpha` the frustration, :math:`N` the number of oscillators, :math:`a_{i,j}` is element of the adjacency matrix, :math:`a_{i,j}=1` if there is a directed link from the node j to i; otherwise :math:`a_{i,j}=0`. 

The parameter of the model is coupling.
The initial phase also could be changed in repeated simulations.
The output is plotting the Kuramoto order parameter vs time.
"""


import numpy as np
from numpy import pi
import networkx as nx
from numpy.random import uniform, normal
from jitcsim.models.kuramoto import Kuramoto_II
from jitcsim.visualization import plot_order
from jitcsim.networks import make_network


if __name__ == "__main__":

    np.random.seed(1)

    N = 30
    alpha0 = 0.1
    coupling0 = 0.5 / (N - 1)
    omega0 = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)

    # make complete network
    net = make_network()
    adj = net.complete(N)

    parameters = {
        'N': N,                             # number of nodes
        'adj': adj,                         # adjacency matrix
        't_initial': 0.,                    # initial time of integration
        "t_final": 100,                     # final time of integration
        't_transition': 2.0,                # transition time
        "interval": 1.0,                    # time interval for sampling

        "alpha": alpha0,                    # frustration
        "omega": omega0,                    # initial angular frequencies
        'initial_state': initial_state,     # initial phase of oscillators

        'integration_method': 'dopri5',     # integration method
        'control': ['coupling'],            # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }

    # make an instance of the model
    sol = Kuramoto_II(parameters)
    # compile the model
    sol.compile()

    # run the simulation by setting the control parameters
    controls = [coupling0]
    data = sol.simulate(controls)
    x = data['x']
    t = data['t']

    # calculate the Kuramoto order parameter
    order = sol.order_parameter(x)

    # plot order parameter vs time
    plot_order(t,
               order,
               filename="data/01.png",
               xlabel="time", 
               ylabel="r(t)")
