"""
plot kuramoto order parameter vs time for second order Kuramoto model.
"""

import numpy as np
from math import pi
import pylab as plt
from numpy.random import uniform
from jitcsim.networks import make_network
from jitcsim.visualization import plot_order, plot_phases
from jitcsim.models.kuramoto import SOKM_SingleLayer


if __name__ == "__main__":

    np.random.seed(2)

    dt = 1                                      # time interval sampling
    N = 10                                      # number of nodes
    m = 1.0                                     # mass
    k_ave = 9                                   # the average degree of the network
    inv_m = 1.0 / m

    t_initial = 0.0                             # initial time of the simulation
    t_final = 1000.0                             # final time of the simulation
    t_transition = 0.0                          # transition time

    omega = uniform(-1, 1, N)                   # initial angular frequencies
    omega.sort()

    initial_state = uniform(-2*pi, 2*pi, 2*N)   # initial state of oscillators
    coupling = 0.8 / k_ave                      # coupling strength

    alpha = 0.0                                 # frustration
    net = make_network()
    adj = net.complete(N)                       # adjacency matrix

    num_processes = 1                           # number of processors

    parameters = {
        'N': N,
        'adj': adj,
        't_initial': 0.,
        "t_final": t_final,
        't_transition': t_transition,
        "interval": dt,

        "inv_m": inv_m,
        "alpha": alpha,
        "omega": omega,
        'initial_state': initial_state,

        'integration_method': 'dopri5',
        'control': ['coupling'],
        "use_omp": False,
        "output": "data",
    }

    I = SOKM_SingleLayer(parameters)
    I.compile()

    controls = [coupling]
    data = I.simulate(controls)
    x = data['x']
    t = data['t']

    phases = x[:, :N]

    order = I.order_parameter(phases)

    fig, ax = plt.subplots(2, sharex=True)

    plot_order(t,
               order,
               ax=ax[0],
               xlabel="time",
               ylabel="r(t)")
    plot_phases(phases, [0, t[-1], 0, N], ax=ax[1], xlabel="time", ylabel="# index")
    plt.savefig("data/phases.png", dpi=150)
    plt.close()
    
