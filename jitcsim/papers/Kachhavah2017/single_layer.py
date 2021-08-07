"""
Kachhvah, A.D. and Jalan, S., 2017. Multiplexing induced explosive synchronization in Kuramoto oscillators with inertia. EPL (Europhysics Letters), 119(6), p.60005.
Fig 2.b. Synchronization transition plots (R vs. coupling) of a single
layer ER network.
"""

import sys
import numpy as np
from math import pi
import pylab as plt
from copy import copy
from time import time
from multiprocessing import Pool
from numpy.random import uniform
from jitcsim.visualization import plot_order
from jitcsim.models.kuramoto import SOKM_Single
from jitcsim.utility import display_time
from jitcsim.networks import make_network


def simulateHalfLoop(direction):

    if direction == "backward":
        Couplings = copy(couplings[::-1])
    else:
        Couplings = copy(couplings)

    n = len(Couplings)
    orders = np.zeros(n)

    prev_phases = parameters['initial_state']

    for i in range(n):

        print("direction = {:10s}, coupling = {:10.6f}".format(
            direction, Couplings[i]))
        sys.stdout.flush()

        I = SOKM_Single(parameters)
        I.set_initial_state(prev_phases)
        data = I.simulate([Couplings[i]])
        x = data['x']
        prev_phases = x[-1, :]
        orders[i] = np.mean(I.order_parameter(x[:, :N]))

    return orders


if __name__ == "__main__":

    np.random.seed(1)

    dt = 1                                      # time interval sampling
    N = 100                                     # number of nodes
    m = 1.0                                     # mass
    k_ave = 12                                  # the average degree of the network
    inv_m = 1.0 / m

    t_initial = 0.0                             # initial time of the simulation
    t_final = 500.0                             # final time of the simulation
    t_transition = 0.0                          # transition time

    omega = uniform(-1, 1, size=N)              # initial angular frequencies
    omega.sort()

    initial_state = uniform(-2*pi, 2*pi, 2*N)   # initial state of oscillators
    couplings = np.linspace(0.5, 2, 50) / k_ave  # coupling strength

    alpha = 0.0                                 # frustration
    net = make_network(seed=2)
    adj = net.erdos_renyi(N, 0.12)              # adjacency matrix

    num_processes = 2                           # number of processors

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

    I = SOKM_Single(parameters)
    I.compile()

    start = time()
    args = ["forward", "backward"]

    with Pool(processes=num_processes) as pool:
        orders = (pool.map(simulateHalfLoop, args))

    r_forward, r_backward = orders

    # save orders to npz file
    np.savetxt("data/R.txt", np.column_stack((couplings,
                                              r_forward,
                                              r_backward)),
               fmt="%25.12f")

    # plotting 
    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, figsize=(10, 4.5))
    plot_order(couplings, r_forward,
               label="FW",
               close_fig=False,
               ax=ax,
               color="b",
               marker="*")
    plot_order(couplings[::-1],
               r_backward,
               label="BW",
               ax=ax,
               color="r",
               marker="o",
               close_fig=False,
               xlabel="coupling",
               ylabel="R")
    plt.savefig("data/expl.png", dpi=150)
    display_time(time()-start)
