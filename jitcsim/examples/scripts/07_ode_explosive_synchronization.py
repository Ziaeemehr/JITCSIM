"""

Consider the Kuramoto model on a Frequency Gap-conditioned (FGC) random network.
To construct such a network which is proper to see explosive synchronization, we make a link from node i to j only if
:math:`|\\omega_i - \\omega_j|> \\gamma`, where :math:`\\gamma` is frequency threshold [Leyva2013]_ .

Start with importing modules

.. literalinclude:: ../../jitcsim/examples/scripts/07_ode_explosive_synchronization.py
        :start-after: example-st\u0061rt
        :lines: 1-12
        :caption:

start with a low coupling strength, run the simulation and calculate the time average of order parameter and keep the last state of oscillators and use it for the next simulation with slightly increasing the coupling strength. We define a function to do this:

.. literalinclude:: ../../jitcsim/examples/scripts/07_ode_explosive_synchronization.py
        :start-after: example-st\u0061rt
        :lines: 15-40
        
then we need to make our network

.. literalinclude:: ../../jitcsim/examples/scripts/07_ode_explosive_synchronization.py
        :start-after: example-st\u0061rt
        :lines: 55-61
        :dedent: 4

define a variable which determine the direction of movement toward larger or smaller couplings and run our 
`simulateHalfLoop` function with 2 processors.

.. literalinclude:: ../../jitcsim/examples/scripts/07_ode_explosive_synchronization.py
        :start-after: example-st\u0061rt
        :lines: 85-90
        :dedent: 4

.. figure:: ../../jitcsim/examples/scripts/data/expl.png
    :scale: 50 %

    Time average of the Kuramoto order parameter in forward and backward direction and appearence of the hysteresis loop in explosive synchronization.


.. [Leyva2013] Leyva, I., Navas, A., Sendina-Nadal, I., Almendral, J.A., Buldu, J.M., Zanin, M., Papo, D. and Boccaletti, S., 2013. Explosive transitions to synchronization in networks of phase oscillators. Scientific reports, 3(1), pp.1-5.
"""

# example-start
import sys
import numpy as np
import pylab as plt
from numpy import pi
from copy import copy
from time import time
from multiprocessing import Pool
from numpy.random import uniform
from jitcsim.visualization import plot_order
from jitcsim.models.kuramoto_ode import Kuramoto_II
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

        I = Kuramoto_II(parameters)
        I.set_initial_state(prev_phases)
        data = I.simulate([Couplings[i]])
        x = data['x']
        prev_phases = x[-1, :]
        orders[i] = np.mean(I.order_parameter(x))

    return orders


if __name__ == "__main__":

    dt = 0.1
    N = 50
    t_initial = 0.0
    t_final = 1500.0
    t_transition = 400.0
    noise_amplitude = 0.0
    omega = uniform(low=0, high=1, size=N)
    initial_state = uniform(-2*pi, 2*pi, N)
    couplings = list(np.linspace(0.025, 0.040, 41))

    ki = 20
    gamma = 0.45
    alpha = 0.0
    num_processes = 2

    net = make_network()
    adj = net.fgc(N=N, k=ki, omega=omega, gamma=gamma)

    parameters = {
        'N': N,
        'adj': adj,
        't_initial': 0.,
        "t_final": t_final,
        't_transition': t_transition,
        "interval": dt,                    # time interval for sampling

        "alpha": alpha,
        "omega": omega,
        'initial_state': initial_state,

        'integration_method': 'dopri5',
        'control': ['coupling'],
        "use_omp": False,
        "output": "data",
    }

    I = Kuramoto_II(parameters)
    I.compile()

    start = time()
    args = ["forward", "backward"]

    with Pool(processes=num_processes) as pool:
        orders = (pool.map(simulateHalfLoop, args))

    r_forward, r_backward = orders

    # save orders to npz file
    np.savetxt("data/r.txt", np.column_stack((couplings,
                                              r_forward,
                                              r_backward)),
               fmt="%25.12f")

    # plotting orders
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
