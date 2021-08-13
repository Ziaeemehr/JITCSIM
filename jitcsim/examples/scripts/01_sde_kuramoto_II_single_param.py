"""
**Simulation of the Kuramoto model with noise.**

.. math::
        \\frac{d\\theta_i}{dt} = \\omega_i + \\xi_i + \\sum_{j=0}^{N-1} a_{i,j} \\sin(y_j - y_i - \\alpha)  

where :math:`\\xi_i` is white noise for each osillator.

The control parameter of the model is coupling.
The initial phase also could be changed in repeated simulations.
The output is plotting the Kuramoto order parameter vs time.

Start with importing modules

.. literalinclude:: ../../jitcsim/examples/scripts/01_sde_kuramoto_II_single_param.py    
        :start-after: example-st\u0061rt
        :lines: 1-6
        :caption:

define the amplitude of the noise with normal distribution, set the parameters and runnig the simulation:

.. literalinclude:: ../../jitcsim/examples/scripts/01_sde_kuramoto_II_single_param.py    
        :start-after: example-st\u0061rt
        :lines: 10-47
        :dedent: 4
        
and finally plotting the order parameter:

.. literalinclude:: ../../jitcsim/examples/scripts/01_sde_kuramoto_II_single_param.py    
        :start-after: example-st\u0061rt
        :lines: 49-57
        :dedent: 4

.. figure:: ../../jitcsim/examples/scripts//data/01_sde.png
    :scale: 50 %

    Kuramoto order parameter vs time for complete network. The system of equations include noise.




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

    controls = [coupling0]
    data = sol.simulate(controls)
    x = data['x']
    t = data['t']

    # calculate the Kuramoto order parameter
    order = sol.order_parameter(x)

    # plot order parameter vs time
    plot_order(t,
               order,
               filename="data/01_sde.png",
               xlabel="time", 
               ylabel="r(t)")
