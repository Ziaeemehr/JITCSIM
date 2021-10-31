"""
**Simulation of the Kuramoto model with delay.**

The system of equations reads [Yeung1999]_ :

.. math::
    \\frac{d\\theta_i}{dt} = \\omega_i + \\sum_{j=0}^{N-1} a_{i,j} \\sin(y_j(t - \\tau_{ij}) - y_i - \\alpha)

The control parameter of the model is coupling.
The output is plotting the Kuramoto order parameter vs time.


Start with importing required modules:

.. literalinclude:: ../../jitcsim/examples/scripts/01_dde_kuramoto_II_single_param.py    
        :start-after: example-st\u0061rt
        :lines: 1-6
    
setting the parameters of the model. The new item here is definition of delay matrix.

.. literalinclude:: ../../jitcsim/examples/scripts/01_dde_kuramoto_II_single_param.py    
        :start-after: example-st\u0061rt
        :lines: 10-38
        :dedent: 4

compiling and run the simulation. We can also determine type of `Dealing with initial discontinuities <https://jitcdde.readthedocs.io/en/stable/#discontinuities>`_.
To suppress warning messages use :code:`python -W ignore script.py`.

.. literalinclude:: ../../jitcsim/examples/scripts/01_dde_kuramoto_II_single_param.py        
        :start-after: example-st\u0061rt
        :lines: 40-53
        :dedent: 4


.. figure:: ../../jitcsim/examples/scripts/data/01_dde.png
    :scale: 50 %

    Kuramoto order parameter vs time for a complete network. The system of equations include delay.
    




.. [Yeung1999] Yeung, M.S. and Strogatz, S.H., 1999. Time delay in the Kuramoto model of coupled oscillators. Physical Review Letters, 82(3), p.648. Figure 3.
"""

# example-start
import numpy as np
from numpy import pi
from numpy.random import uniform
from jitcsim.visualization import plot_order
from jitcsim.models.kuramoto_dde import Kuramoto_II
from jitcsim.networks import make_network

if __name__ == "__main__":

    np.random.seed(2)

    N = 12
    alpha0 = 0.0
    sigma0 = 0.05
    coupling0 = 1.5 / (N - 1)
    omega0 = [0.5*pi] * N
    initial_state = uniform(-pi, pi, N)
    net = make_network()
    adj = net.complete(N)
    delays = adj * 2.0

    parameters = {
        'N': N,                             # number of nodes
        'adj': adj,                         # adjacency matrix
        'delays': delays,                   # matrix of delays
        't_initial': 0.,                    # initial time of integration
        "t_final": 100,                     # final time of integration
        't_transition': 10.0,               # transition time
        "interval": 0.2,                    # time interval for sampling

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
    data = sol.simulate(controls, disc="blind", rtol=0, atol=1e-5)
    x = data['x']
    t = data['t']

    order = sol.order_parameter(x)
    plot_order(t,
               order,
               filename="data/01_dde.png",
               xlabel="time",
               ylabel="r(t)",
               close_fig=False)
    
    print(np.mean(order))
