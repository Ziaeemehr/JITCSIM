"""
**Simulation of the Kuramoto model on complete network.**

.. math::
    \\frac{d\\theta_i}{dt} &= \\omega_i + \\sum_{j=0}^{N-1} a_{i,j} \\sin(y_j - y_i - \\alpha) 

where :math:`\\theta_i` is phase of oscillator i, :math:`\\omega_i` angular frequency of oscillator i, :math:`\\alpha` frustration, :math:`N` number of oscillators, :math:`a_{i,j}` is an element of the adjacency matrix, :math:`a_{i,j}=1` if there is a directed link from the node j to i; otherwise :math:`a_{i,j}=0`. 

The output of the example is plotting the Kuramoto order parameter vs time.




Starting with a few imports

.. literalinclude:: ../../jitcsim/examples/scripts/00_ode_kuramoto_II.py    
        :start-after: example-st\u0061rt
        :lines: 1-6
        :caption:

then we set the parameters of the system. We consier a complete network with :math:`N` oscillators.
        
.. literalinclude:: ../../jitcsim/examples/scripts/00_ode_kuramoto_II.py    
        :start-after: example-st\u0061rt
        :lines: 11-38
        :dedent: 4

make an instance of the model and pass the parameters to it. 
Then compile the model.

.. literalinclude:: ../../jitcsim/examples/scripts/00_ode_kuramoto_II.py    
        :start-after: example-st\u0061rt
        :lines: 40-41
        :dedent: 4

and run the simulation

.. literalinclude:: ../../jitcsim/examples/scripts/00_ode_kuramoto_II.py    
        :start-after: example-st\u0061rt
        :lines: 43-45
        :dedent: 4

to calculate and plot the order parameter of the system vs time:

.. literalinclude:: ../../jitcsim/examples/scripts/00_ode_kuramoto_II.py    
        :start-after: example-st\u0061rt
        :lines: 47-53
        :dedent: 4


.. figure:: ../../jitcsim/examples/scripts//data/00.png
    :scale: 50 %

    Kuramoto order parameter vs time, for a complete network at a fixed coupling strength.

"""
# example-start
import numpy as np
from numpy import pi
from numpy.random import uniform, normal
from jitcsim.models.kuramoto_ode import Kuramoto_II
from jitcsim import plot_order
from jitcsim import make_network


if __name__ == "__main__":

    np.random.seed(1)
    N = 30
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

        "coupling": 0.5 / (N - 1),          # coupling strength
        "alpha": 0.0,                       # frustration
        "omega": omega0,                    # initial angular frequencies
        'initial_state': initial_state,     # initial phase of oscillators

        'integration_method': 'dopri5',     # integration method
        'control': [],                      # control parameters

        "use_omp": False,                   # use OpenMP
        "output": "data",                   # output directory
    }
    
    sol = Kuramoto_II(parameters)
    sol.compile()
    
    data = sol.simulate([])
    x = data['x']
    t = data['t']

    order = sol.order_parameter(x)

    plot_order(t,
               order,
               filename="data/00.png",
               xlabel="time", 
               ylabel="r(t)")

    # example-end