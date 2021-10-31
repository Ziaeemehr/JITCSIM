"""

If you try to run a simulation multiple times and for each simulation change a parameter, it will be more efficient to avoid multiple compiling the model and use control parameter.
In next example we consider the `coupling` as a control parameter:

The initial phase also could be changed in repeated simulations.
The output is plotting the time average of the Kuramoto order parameter vs coupling.
Only difference with respect to the previous examples is shown:

.. literalinclude:: ../../jitcsim/examples/scripts/02_ode_kuramoto_II_single_param_repeated_run.py    
        :start-after: example-st\u0061rt
        :lines: 15, 23, 36, 39
        :dedent: 4
        :caption:

to make an instance of the model and measure the compilation time:
(time module need to be imported)

.. literalinclude:: ../../jitcsim/examples/scripts/02_ode_kuramoto_II_single_param_repeated_run.py    
        :start-after: example-st\u0061rt
        :lines: 41-44
        :dedent: 4

then define an array for the various coupling strenghts
and an zero array to record the order parameter at different couplings:

.. literalinclude:: ../../jitcsim/examples/scripts/02_ode_kuramoto_II_single_param_repeated_run.py    
        :start-after: example-st\u0061rt
        :lines: 46-47
        :dedent: 4

we need a loop over each coupling and repeat the simulation `num_ensembles` times. The initial state of the oscillators can also be changed for each simulation using :py:meth:`set_initial_state`.

.. literalinclude:: ../../jitcsim/examples/scripts/02_ode_kuramoto_II_single_param_repeated_run.py    
        :start-after: example-st\u0061rt
        :lines: 49-61
        :dedent: 4

and finally plot the time averaged order parameter vs coupling strength:

.. literalinclude:: ../../jitcsim/examples/scripts/02_ode_kuramoto_II_single_param_repeated_run.py    
        :start-after: example-st\u0061rt
        :lines: 63-67
        :dedent: 4

.. code-block:: bash

        saving file to data/km.so
        Compile time : 1.578 secondes.
        Simulation time: 3.385 seconds



.. figure:: ../../jitcsim/examples/scripts//data/02.png
    :scale: 50 %

    Time average of the Kuramoto order parameter vs coupling strength.

"""



# example-start
import numpy as np
from numpy import pi
from time import time
from numpy.random import uniform, normal
from jitcsim.visualization import plot_order
from jitcsim.models.kuramoto_ode import Kuramoto_II
from jitcsim.networks import make_network


if __name__ == "__main__":

    np.random.seed(1)

    N = 30
    num_ensembles = 10
    omega0 = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)

    # make complete network
    net = make_network()
    adj = net.complete(N)

    parameters = {
        'N': N,
        'adj': adj,
        't_initial': 0.,
        "t_final": 100,
        't_transition': 20.0,
        "interval": 1.0,                    # time interval for sampling

        "alpha": 0.0,
        "omega": omega0,
        'initial_state': initial_state,

        'integration_method': 'dopri5',
        'control': ['coupling'],
        "use_omp": False,
        "output": "data",
    }

    sol = Kuramoto_II(parameters)
    compile_time = time()
    sol.compile()
    print("Compile time : {:.3f} secondes.".format(time()-compile_time))

    couplings = np.arange(0, 0.8, 0.05) / (N-1)
    orders = np.zeros((len(couplings), num_ensembles))

    start_time = time()
    
    for i in range(len(couplings)):
        for j in range(num_ensembles):

            controls = [couplings[i]]
            sol.set_initial_state(uniform(-pi, pi, N))
            data = sol.simulate(controls)
            x = data['x']
            t = data['t']
            orders[i, j] = np.mean(sol.order_parameter(x))

    print("Simulation time: {:.3f} seconds".format(time()-start_time))

    plot_order(couplings,
               np.mean(orders, axis=1),
               filename="data/02.png",
               ylabel="R",
               xlabel="coupling")
