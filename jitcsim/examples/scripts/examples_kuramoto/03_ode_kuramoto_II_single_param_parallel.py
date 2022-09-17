"""
In this example we use **multiprocessing** to speed up the computation by runnig the programm in parallel.
The parameter of the model is coupling.

The output is plotting the time average of the Kuramoto order parameter vs coupling.

The only difference with respect to the previous examples is as follows:

.. literalinclude:: ../../jitcsim/examples/scripts/03_ode_kuramoto_II_single_param_parallel.py
        :start-after: example-st\u0061rt
        :lines: 4
        :caption:

.. literalinclude:: ../../jitcsim/examples/scripts/03_ode_kuramoto_II_single_param_parallel.py
        :start-after: example-st\u0061rt
        :lines: 15-17, 42-52
        :dedent: 4

after compiling we need to provide the `par` and make a pool:

.. literalinclude:: ../../jitcsim/examples/scripts/03_ode_kuramoto_II_single_param_parallel.py
        :start-after: example-st\u0061rt
        :lines: 65-72
        :dedent: 4



"""

# example-start
import numpy as np
from numpy import pi
from time import time
from multiprocessing import Pool
from numpy.random import uniform, normal
from jitcsim.visualization import plot_order
from jitcsim.models.kuramoto_ode import Kuramoto_II
from jitcsim.networks import make_network


if __name__ == "__main__":

    np.random.seed(1)

    N = 30
    num_ensembles = 100
    num_processes = 4                       # number of processes
    alpha0 = 0.0
    omega0 = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)
    
    net = make_network()
    adj = net.complete(N)

    parameters = {
        'N': N,
        'adj': adj,
        't_initial': 0.,
        "t_final": 100,
        't_transition': 20.0,
        "interval": 1.0,                    # time interval for sampling

        "alpha": alpha0,
        "omega": omega0,
        'initial_state': initial_state,

        'integration_method': 'dopri5',
        'control': ['coupling'],
        "use_omp": False,
        "output": "data",
    }

    def run_for_each(coupl):

        controls = [coupl]
        I = Kuramoto_II(parameters)
        I.set_initial_state(uniform(-pi, pi, N))
        data = I.simulate(controls)
        x = data['x']
        order = np.mean(I.order_parameter(x))

        return order

    # make an instance of the model
    sol = Kuramoto_II(parameters)
    compile_time = time()
    sol.compile()
    print("Compile time : {:.3f} secondes.".format(time() - compile_time))

    couplings = np.arange(0, 0.8, 0.05) / (N-1)

    start_time = time()

    # prepare parameters for run in parallel
    par = []
    for i in range(len(couplings)):
        for j in range(num_ensembles):
            par.append(couplings[i])

    with Pool(processes=num_processes) as pool:
        orders = (pool.map(run_for_each, par))
    orders = np.reshape(orders, (len(couplings), num_ensembles))

    print("Simulation time: {:.3f} seconds".format(time()-start_time))

    # plotting time average of the order parameters vs coupling
    plot_order(couplings,
               np.mean(orders, axis=1),
               filename="data/03.png",
               ylabel="R",
               xlabel="coupling")
