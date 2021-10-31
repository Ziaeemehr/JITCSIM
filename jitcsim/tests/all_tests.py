import os
import unittest
import numpy as np
from math import pi
from time import time
from jitcsim import plot_order
from jitcsim import make_network
from numpy.random import normal, uniform
from jitcsim.models.kuramoto_ode import Kuramoto_II as KM_II_ODE
from jitcsim.models.kuramoto_dde import Kuramoto_II as KM_II_DDE
from jitcsim.models.kuramoto_sde import Kuramoto_II as KM_II_SDE

# import warnings
# warnings.filterwarnings("ignore")


class TestModules(unittest.TestCase):

    def test_ode_Kuramoto_II(self):

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

        sol = KM_II_ODE(parameters)
        sol.compile()

        data = sol.simulate([])
        x = data['x']
        order = sol.order_parameter(x)
        os.remove("data/km.so")
        self.assertEqual(abs(np.mean(order)-0.909) < 0.1, True)

    def test_ode_Kuramoro_II_single_param_repeated_run(self):

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
            "modulename": "km1"
        }

        sol = KM_II_ODE(parameters)
        sol.compile()

        couplings = np.arange(0, 0.8, 0.05) / (N-1)
        orders = np.zeros((len(couplings), num_ensembles))

        for i in range(len(couplings)):
            for j in range(num_ensembles):

                controls = [couplings[i]]
                sol.set_initial_state(uniform(-pi, pi, N))
                data = sol.simulate(controls)
                x = data['x']
                orders[i, j] = np.mean(sol.order_parameter(x))

        plot_order(couplings,
                   np.mean(orders, axis=1),
                   filename="data/02.png",
                   ylabel="R",
                   xlabel="coupling")

        os.remove("data/km1.so")
        self.assertEqual(abs(np.mean(orders, axis=1)[0]-0.184) < 0.1, True)

    def test_Kuramoto_II_dde_single_param(self):

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
            "modulename": "km_dde"
        }

        sol = KM_II_DDE(parameters)
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
        os.remove("data/km_dde.so")

        self.assertEqual(abs(np.mean(order)-0.54) < 0.1, True)

    def test_Kuramoto_II_SDE_single_param(self):

        seed = 2
        np.random.seed(seed)

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

            # noise amplitude (normal distribution)
            "sigma": sigma0,
            "alpha": alpha0,                    # frustration
            "omega": omega0,                    # initial angular frequencies
            'initial_state': initial_state,     # initial phase of oscillators

            'control': ['coupling'],            # control parameters

            "use_omp": False,                   # use OpenMP
            "output": "data",                   # output directory
            "modulename": "km_sde"
        }

        sol = KM_II_SDE(parameters)
        sol.compile()
        sol.set_seed(seed)

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
        os.remove("data/km_sde.so")
        self.assertEqual(abs(np.mean(order)-0.94) < 0.1, True)


if __name__ == "__main__":
    unittest.main()
