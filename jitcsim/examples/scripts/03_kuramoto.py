import os
import numpy as np
import pylab as plt
from numpy import pi
from time import time
import networkx as nx
from jitcode import jitcode, y
from symengine import sin, Symbol, symarray
from numpy.random import uniform, normal
from jitcsim.utility import order_parameter
from jitcsim.visualization import plot_order

if not os.path.exists("data"):
    os.makedirs("data")


def kuramotos():

    for i in range(N):
        interaction = sum(sin(y(j)-y(i)) for j in range(N) if adj[i, j])
        yield omega[i] + coupling * interaction


if __name__ == "__main__":

    np.random.seed(1)

    N = 50
    coupling0 = 0.5 / (N-1)
    omega0 = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)
    adj = nx.to_numpy_array(nx.gnp_random_graph(N, 1, seed=1))

    param = {
        'N': N,
        'adj': adj,
        # 'omega': omega,
        # 'coupling': coupling,
        'initial_state': initial_state,
        'control_pars': ['coupling']
    }

    omega = symarray("omega", N)
    coupling = Symbol("coupling")

    start_time = time()
    times = range(0, 201)

    control_pars = [coupling]
    control_pars.extend(omega)
    par = [coupling0]
    par.extend(omega0)

    I = jitcode(kuramotos, n=N, control_pars=control_pars)
    I.generate_f_C()
    I.compile_C(omp=False)
    I.save_compiled(overwrite=True, destination="KM.so")

    I = jitcode(n=N,
                control_pars=control_pars,
                module_location="KM.so")

    I.set_integrator(name="RK23")
    I.set_parameters(par)
    I.set_initial_value(initial_state, time=0.0)
    phases = np.zeros((len(times), N))

    for i in range(len(times)):
        phases[i, :] = I.integrate(times[i]) % (2*np.pi)

    order = order_parameter(phases)
    plot_order(times, order, "f.png")
