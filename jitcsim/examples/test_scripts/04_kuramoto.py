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

os.environ["CC"] = "clang"


def kuramotos():

    for i in range(N):
        interaction = sum(sin(y(j)-y(i)) for j in range(N) if adj[i, j])
        yield omega[i] + coupling * interaction


if __name__ == "__main__":

    np.random.seed(1)

    N = 100
    num_ensembles = 2
    times = range(0, 30000)
    coupling0 = 0.5 / (N-1)
    omega = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)
    couplings = np.arange(0, 0.8, 0.05) / (N-1)
    couplings = np.array([0.1, 0.2]) / (N-1)
    orders = np.zeros((len(couplings), num_ensembles))
    adj = nx.to_numpy_array(nx.gnp_random_graph(N, 1, seed=1))

    compile_time = time()

    coupling = Symbol("coupling")

    I = jitcode(kuramotos, n=N, control_pars=[coupling])
    I.generate_f_C(chunk_size=25)
    I.compile_C(omp=True, modulename="KM")
    I.save_compiled(overwrite=True, destination="data/")

    print("Compile time : {:.3f} seconds.".format(time() - compile_time))

    start_time = time()
    for i in range(len(couplings)):
        for j in range(num_ensembles):
            print(i, j)
            phases = np.zeros((len(times), N))
            I = jitcode(n=N,
                        control_pars=[coupling],
                        module_location="data/KM.so")
            I.set_integrator(name="RK23")
            I.set_parameters(couplings[i])
            I.set_initial_value(initial_state, time=0.0)

            for k in range(len(times)):
                phases[k, :] = I.integrate(times[k]) % (2*np.pi)

            orders[i, j] = np.mean(order_parameter(phases))

    print("Simulation time: {:.3f} seconds".format(time()-start_time))

    plt.plot(couplings, np.mean(orders, axis=1), lw=1)
    plt.savefig("f.png")
