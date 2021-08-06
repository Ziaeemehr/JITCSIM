import os
import numpy as np
import pylab as plt
from numpy import pi
from time import time
import networkx as nx
from jitcode import jitcode, y
from symengine import sin, cos, Symbol
from numpy.random import uniform, normal
from jitcsim.utility import order_parameter
from jitcsim.visualization import plot_order

if not os.path.exists("data"):
    os.makedirs("data")


def kuramotos():
    '''!
    Kuramoto model of type I and II

    \f$    
    \frac{d\theta}{dt} = \sum_{j=0}^{N-1} a_{i,j} \sin(y_j - y_i)  \, \text{Type II}\\
    \frac{d\theta}{dt} = 0.5 * \sum_{j=0}^{N-1} a_{i,j} \Big(1 - \cos(y_j - y_i) \Big)  \, \text{Type I}\\
    $\f

    @return right hand side of the Kuramoto model
    '''
    for i in range(N):
        interaction = 0.0
        if kind == 2:
            interaction = sum(sin(y(j)-y(i)) for j in range(N) if adj[i, j])
        else:
            interaction = 0.5 * sum(1-cos(y(j)-y(i))
                                    for j in range(N) if adj[i, j])
        yield omega[i] + coupling * interaction #+ np.random.rand()*0.01


if __name__ == "__main__":

    np.random.seed(1)

    N = 30
    kind = 2
    coupling = 0.5 / (N-1)
    omega = normal(0, 0.1, N)
    initial_state = uniform(-pi, pi, N)
    adj = nx.to_numpy_array(nx.gnp_random_graph(N, 0.5, seed=1))

    param = {
        'N': N,
        'adj': adj,
        'kind': kind,
        'omega': omega,
        'coupling': coupling,
        'initial_state': initial_state,
    }

    start_time = time()
    times = range(0, 201)

    I = jitcode(kuramotos, n=N)
    I.set_integrator("RK23", atol=1e-6, rtol=1e-5)
    I.set_initial_value(initial_state, time=0.0)
    phases = np.empty((len(times), N))

    for i in range(len(times)):
        phases[i, :] = I.integrate(times[i]) % (2*np.pi)

    order = order_parameter(phases)
    plot_order(times, order, "f.png")
