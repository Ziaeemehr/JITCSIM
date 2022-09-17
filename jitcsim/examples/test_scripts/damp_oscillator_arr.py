import os
import numpy as np
import pylab as plt
from os.path import join
from jitcsde import jitcsde, y


def f_sym():
    for i in range(N):
        yield y(i*N) - y(i*N) * y(i*N+1) - a * y(i*N) * y(i*N)
        yield y(i*N) * y(i*N+1) - y(i*N+1) - b * y(i*N+1) * y(i*N+1)


def g_sym():
    for _ in range(N):
        yield sigma1
        yield sigma2


if __name__ == "__main__":


    N = 2
    a = 0.1
    b = 0.05
    initial_state = np.random.rand(2*N)
    dimension = 2

    t_final = 40
    t_initial = 0
    t_transition = 10
    interval = 0.1

    sigma1 = 0.01
    sigma2 = 0.0

    SDE = jitcsde(f_sym,g_sym)
    SDE.set_initial_value(initial_state,t_initial)

    times = t_transition + np.arange(t_initial, t_final - t_transition, interval)
    num_steps = len(times)
    x = np.empty((N, num_steps))
    v = np.empty((N, num_steps))

    for i in range(num_steps):
        x0 = SDE.integrate(times[i])
        for j in range(N):
            x[j, i] = x0[j*N]
            v[j, i] = x0[j*N+1]

    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(times, x.T)
    ax[1].plot(times, v.T)
    ax[0].set_ylabel("x")
    ax[1].set_ylabel("v")
    plt.savefig("damp.png", dpi=150)
    plt.show()
