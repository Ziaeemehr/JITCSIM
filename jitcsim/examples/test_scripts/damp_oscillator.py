import os
import numpy as np
import pylab as plt
from os.path import join
from jitcsde import jitcsde, y


def f_sym():
    yield y(0) - y(0) * y(1) - a * y(0) * y(0)
    yield y(0) * y(1) - y(1) - b * y(1) * y(1)


def g_sym():
    yield sigma1
    yield sigma2


if __name__ == "__main__":


    a = 0.1
    b = 0.05
    initial_state = [0.1, 2.0]
    dimension = 2

    t_final = 400
    t_initial = 0
    t_transition = 10
    interval = 0.1

    sigma1 = 0.1
    sigma2 = 0.0

    SDE = jitcsde(f_sym,g_sym)
    SDE.set_initial_value(initial_state,t_initial)

    times = t_transition + np.arange(t_initial, t_final - t_transition, interval)
    num_steps = len(times)
    x = np.empty(num_steps)
    v = np.empty(num_steps)

    for i in range(num_steps):
        x0 = SDE.integrate(times[i])
        x[i] = x0[0]
        v[i] = x0[1]

    fig, ax = plt.subplots(2, sharex=True)

    ax[0].plot(times, x, label="x")
    ax[1].plot(times, v, label="v")
    ax[0].legend()
    ax[1].legend()
    plt.savefig("damp.png", dpi=150)
    plt.show()
