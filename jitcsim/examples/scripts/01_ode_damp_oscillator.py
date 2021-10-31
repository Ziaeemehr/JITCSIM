import pylab as plt
import numpy as np
from os.path import join
from jitcsim.models.damp_oscillator import DampOscillator


def test_simulation():
    """
    plot theta and omega time series given parameters a and b.
    """

    parameters = {
        "a": 0.1,
        "b": 0.05,
        "interval": 0.1,
        "dim": 2,
        "t_initial": 0,
        "t_final": 100.0,
        "t_transition": 20,
        "control": ["a", "b"],
        "output": "data",
        "initial_state": [0.5, 1.0],
        "integration_method": "RK45",
        "modulename": "do_",
        "use_omp": False,
    }
    controls = [0.1, 0.05]

    ode = DampOscillator(parameters)
    ode.compile()
    sol = ode.simulate(controls)
    t = sol["t"]
    x = sol["x"]

    plt.style.use("ggplot")
    plt.plot(t, x[:, 0], label='$\\theta$')
    plt.plot(t, x[:, 1], label='$\omega$')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("x")
    plt.savefig(join("data", "damp.png"), dpi=150)


if __name__ == "__main__":
    test_simulation()
