import os
import numpy as np 
import pylab as plt
from os.path import join
from jitcsde import jitcsde, y
from symengine import Symbol

N = 1
# tau = 1
tau = Symbol("tau")
Delta = 2.0
eta = -5 * Delta
J = 15  * np.sqrt(Delta)
initial_state = [0.01, -2.0]
dimension=2
output="data"
sigma_r = 0.01
sigma_v = 0.01
I_app = 4

t_final = 40
t_initial = 0
t_transition = 1
interval = 0.1

def f_sym():
    yield Delta / (tau * np.pi) + 2 * y(0)*y(1) / tau
    yield 1.0/tau * (y(1)**2 + eta + I_app + J *
            tau * y(0) - (np.pi*tau * y(0))**2)
    
def g_sym():
    yield sigma_r
    yield sigma_v

if not os.path.exists(output):
    os.makedirs(output)

control_pars = [tau]
modulename="mb_model"

I = jitcsde(f_sym, g_sym, n=N*dimension,
            control_pars=control_pars, additive=True)
I.compile_C(omp=False)
I.save_compiled(overwrite=True,
                        destination=join(output, modulename))
I = jitcsde(n=N*dimension, verbose=False,
                    control_pars=control_pars,
                    module_location=join(output, modulename+".so"))
I.set_initial_value(initial_state, time=t_initial)
I.set_parameters([1])

times = t_transition + np.arange(t_initial, t_final -t_transition, interval)
num_steps = len(times)
r_act = np.empty(num_steps)
v_act = np.empty(num_steps)

for i in range(num_steps):
    x0 = I.integrate(times[i])
    r_act[i] = x0[0]
    v_act[i] = x0[1]

import pylab as plt 
fig, ax = plt.subplots(2, sharex=True)

ax[0].plot(times, r_act, label="r")
ax[1].plot(times, v_act, label="v")
ax[0].legend()
ax[1].legend()
plt.savefig("data/01_montbrio.png", dpi=150)

plt.show()
