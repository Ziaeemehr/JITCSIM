'''

.. math::
    \\frac{dx}{dt} &= x-xy-ax^2 \\

    \\frac{dy}{dt} &= xy - y -by^2 


To add a new damp oscillator model we need to define the model in `jitcsim/models/damp_oscillator.py`
The file can be used as a template for any other ODE model. For other models, `rhs()` need to be redefined.

Starting with a few imports

.. literalinclude:: ../../jitcsim/models/damp_oscillator.py    
        :start-after: example-st\u0061rt
        :lines: 1-6
        :caption:

set compiler as `clang` which accelerate compiling the code for large system of equations, but this is optional. By default it uses `gcc`.

.. literalinclude:: ../../jitcsim/models/damp_oscillator.py    
        :start-after: example-st\u0061rt
        :lines: 8


choose a name for class start with Capital letter:

.. literalinclude:: ../../jitcsim/models/damp_oscillator.py    
        :start-after: example-st\u0061rt
        :lines: 10

define `__init__()` which you don't need to modify it usually.

define `rhs()` which is your system of equations:

.. literalinclude:: ../../jitcsim/models/damp_oscillator.py    
        :start-after: example-st\u0061rt
        :lines: 39-46
        :dedent: 4

define `compile()`, `initial_state()` and `simulate()` functions. Again you don't need to modify these functions usually. Sometimes you need to modify `simulate` function if you need to process time series befor passing them as the result.


'''




# example-start
import os
import os.path
import numpy as np
from os.path import join
from jitcode import jitcode, y
from symengine import Symbol

os.environ["CC"] = "clang"

class DampOscillator:

    # ---------------------------------------------------------------

    def __init__(self, par) -> None:

        for item in par.items():
            name = item[0]
            value = item[1]
            if name not in par['control']:
                setattr(self, name, value)

        self.control_pars = []
        for i in self.control:
            name = i
            value = Symbol(name)
            setattr(self, name, value)
            self.control_pars.append(value)

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if not "modulename" in par.keys():
            self.modulename = "dam"

        if not "N" in par.keys():
            self.N = 1
    # ---------------------------------------------------------------

    def rhs(self):
        '''
        The right hand side of the system of equations : dy/dt = f(y, t)
        for a damp oscillator.
        '''

        yield y(0) - y(0) * y(1) - self.a * y(0) * y(0)
        yield y(0) * y(1) - y(1) - self.b * y(1) * y(1)

    # ---------------------------------------------------------------

    def compile(self, **kwargs):
        """
        compile model and produce shared library.
        """

        I = jitcode(self.rhs, n=self.N*self.dim,
                    control_pars=self.control_pars)
        I.generate_f_C(**kwargs)
        I.compile_C(omp=self.use_omp, modulename=self.modulename)
        I.save_compiled(overwrite=True, destination=join(self.output, ''))
    # ---------------------------------------------------------------

    def set_initial_state(self, x0):

        assert(len(x0) == self.N*self.dim)
        self.initial_state = x0
    # ---------------------------------------------------------------

    def simulate(self, par, **integrator_params):
        '''
        integrate the system of equations and return the
        coordinates and times

        Parameters
        ----------

        par : list
            list of values for control parameters in order of appearence in control

        Return : dict(t, x)
                - t times
                - x coordinates.
        '''

        I = jitcode(n=self.N*self.dim,
                    control_pars=self.control_pars,
                    module_location=join(self.output, self.modulename+".so"))
        I.set_integrator(name=self.integration_method,
                         **integrator_params)
        I.set_parameters(par)
        I.set_initial_value(self.initial_state, time=self.t_initial)

        times = self.t_transition + \
            np.arange(self.t_initial, self.t_final -
                      self.t_transition, self.interval)
        x = np.zeros((len(times), self.N*self.dim))
        for i in range(len(times)):
            x[i, :] = I.integrate(times[i])

        return {"t": times, "x": x}
    # ---------------------------------------------------------------
