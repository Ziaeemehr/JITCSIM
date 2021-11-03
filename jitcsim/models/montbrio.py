import os
import os.path
import numpy as np
from numpy import pi
from os.path import join
from symengine import Symbol
from jitcode import jitcode, y, t
import symengine


class Montbrio_Base:

    """
    Montbrio base
    """

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
            self.modulename = "mb"
        if not "verbose" in par.keys():
            self.verbose = False

        self.APPLY_CURRENT_SET = False
    # ---------------------------------------------------------------

    def compile(self, **kwargs):

        """
        compile model and produce shared library.
        translates the derivative to C code using SymEngine’s `C-code printer <https://github.com/symengine/symengine/pull/1054>`_.

        Parameters
        ----------
          kwargs : (key, value)
                used in generate_f_C including
                    - simplify : boolean or None 
                    - do_cse: boolian, 
                        Whether SymPy’s `common-subexpression detection <http://docs.sympy.org/dev/modules/rewriting.html#module-sympy.simplify.cse_main>`_ should be applied before translating to C code. It is almost always better to let the compiler do this (unless you want to set the compiler optimisation to `-O2` or lower): For simple differential equations this should not make any difference to the compiler’s optimisations. For large ones, it may make a difference but also take long. As this requires all entries of `f` at once, it may void advantages gained from using generator functions as an input. Also, this feature uses SymPy and not SymEngine.
                    - chunk_size: int 
                        If the number of instructions in the final C code exceeds this number, it will be split into chunks of this size. See `Handling very large differential equations <http://jitcde-common.readthedocs.io/#handling-very-large-differential-equations>`_ on why this is useful and how to best choose this value.
                        It also used for paralleling using OpenMP to determine task scheduling.
                        If smaller than 1, no chunking will happen.
        """


        I = jitcode(self.rhs, n=self.N * self.dimension,
                    control_pars=self.control_pars)
        I.generate_f_C(**kwargs)
        I.compile_C(omp=self.use_omp, modulename=self.modulename)
        I.save_compiled(overwrite=True, destination=join(self.output, ''))

    # ---------------------------------------------------------------
    
    def set_initial_state(self, x0):

        assert(len(x0) == self.N)
        self.initial_state = x0
    # ---------------------------------------------------------------
    
    def set_current(self, par):

        for item in par.items():
            name = item[0]
            value = item[1]
            setattr(self, name, value)

        self.APPLY_CURRENT_SET = True
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

        I = jitcode(n=self.N * self.dimension,
                    control_pars=self.control_pars,
                    module_location=join(self.output, self.modulename+".so"))
        I.set_integrator(name=self.integration_method,
                         **integrator_params)
        I.set_parameters(par)
        I.set_initial_value(self.initial_state, time=self.t_initial)

        times = self.t_transition + \
            np.arange(self.t_initial, self.t_final -
                      self.t_transition, self.interval)
        x = np.zeros((len(times), self.N * self.dimension))
        for i in range(len(times)):
            x[i, :] = I.integrate(times[i])

        return {"t": times, "x": x}
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


class Montbrio(Montbrio_Base):

    """

    Reference
    -----------

    Montbrio, E., Pazo, D. and Roxin, A., 2015. Macroscopic description for networks of spiking neurons. Physical Review X, 5(2), p.021028.
    """

    N = 1
    # ---------------------------------------------------------------

    def __init__(self, par) -> None:
        super().__init__(par)

        self.Iapp = symengine.Function("Iapp")
    # ---------------------------------------------------------------

    def compile(self, **kwargs):

        I = jitcode(self.rhs, n=self.N * self.dimension,
                    callback_functions=[(self.Iapp, self.Iapp_callback, 1)],
                    control_pars=self.control_pars)
        I.generate_f_C(**kwargs)
        I.compile_C(omp=self.use_omp, modulename=self.modulename)
        I.save_compiled(overwrite=True, destination=join(self.output, ''))
    # ---------------------------------------------------------------

    def simulate(self, par, **integrator_params):

        I = jitcode(n=self.N * self.dimension,
                    callback_functions=[(self.Iapp, self.Iapp_callback, 0)],
                    control_pars=self.control_pars,
                    module_location=join(self.output, self.modulename+".so"))
        I.set_integrator(name=self.integration_method,
                         **integrator_params)
        I.set_parameters(par)
        I.set_initial_value(self.initial_state, time=self.t_initial)

        times = self.t_transition + \
            np.arange(self.t_initial, self.t_final -
                      self.t_transition, self.interval)
        x = np.zeros((len(times), self.N * self.dimension))
        for i in range(len(times)):
            x[i, :] = I.integrate(times[i])

        return {"t": times, "x": x}
    # ---------------------------------------------------------------

    def Iapp_callback(self, y, t0):

        if self.current_type == "step":
            if (t0 < self.current_t_start) or (t0 > self.current_t_end):
                return self.current_amplitude
            else:
                return 0
        else:
            return 0
    # ---------------------------------------------------------------

    def rhs(self):
        """
        single population Montbrio model without frequency addaptation
        """

        yield self.Delta / (self.tau * pi * y(0)) + 2 * y(0)*y(1) / self.tau
        yield 1.0/self.tau * (y(1)**2 + self.eta + self.Iapp(t) + self.J * 
                              self.tau * y(0) - (pi*self.tau * y(0))**2)
        # 3 -> self.Iapp(t)
    # ---------------------------------------------------------------

class Montbrio_Adap(Montbrio_Base):

    N = 1

    def __init__(self, par) -> None:
        super().__init__(par)

    def rhs(self):
        """
        single population Montbrio model without frequency addaptation
        """

        yield self.Delta / (self.tau * pi * y(0)) + 2 * y(0)*y(1) / self.tau
        yield 1.0/self.tau * (y(1)**2 + self.eta + self.Iapp + self.J *
                              self.tau * y(0)*(1.0 - y(2)) - (pi*self.tau * y(0))**2)
        yield 1.0/self.tau_a * y(3)
        yield 1.0/self.tau_a * (self.alpha * y(0) - 2.0*y(3)-y(2))
