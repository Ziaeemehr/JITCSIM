import os
import os.path
import numpy as np
from numpy import pi
from os.path import join
from jitcode import jitcode, jitcode_lyap, y
from symengine import sin, cos, Symbol, symarray
from jitcsim.utility import (order_parameter as _order,
                             local_order_parameter as _local_order)
os.environ["CC"] = "clang"


class Kuramoto_Base:

    """
    Base class for the Kuramoto model.

    Parameters
    ----------

    N: int
        number of nodes
    adj: 2d array
        adjacency matrix
    t_initial: float, int
        initial time of integration
    t_final: float, int
        final time of integration
    t_transition: float, int
        transition time
    interval : float
        time interval for sampling
    alpha : flaot
        frustration
    omega : float
        initial angular frequencies
    initial_state : array of size N
        initial phase of oscillators
    integration_method : str
            name of the integrator
			One of the following (or a new method supported by either backend):
			
			* `"dopri5"` – Dormand’s and Prince’s explicit fifth-order method via `ode`
			* `"RK45"` – Dormand’s and Prince’s explicit fifth-order method via `solve_ivp`
			* `"dop853"` – DoP853 (explicit) via `ode`
			* `"DOP853"` – DoP853 (explicit) via `solve_ivp`
			* `"RK23"` – Bogacki’s and Shampine’s explicit third-order method via `solve_ivp`
			* `"BDF"` – Implicit backward-differentiation formula via `solve_ivp`
			* `"lsoda"` – LSODA (implicit) via `ode`
			* `"LSODA"` – LSODA (implicit) via `solve_ivp`
			* `"Radau"` – The implicit Radau method via `solve_ivp`
			* `"vode"` – VODE (implicit) via `ode`
			
			The `solve_ivp` methods are usually slightly faster for large differential equations, but they come with a massive overhead that makes them considerably slower for small differential equations. Implicit solvers are slower than explicit ones, except for stiff problems. If you don’t know what to choose, start with `"dopri5"`.

    control : list of str 
        control parameters 
    use_omp : boolian 
        if `True` allow to use OpenMP
    output : str
        output directory
    verbose: boolian
        if  `True` some information about the process will be desplayed.

    """

    def __init__(self, par) -> None:

        for item in par.items():
            name = item[0]
            value = item[1]
            if name not in par['control']:
                setattr(self, name, value)

        self.control_pars = []
        for i in self.control:
            if i == "omega":
                name = i
                value = symarray(name, self.N)
                setattr(self, name, value)
                self.control_pars.extend(value)

            else:
                name = i
                value = Symbol(name)
                setattr(self, name, value)
                self.control_pars.append(value)

        if not os.path.exists(self.output):
            os.makedirs(self.output)

        if not "modulename" in par.keys():
            self.modulename = "km"

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

        I = jitcode(self.rhs, n=self.N,
                    control_pars=self.control_pars)
        I.generate_f_C(**kwargs)
        I.compile_C(omp=self.use_omp, modulename=self.modulename)
        I.save_compiled(overwrite=True, destination=join(self.output, ''))
    # ---------------------------------------------------------------

    def set_initial_state(self, x0):

        assert(len(x0) == self.N)
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

        I = jitcode(n=self.N,
                    control_pars=self.control_pars,
                    module_location=join(self.output, self.modulename+".so"))
        I.set_integrator(name=self.integration_method,
                         **integrator_params)
        I.set_parameters(par)
        I.set_initial_value(self.initial_state, time=self.t_initial)

        times = self.t_transition + \
            np.arange(self.t_initial, self.t_final -
                      self.t_transition, self.interval)
        phases = np.zeros((len(times), self.N))
        for i in range(len(times)):
            phases[i, :] = I.integrate(times[i]) % (2*np.pi)

        return {"t": times, "x": phases}
    # ---------------------------------------------------------------

    def order_parameter(self, phases):
        """
        calculate the Kuramoto order parameter

        Parameters
        -----------

        phases : 2D numpy array [nstep by N]
            phase of oscillators 

        """
        order = _order(phases)
        return order
    # ---------------------------------------------------------------

    def local_order_parameter(self, phases, indices):
        """
        calculate local Kuramoto order parameter for given node indices

        Parameters
        ----------

        phases : float numpy array
            phase of each oscillator.
        indices : int numpy array 
            indices of given nodes.
        """
        order = _local_order(phases, indices)
        return order

    # ---------------------------------------------------------------


class Kuramoto_II(Kuramoto_Base):

    """
    **Kuramoto model type II**

    .. math::
            \\frac{d\\theta_i}{dt} &= \\omega_i + \\sum_{j=0}^{N-1} a_{i,j} \\sin(y_j - y_i - \\alpha) 

    Parameters
    ----------

    N: int
        number of nodes
    adj: 2d array
        adjacency matrix
    t_initial: float, int
        initial time of integration
    t_final: float, int
        final time of integration
    t_transition: float, int
        transition time
    interval : float
        time interval for sampling
    alpha : flaot
        frustration
    omega : float
        initial angular frequencies
    initial_state : array of size N
        initial phase of oscillators
    integration_method : str
            name of the integrator
			One of the following (or a new method supported by either backend):
			
			* `"dopri5"` – Dormand’s and Prince’s explicit fifth-order method via `ode`
			* `"RK45"` – Dormand’s and Prince’s explicit fifth-order method via `solve_ivp`
			* `"dop853"` – DoP853 (explicit) via `ode`
			* `"DOP853"` – DoP853 (explicit) via `solve_ivp`
			* `"RK23"` – Bogacki’s and Shampine’s explicit third-order method via `solve_ivp`
			* `"BDF"` – Implicit backward-differentiation formula via `solve_ivp`
			* `"lsoda"` – LSODA (implicit) via `ode`
			* `"LSODA"` – LSODA (implicit) via `solve_ivp`
			* `"Radau"` – The implicit Radau method via `solve_ivp`
			* `"vode"` – VODE (implicit) via `ode`
			
			The `solve_ivp` methods are usually slightly faster for large differential equations, but they come with a massive overhead that makes them considerably slower for small differential equations. Implicit solvers are slower than explicit ones, except for stiff problems. If you don’t know what to choose, start with `"dopri5"`.
            
    control : list of str 
        control parameters 
    use_omp : boolian 
        if `True` allow to use OpenMP
    output : str
        output directory
    verbose: boolian
        if  `True` some information about the process will be desplayed.

    """

    def __init__(self, par) -> None:
        super().__init__(par)

    # ---------------------------------------------------------------

    def rhs(self):
        '''
        **Kuramoto model of type II**

        .. math::
            \\frac{d\\theta_i}{dt} &= \\omega_i + \\sum_{j=0}^{N-1} a_{i,j} \\sin(y_j - y_i - \\alpha) 

        
        '''

        for i in range(self.N):
            sumj = np.sum(sin(y(j)-y(i) - self.alpha)
                          for j in range(self.N) if self.adj[i, j])

            yield self.omega[i] + self.coupling * sumj
    # ---------------------------------------------------------------


class Kuramoto_I(Kuramoto_Base):
    """
    **Kuramot model type I**

    .. math::
            \\frac{d\\theta_i}{dt} = \\omega_i + 0.5 * \\sum_{j=0}^{N-1} a_{i,j} \\Big(1 - \\cos(y_j - y_i - \\alpha) \\Big)
        

    Parameters
    ----------

    N: int
        number of nodes
    adj: 2d array
        adjacency matrix
    t_initial: float, int
        initial time of integration
    t_final: float, int
        final time of integration
    t_transition: float, int
        transition time
    interval : float
        time interval for sampling
    alpha : flaot
        frustration
    omega : float
        initial angular frequencies
    initial_state : array of size N
        initial phase of oscillators
    integration_method : str
            name of the integrator
			One of the following (or a new method supported by either backend):
			
			* `"dopri5"` – Dormand’s and Prince’s explicit fifth-order method via `ode`
			* `"RK45"` – Dormand’s and Prince’s explicit fifth-order method via `solve_ivp`
			* `"dop853"` – DoP853 (explicit) via `ode`
			* `"DOP853"` – DoP853 (explicit) via `solve_ivp`
			* `"RK23"` – Bogacki’s and Shampine’s explicit third-order method via `solve_ivp`
			* `"BDF"` – Implicit backward-differentiation formula via `solve_ivp`
			* `"lsoda"` – LSODA (implicit) via `ode`
			* `"LSODA"` – LSODA (implicit) via `solve_ivp`
			* `"Radau"` – The implicit Radau method via `solve_ivp`
			* `"vode"` – VODE (implicit) via `ode`
			
			The `solve_ivp` methods are usually slightly faster for large differential equations, but they come with a massive overhead that makes them considerably slower for small differential equations. Implicit solvers are slower than explicit ones, except for stiff problems. If you don’t know what to choose, start with `"dopri5"`.
            
    control : list of str 
        control parameters 
    use_omp : boolian 
        if `True` allow to use OpenMP
    output : str
        output directory
    verbose: boolian
        if  `True` some information about the process will be desplayed.

    """

    def __init__(self, par) -> None:
        super().__init__(par)

    def rhs(self):
        '''
        **Kuramoto model of type I**

        .. math::
            \\frac{d\\theta_i}{dt} = \\omega_i + 0.5 * \\sum_{j=0}^{N-1} a_{i,j} \\Big(1 - \\cos(y_j - y_i - \\alpha) \\Big)
        

        Return :  
            right hand side of the Kuramoto model
        '''

        for i in range(self.N):
            sumj = 0.5 * np.sum(1-cos(y(j)-y(i) - self.alpha)
                                for j in range(self.N) if self.adj[j, i])
            yield self.omega[i] + self.coupling * sumj


class SOKM_SingleLayer(Kuramoto_Base):
    
    """
    **Second order Kuramoto Model for single layer network**

    .. math::
        m \\frac{d^2 \\theta_i(t)}{dt^2}+\\frac{d\\theta_i(t)}{dt} = \\omega_i + \\frac{\\lambda}{\\langle k \\rangle} \\sum_{j=1}^N \\sin \\Big[ \\theta_j(t) - \\theta_i(t) \\Big]


    Reference: 

    Kachhvah, A.D. and Jalan, S., 2017. Multiplexing induced explosive synchronization in Kuramoto oscillators with inertia. EPL (Europhysics Letters), 119(6), p.60005.

    """

    def __init__(self, par) -> None:
        super().__init__(par)

    def rhs(self):

        for i in range(self.N):
            yield y(i+self.N)

        for i in range(self.N):
            sumj = sum(sin(y(j)-y(i))
                       for j in range(self.N)
                       if self.adj[j, i])
            yield (-y(i+self.N) + self.omega[i] +
                   self.coupling * sumj) * self.inv_m

    def compile(self, **kwargs):

        I = jitcode(self.rhs, n=2 * self.N,
                    control_pars=self.control_pars)
        I.generate_f_C(**kwargs)
        I.compile_C(omp=self.use_omp, modulename=self.modulename)
        I.save_compiled(overwrite=True, destination=join(self.output, ''))

    def set_initial_state(self, x0):

        assert(len(x0) == 2 * self.N)
        self.initial_state = x0

    def simulate(self, par, **integrator_params):
        '''
        Integrate the system of equations and return the
        coordinates and times

        Return : dict(t, x)
            - **t** times
            - **x** coordinates.
        '''

        I = jitcode(n=2 * self.N,
                    control_pars=self.control_pars,
                    module_location=join(self.output, self.modulename+".so"))
        I.set_integrator(name=self.integration_method,
                         **integrator_params)
        I.set_parameters(par)
        I.set_initial_value(self.initial_state, time=self.t_initial)

        times = self.t_transition + \
            np.arange(self.t_initial, self.t_final -
                      self.t_transition, self.interval)
        phases = np.zeros((len(times), 2 * self.N))

        for i in range(len(times)):
            phases[i, :] = I.integrate(times[i])
            phases[i, :self.N] = phases[i, :self.N] % (2*np.pi)

        return {"t": times, "x": phases}


class Lyap_Kuramoto_II(Kuramoto_Base):

    def __init__(self, par) -> None:
        super().__init__(par)

        if not "modulename" in par.keys():
            self.modulename = "lyap_km"

        try:
            self.verbose = par['verbose']
        except:
            self.verbose = False

    def rhs(self):
        '''
        **Kuramoto model of type II**

        .. math::
            \\frac{d\\theta_i}{dt} = \\omega_i + \\sum_{j=0}^{N-1} a_{i,j} \\sin(y_j - y_i - \\alpha)  
        

        Return :
            right hand side of the Kuramoto model.
        '''

        for i in range(self.N):
            sumj = np.sum(sin(y(j)-y(i) - self.alpha)
                          for j in range(self.N) if self.adj[j, i])

            yield self.omega[i] + self.coupling * sumj

    def compile(self, **kwargs):

        I = jitcode_lyap(self.rhs, n=self.N, n_lyap=self.n_lyap,
                         control_pars=self.control_pars)
        I.generate_f_C(**kwargs)
        I.compile_C(omp=self.use_omp, modulename=self.modulename,
                    verbose=self.verbose)
        I.save_compiled(overwrite=True, destination=join(self.output, ''))

    def simulate(self, par, **integrator_params):
        '''
        integrate the system of equations and calculate the Lyapunov exponents.

        Parameters
        -----------
        
        par : [list of int, float] 
            values for control parameter(s).

        Return : dict(t, x)
            - **t** times
            - **x** coordinates.
        '''

        I = jitcode_lyap(n=self.N, n_lyap=self.n_lyap,
                         control_pars=self.control_pars,
                         module_location=join(self.output,
                                              self.modulename+".so"))
        I.set_integrator(name=self.integration_method,
                         **integrator_params)
        I.set_parameters(par)
        I.set_initial_value(self.initial_state, time=self.t_initial)

        times = np.arange(self.t_initial, self.t_final, self.interval)
        lyaps = np.zeros((len(times), self.n_lyap))
        for i in range(len(times)):
            lyaps[i, :] = I.integrate(times[i])[1]

        return {"t": times, "lyap": lyaps}
    # ---------------------------------------------------------------
