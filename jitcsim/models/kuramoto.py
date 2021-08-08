import os
import os.path
import numpy as np
from numpy import pi
from os.path import join
from jitcode import jitcode, y
from symengine import sin, cos, Symbol, symarray
from jitcsim.utility import (order_parameter as _order,
                             local_order_parameter as _local_order)
os.environ["CC"] = "clang"


class Kuramoto_Base:

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
        '''!
        integrate the system of equations and return the
        coordinates and times

        @param par 

        @return dict(t, x)
            - **t** times
            - **x** coordinates.
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
        order = _order(phases)
        return order
    # ---------------------------------------------------------------

    def local_order_parameter(self, phases, indices):
        order = _local_order(phases, indices)
        return order

    # ---------------------------------------------------------------


class Kuramoto_II(Kuramoto_Base):

    def __init__(self, par) -> None:
        super().__init__(par)

    # ---------------------------------------------------------------

    def rhs(self):
        '''!
        Kuramoto model of type II

        \f$
        \frac{d\theta_i}{dt} = \omega_i + \sum_{j=0}^{N-1} a_{i,j} \sin(y_j - y_i - alpha)  \hspace{3.5cm} \text{for Type II}\\
        \f$

        @return right hand side of the Kuramoto model
        '''

        for i in range(self.N):
            sumj = np.sum(sin(y(j)-y(i) - self.alpha)
                          for j in range(self.N) if self.adj[i, j])

            yield self.omega[i] + self.coupling * sumj
    # ---------------------------------------------------------------


class Kuramoto_I(Kuramoto_Base):
    def __init__(self, par) -> None:
        super().__init__(par)

    def rhs(self):
        '''!
        Kuramoto model of type I

        \f$
        \frac{d\theta_i}{dt} = \omega_i + 0.5 * \sum_{j=0}^{N-1} a_{i,j} \Big(1 - \cos(y_j - y_i - alpha) \Big)  \hspace{1cm} \text{for Type I}\\
        \f$

        @return right hand side of the Kuramoto model
        '''

        for i in range(self.N):
            sumj = 0.5 * np.sum(1-cos(y(j)-y(i) - self.alpha)
                                for j in range(self.N) if self.adj[i, j])
            yield self.omega[i] + self.coupling * sumj


class SOKM_SingleLayer(Kuramoto_Base):
    """!
    Second order Kuramoto Model for single layer network
    
    \f$
    m \frac{d^2 \theta_i(t)}{dt^2}+\frac{d\theta_i(t)}{dt} = \omega_i + \frac{\lambda}{\langle k \rangle} \sum_{j=1}^N \sin \Big[ \theta_j(t) - \theta_i(t) \Big]
    \f$


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
                       if self.adj[i, j])
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
        '''!
        integrate the system of equations and return the
        coordinates and times

        @return dict(t, x)
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


class Lyap_Kuramoto_II(Kuramoto_II):

    def __init__(self, par) -> None:
        super().__init__(par)
