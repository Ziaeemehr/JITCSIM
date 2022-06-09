import os
import os.path
import numpy as np
from numpy import pi
from os.path import join
from symengine import Symbol
from jitcsde import jitcsde, y, t
from jitcxde_common.symbolic import conditional
from jitcsim.utility import binarize


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
            self.modulename = "mb_model"
        if not "verbose" in par.keys():
            self.verbose = False

        self.integtaror_params_set = False
        self.SET_SEED = False
        self.seed = None

    # ---------------------------------------------------------------

    def set_seed(self, seed):
        self.SET_SEED = True
        self.seed = seed
    # ---------------------------------------------------------------

    def compile(self, **kwargs):

        I = jitcsde(self.f_sym, self.g_sym, n=self.N*self.dimension,
                    control_pars=self.control_pars, additive=True)
        I.compile_C(omp=self.use_omp, **kwargs)
        I.save_compiled(overwrite=True,
                        destination=join(self.output, self.modulename))
        # ---------------------------------------------------------------

    def set_integrator_parameters(self,
                                  atol=1e-5,
                                  rtol=1e-2,
                                  min_step=1e-5,
                                  max_step=10.0):
        """!
        set properties for integrator        
        """

        self.integrator_params = {"atol": atol,
                                  "rtol": rtol,
                                  "min_step": min_step,
                                  "max_step": max_step
                                  }
        self.integtaror_params_set = True
    # ---------------------------------------------------------------

    def set_initial_state(self, x0):

        assert(len(x0) == self.N * self.dimension)
        self.initial_state = x0
    # ---------------------------------------------------------------

    def simulate(self, par=[]):

        I = jitcsde(n=self.N*self.dimension, verbose=False,
                    control_pars=self.control_pars,
                    module_location=join(self.output, self.modulename+".so"))
        if self.SET_SEED:
            I.set_seed(self.seed)

        I.set_initial_value(self.initial_state, time=self.t_initial)

        if (len(self.control_pars) > 0):
            I.set_parameters(par)

        if not self.integtaror_params_set:
            self.set_integrator_parameters()
        I.set_integration_parameters(**self.integrator_params)

        times = self.t_transition + \
            np.arange(self.t_initial, self.t_final -
                      self.t_transition, self.interval)
        x = np.zeros((len(times), self.N*self.dimension))
        for i in range(len(times)):
            x[i, :] = I.integrate(times[i])

        return {"t": times, "x": x}

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


class MontbrioSingleNode(Montbrio_Base):
    def __init__(self, par) -> None:
        super().__init__(par)

    def f_sym(self):
        """
        single population Montbrio model without frequency addaptation
        """
        # if y(0) < 0:
        #     y(0) = 0.0

        yield 1.0/self.tau*(self.Delta / (self.tau * pi) + 2 * y(0)*y(1))
        yield 1.0/self.tau * (y(1)**2 + self.eta + self.I_app + self.J *
                              self.tau*y(0)-(pi*self.tau*y(0))**2)

    def g_sym(self):
        yield self.sigma_r
        yield self.sigma_v
# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------


class MontbrioNetwork(Montbrio_Base):
    def __init__(self, par) -> None:
        super().__init__(par)

        assert(hasattr(self, "adj"))
        self.adj_bin = binarize(self.adj)

        """
        Montbrio model on network

        Reference:
        -------------
            - to do
        """

    def f_sym(self):
        """
        single population Montbrio model without frequency addaptation
        """
        for i in range(self.N):
            sumj = sum(self.adj[j, i] * y(2*j)
                       for j in range(self.N) if self.adj_bin[j, i])
            yield 1.0/self.tau*(self.Delta / (self.tau * pi) + 2 * y(2*i)*y(2*i+1))
            yield 1.0/self.tau*(y(2*i+1)**2 + self.eta + self.I_app +
                                self.J*self.tau*y(2*i)-(pi*self.tau*y(2*i))**2+self.coupling*sumj)

    def g_sym(self):
        for _ in range(self.N):
            yield self.sigma_r
            yield self.sigma_v

    def simulate(self, par=[]):

        I = jitcsde(n=self.N*self.dimension, verbose=False,
                    control_pars=self.control_pars,
                    module_location=join(self.output, self.modulename+".so"))

        if self.SET_SEED:
            I.set_seed(self.seed)

        I.set_initial_value(self.initial_state, time=self.t_initial)

        if (len(self.control_pars) > 0):
            I.set_parameters(par)

        if not self.integtaror_params_set:
            self.set_integrator_parameters()
        I.set_integration_parameters(**self.integrator_params)

        times = self.t_transition + \
            np.arange(self.t_initial, self.t_final -
                      self.t_transition, self.interval)
        num_steps = len(times)
        r = np.zeros((num_steps, self.N))
        v = np.zeros_like(r)
        for i in range(num_steps):
            x = I.integrate(times[i])
            r[i, :] = x[::2]
            v[i, :] = x[1::2]

        return {"t": times,
                "r": r,
                "v": v}

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------

# class Montbrio_BW(Montbrio_Base):

#     '''
#     Montbrio with Balloon–Windkessel model
#     '''
#     # ---------------------------------------------------------------

#     def __init__(self, par) -> None:
#         super().__init__(par)

#         assert(hasattr(self, "adj"))
#         self.adj_bin = binarize(self.adj)
#     # ---------------------------------------------------------------

#     def simulate(self, par=[], return_rv=True, **integrator_params):

#         I = jitcsde(n=self.dimension*self.N,
#                     verbose=False,
#                     control_pars=self.control_pars,
#                     module_location=join(self.output,
#                                          self.modulename+".so"))
#         if self.SET_SEED:
#             I.set_seed(self.seed)

#         I.set_initial_value(self.initial_state, time=self.t_initial)

#         if len(self.control_pars) > 0:
#             I.set_parameters(par)

#         if not self.integtaror_params_set:
#             self.set_integrator_parameters()
#         I.set_integration_parameters(**self.integrator_params)

#         times = self.t_transition + \
#             np.arange(self.t_initial, self.t_final -
#                       self.t_transition, self.interval)

#         num_steps = len(times)
#         if return_rv:
#             r_act = np.empty((num_steps, self.N))
#             v_act = np.empty((num_steps, self.N))
#         fmri = np.empty((num_steps, self.N))

#         V0 = self.V0
#         k1 = self.k1
#         k2 = self.k2
#         k3 = self.k3

#         for i in range(num_steps):

#             x0 = I.integrate(times[i])
#             if return_rv:
#                 r_act[i, :] = x0[::6]
#                 v_act[i, :] = x0[1::6]
#             fmri[i, :] = V0*(k1 - k1*x0[5::6] + k2-k2 *
#                              (x0[5::6]/x0[4::6]) + k3-k3*x0[4::6])
#         if return_rv:
#             return {
#                 "t":times,
#                 "fmri":fmri,
#                 "r":r_act,
#                 "v":v_act
#             }
#         else:
#             return {
#                 "t":t,
#                 "fmri":fmri
#             }
#     # ---------------------------------------------------------------

#     def f_sym(self):
#         """
#         Montbrio model on network
#         """
#         for i in range(self.N):
#             sumj = sum(self.adj[j,i] * y(6*j) for j in range(self.N) if self.adj_bin[j,i])
#             yield self.Delta / (self.tau * pi) + 2 * y(6*i)*y(6*i+1) / self.tau
#             yield 1.0/self.tau * (y(6*i+1)**2 + self.eta + self.I_app + self.J *
#                                 self.tau * y(6*i) - (pi*self.tau * y(6*i))**2) + self.coupling * sumj

#             # from TVB
#             # yield y(6*i)-1/self.tau_s*y(6*i+2)-1/self.tau_f*(y(6*i+3)-1)
#             # yield y(6*i+2)
#             # yield (1/self.tau_o)*(y(6*i+3)-y(6*i+4)**(1/self.alpha))
#             # yield (1/self.tau_o)*((y(6*i+3)*(1-(1-self.E0)**(1/y(6*i+3)))/self.E0) -
#             #                       (y(6*i+4)**(1/self.alpha))*(y(6*i+5)/y(6*i+5)))

#             yield self.epsilon*y(6*i+1)-self.tau_s*y(6*i+2)-self.tau_f*(y(6*i+3)-1)
#             yield y(6*i+2)
#             yield self.tau_o*(y(6*i+3)-y(6*i+4)**self.alpha)
#             yield self.tau_o*(y(6*i+3)*(1-(1-self.E0)**(1/y(6*i+3)))/self.E0 -
#                              (y(6*i+4)**self.alpha)*y(6*i+5)/y(6*i+4))

#     def g_sym(self):
#         for _ in range(self.N):
#             yield self.sigma_r
#             yield self.sigma_v
#             yield 0
#             yield 0
#             yield 0
#             yield 0
    # ---------------------------------------------------------------

# class Montbrio_BWL(Montbrio_BW):

#     def __init__(self, par) -> None:
#         super().__init__(par)

#     def f_sym(self):
#         """
#         Montbrio model on network
#         """
#         for i in range(self.N):
#             sumj = sum(self.adj[j,i] * y(4*j) for j in range(self.N) if self.adj_bin[j,i])
#             yield self.Delta / (self.tau * pi) + 2 * y(4*i)*y(4*i+1) / self.tau
#             yield 1.0/self.tau * (y(4*i+1)**2 + self.eta + self.I_app + self.J *
#                                 self.tau * y(4*i) - (pi*self.tau * y(4*i))**2) + self.coupling * sumj

#             yield self.epsilon*y(4*i+1)-self.taus*y(4*i+2)-self.tauf*(y(4*i+3)-1)
#             yield y(4*i+2)
#             yield self.tauo*(y(4*i+3)-y(4*i+4)**self.alpha)
#             yield self.tauo*(y(4*i+3)*(1-(1-self.E0)**(1/y(4*i+3)))/self.E0 -
#                              (y(4*i+4)**self.alpha)*y(4*i+5)/y(4*i+4))

#     def g_sym(self):
#         for _ in range(self.N):
#             yield self.sigma
#             yield self.sigma
#             yield 0
#             yield 0

#     def simulate(self, par, return_rv=False, **integrator_params):

#         I = jitcsde(n=self.dimension * self.N,
#                     verbose=False,
#                     control_pars=self.control_pars,
#                     module_location=join(self.output_path,
#                                          self.modulename+".so"))
#         I.set_initial_value(self.initial_state, time=self.t_initial)
#         I.set_parameters(par)

#         if not self.integtaror_params_set:
#             self.set_integrator_parameters()
#         I.set_integration_parameters(**self.integrator_params)

#         times = self.t_transition + \
#             np.arange(self.t_initial, self.t_final -
#                       self.t_transition, self.interval)
#         num_steps = len(times)
#         r_act = np.empty((num_steps, self.N))
#         v_act = np.empty((num_steps, self.N))
#         fmri = np.empty((num_steps, self.N))

#         # V0 = self.V0
#         # k1 = self.k1
#         # k2 = self.k2
#         # k3 = self.k3

#         for i in range(num_steps):

#             x0 = I.integrate(times[i])
#             if return_rv:
#                 r_act[i, :] = x0[::6]
#                 v_act[i, :] = x0[1::6]
#         #     fmri[i, :] = V0*(k1 - k1*x0[5::6] + k2-k2 *
#         #                      (x0[5::6]/x0[4::6]) + k3-k3*x0[4::6])
#         if return_rv:
#             return {
#                 "t":times,
#                 "fmri":fmri,
#                 "r":r_act,
#                 "v":v_act
#             }
#         else:
#             return {
#                 "t":t,
#                 "fmri":fmri
#             }
#     # ---------------------------------------------------------------
