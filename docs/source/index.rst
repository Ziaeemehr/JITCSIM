

***********************************
Welcome to JiTCSim's documentation!
***********************************

Introduction
#############

JiTCSim is a python package for simulation of complex network dynamics, mainly based on  `JITC*DE <https://jitcode.readthedocs.io/en/latest/>`_  packages for integration of ordinary/stochastic and delay differential equations.



What are JiTC*DE:
*****************
`JiTCODE <https://jitcode.readthedocs.io/en/latest/>`_ (just-in-time compilation for ordinary differential equations) is an extension of SciPy `ODE <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ (**scipy.integrate.ode**) or 
`Solve IVP <http://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html>`_ (**scipy.integrate.solve_ivp**).
Where the latter take a Python function as an argument, JiTCODE takes an iterable (or generator function or dictionary) of symbolic expressions, which it translates to **C** code, compiles on the fly, and uses as the function to feed into SciPy ODE or Solve IVP.
Symbolic expressions are mostly handled by `SymEngine <https://github.com/symengine/symengine>`_, `SymPy <http://www.sympy.org/>`_'s compiled-backend-to-be (see `SymPy vs. SymEngine <https://jitcde-common.readthedocs.io/#sympy-vs-symengine>`_ for details).

`JiTCDDE <https://jitcdde.readthedocs.io/en/stable/>`_ (just-in-time compilation for delay differential equations) is a standalone Python implementation of the DDE integration method proposed by Shampine and Thompson [Shampine2001]_, which in turn employs the Bogacki–Shampine Runge–Kutta pair [Bogacki1989]_.


`JiTCSDE <https://jitcsde.readthedocs.io/en/latest/>`_ (just-in-time compilation for stochastic differential equations) is a standalone Python implementation of the adaptive integration method proposed by Rackauckas and Nie [Rackauckas2017]_, which in turn employs Rößler-type stochastic Runge–Kutta methods [Robler2010]_. It can handle both Itō and Stratonovich SDEs, converting the latter internally. JiTCSDE is designed in analogy to JiTCODE.



.. [Shampine2001] Shampine, L.F. and Thompson, S., 2001. Solving ddes in matlab. Applied Numerical Mathematics, 37(4), pp.441-458., `10.1016/S0168-9274(00)00055-6 <http://dx.doi.org/10.1016/S0168-9274(00)00055-6>`_.

.. [Bogacki1989] Bogacki, P. and Shampine, L.F., 1989. A 3 (2) pair of Runge-Kutta formulas. Applied Mathematics Letters, 2(4), pp.321-325. `10.1016/0893-9659(89)90079-7 <http://dx.doi.org/10.1016/0893-9659(89)90079-7>`_.

.. [Rackauckas2017] Rackauckas, C. and Nie, Q., 2017. Adaptive methods for stochastic differential equations via natural embeddings and rejection sampling with memory. Discrete and continuous dynamical systems. Series B, 22(7), p.2731, `10.3934/dcdsb.2017133 <http://dx.doi.org/10.3934/dcdsb.2017133>`_.

.. [Robler2010] Rößler, A., 2010. Runge–Kutta methods for the strong approximation of solutions of stochastic differential equations. SIAM Journal on Numerical Analysis, 48(3), pp.922-952. `10.1137/09076636X <http://dx.doi.org/10.1137/09076636X>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. Quick Start
.. *****************

.. _tutorial:

Tutorial
***************
Considering the Kuramoto model. The following examples simulate a network of coupled Kuramoto oscillators with frustration, delay and noise.

ODE
--------------------------

.. automodule:: jitcsim.examples.scripts.00_ode_kuramoto_II

.. _repeated:

Control parameters
-------------------------------------

.. automodule:: jitcsim.examples.scripts.02_ode_kuramoto_II_single_param_repeated_run

SDE
--------------------------

.. automodule:: jitcsim.examples.scripts.01_sde_kuramoto_II_single_param

DDE
--------------------------

.. automodule:: jitcsim.examples.scripts.01_dde_kuramoto_II_single_param


Parallel using Multiprocessing
--------------------------------

.. automodule:: jitcsim.examples.scripts.03_ode_kuramoto_II_single_param_parallel 


Explosive synchronization and hysteresis loop
-----------------------------------------------

.. automodule:: jitcsim.examples.scripts.07_ode_explosive_synchronization

There is also another exaple for calculation of explosive synchronization and hysteresis loop for the second order Kuramoto model [Kachhvah]_. Look at `papers/Kachhavah2017` for more details.


.. [Kachhvah] Kachhvah, A.D. and Jalan, S., 2017. Multiplexing induced explosive synchronization in Kuramoto oscillators with inertia. EPL (Europhysics Letters), 119(6), p.60005.

Lyapunov exponents
-----------------------

.. automodule:: jitcsim.examples.scripts.08_ode_lyapunov_exponents


Adding new models
##################

One the main purpose of JiTCSim is flexiblity and ease of development for users with medium knowledge about Python. No exprience on C/C++ is required.
Process of adding new model is strightforward as we see in the following.
The system of equations need to be provided in Python syntax, considering requirements of JiTC*DE, and 
also using available models (using ODE/DDE/SDE sovlers) as template.

Damp oscillator
*****************

.. automodule:: jitcsim.models.damp_oscillator






Auto Generated Documentation 
##############################

Kuramoto Model
*****************
The following are the main classes for the Kuramoto model including:

| :ref:`ODE Kuramoto model class <KM_ODE>` 
| :ref:`SDE Kuramoto model class <KM_SDE>`
| :ref:`DDE Kuramoto model class <KM_DDE>`


.. _KM_ODE:

The main class for the ODE Kuramot model
-----------------------------------------

.. autoclass:: jitcsim.models.kuramoto_ode.Kuramoto_Base
    :members:

-------------------------------------------------------    

.. autoclass:: jitcsim.models.kuramoto_ode.Kuramoto_II
  :members:
  :inherited-members:

-------------------------------------------------------    

.. autoclass:: jitcsim.models.kuramoto_ode.Kuramoto_I
  :members:
  :inherited-members:

-------------------------------------------------------    

.. autoclass:: jitcsim.models.kuramoto_ode.SOKM_SingleLayer
  :members:
  :inherited-members:

-------------------------------------------------------    

.. _KM_SDE:

The main class for the SDE Kuramot model
----------------------------------------
     
.. autoclass:: jitcsim.models.kuramoto_sde.Kuramoto_Base
   :members:
   :inherited-members:

-------------------------------------------------------    

.. autoclass:: jitcsim.models.kuramoto_sde.Kuramoto_II
   :members:
   :inherited-members:

-------------------------------------------------------         

.. _KM_DDE:

The main class for the DDE Kuramot model
-----------------------------------------
        
.. autoclass:: jitcsim.models.kuramoto_dde.Kuramoto_Base
    :members:
    :inherited-members:

-------------------------------------------------------     

.. autoclass:: jitcsim.models.kuramoto_dde.Kuramoto_II
    :members:
    :inherited-members:


Visualization
-------------------

.. automodule:: jitcsim.visualization
    :members:
  
    
Networks 
------------

.. automodule:: jitcsim.networks
  :members:


Utility
--------------

.. automodule:: jitcsim.utility 
  :members:



