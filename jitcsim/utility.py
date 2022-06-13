import numpy as np
from numba import jit
import time

def timer(func):
    '''
    decorator to measure elapsed time
    Parameters
    -----------
    func: function
        function to be decorated
    '''

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        display_time(end-start, message="{:s}".format(func.__name__))
        return result
    return wrapper

@jit(nopython=True)
def order_parameter(phases):
    """ 
    calculate the Kuramoto order parameter.

    Parameters
    ----------
    phases : numpy 2D array (num_time_step by num_nodes)
        phase of oscillators

    Return
    ------

    r : float 
        Kuramotoorder parameter.
    """

    n_steps, n_nodes = phases.shape
    r = np.zeros(n_steps)

    for i in range(n_steps):
        r[i] = np.abs(np.sum(np.exp(1j * phases[i, :]))) / n_nodes

    return r


@jit(nopython=True)
def local_order_parameter(phases, indices):
    """
    calculate the local order parameter of given indices

    Parameters
    ----------

    phases : numpy 2D array (num_time_step by num_nodes)
        phase of each node
    indices : array(list) of int 
        indices of nodes to measure their order parameters;
    Return
    -------

    r : float 
        Kuramoto order parameter. 
    """

    n_nodes = len(indices)
    n_steps = phases.shape[0]

    assert(n_nodes > 1), "length of indices need to be larger that 1."
    assert(n_nodes <= phases.shape[1]
           ), "number of indices exceeded the number of nodes"

    r = np.zeros(n_steps)
    for i in range(n_steps):
        r[i] = abs(sum(np.exp(1j * phases[i, indices]))) / n_nodes

    return r


def flatten(t):
    """
    flatten a list of list

    Parameters
    ----------
    t : list of list

    Return: 
        flattend list
    """
    return [item for sublist in t for item in sublist]


@jit(nopython=True)
def kuramoto_correlation(x):
    """
    Calculate the Kuramoto correlation between phase of nodes

    Parameters
    ----------

    x : numpy array, float 
        input phase of oscillators
    Return
    -------
    cor : 2D numpy array
        The correlation matrix.
    """

    n = len(x)
    cor = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            cor[j, i] = cor[i, j] = np.cos(x[j] - x[i])

    cor = cor + np.diag(np.ones(n))

    return cor


def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def display_time(time, message=""):
    '''
    display elapsed time in hours, minutes, seconds
    Parameters
    -----------
    time: float
        elaspsed time in seconds
    '''

    hour = int(time/3600)
    minute = (int(time % 3600))//60
    second = time-(3600.*hour+60.*minute)
    print("{:s} Done in {:d} hours {:d} minutes {:09.6f} seconds".format(
        message, hour, minute, second))


def get_step_current(t_start, t_end, amplitude):

    return {
        "current_type": "step",
        "current_t_end": t_end,
        "current_t_start": t_start,
        "current_amplitude": amplitude
    }
