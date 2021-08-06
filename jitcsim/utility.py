import numpy as np
from numba import jit


@jit(nopython=True)
def order_parameter(phases):
    """ 
    calculate the Kuramoto order parameter.

    :param: phases [numpy 2d array] shape of the 2d array is [num_time_step by num_nodes];
    :return: r [float] measured order parameter. 
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

    :param: phases [numpy 2d array] shape of the 2d array is [num_time_step by num_nodes];
    :param: indices [array(list) of int] indices of nodes to measure their order parameters;
    :return: r [float] measured order parameter. 
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

    :param: t list of list
    :return: * [float] flattend list
    """
    return [item for sublist in t for item in sublist]


@jit(nopython=True)
def kuramoto_correlation(x):
    """
    Calculate the Kuramoto correlation between phase of nodes

    :param: x [numpy array, float] input phase of oscillators
    :return: cor The correlation matrix.
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


def display_time(time):
    ''' 
    print elaped time

    :param: time difference in seconds.

    '''
    hour = time//3600
    minute = int(time % 3600) // 60
    second = time - (3600.0 * hour + 60.0 * minute)
    print("Done in %d hours %d minutes %.4f seconds"
          % (hour, minute, second))
