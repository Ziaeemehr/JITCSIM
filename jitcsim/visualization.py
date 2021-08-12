from mpl_toolkits.axes_grid1 import make_axes_locatable
import networkx as nx
import pylab as plt
import numpy as np


def plot_order(x, y,
               filename="f.png",
               color="k",
               xlabel=None,
               ylabel=None,
               label=None,
               ax=None,
               close_fig=True,
               **kwargs):
    """
    plot y vs x with given x and y labels

    Parameters
    ----------

    x : list or array
        values of x axis
    y : list or array
        values of y axis
    filename: name
        filename for figure, need to end with .png, .jpg , ...
    color : str
        line color
    xlabel : None or str
        label of x axis
    ylabel :  None or str
        label of y axis
    label: None or str
        label of the curve
    ax : None or matplotlib axis object
    
    close_fig : boolian
        if `True` the figure object will be closed.

    """

    plt.style.use('ggplot')

    savefig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        savefig = True

    ax.plot(x, y, lw=1, color=color, label=label, **kwargs)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if label is not None:
        ax.legend(frameon=False, loc="upper right")

    ax.margins(x=0.01)
    ax.set_ylim(0, 1.1)

    if savefig:
        fig.savefig(filename, dpi=150)

    if close_fig and savefig:
        plt.close()


def plot_matrix(A,
                ax=None,
                ticks=None,
                vmax=None,
                vmin=None,
                labelsize=14,
                colorbar=True,
                aspect="equal",
                cmap='seismic',
                filename="F.png"
                ):

    """
    Plot given matrix using imshow module.
    
    Parameters
    ----------

    A : 2D numpy array
    ax: None or matplotlib axis object
    ticks : None or list of int or float
    vmax : None, float or int
    vmin : None, float or int
    labelsize: int
    colorbar: boolian,
    aspect : {'equal', 'auto'} or float, default: rcParams["image.aspect"] (default: 'equal')
    cmap cmapstr or Colormap, default: rcParams["image.cmap"] (default: 'seismic')
        The Colormap instance or registered colormap name used to map scalar data to colors. This parameter is ignored for RGB(A) data.
    filename: str, (default: F.png)
        filename of the figure to be stored.
    """

    savefig = False

    if ax is None:
        _, ax = plt.subplots(1)
        savefig = True

    im = ax.imshow(A, origin="lower", cmap=cmap,
                   vmax=vmax, vmin=vmin, aspect=aspect)

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax, ax=ax, ticks=ticks)
        cbar.ax.tick_params(labelsize=labelsize)

    ax.set_xticks([])
    ax.set_yticks([])

    if savefig:
        plt.savefig(filename)
        plt.close()


def plot_degree_omega_distribution(adj, omega,
                                   filename="omega_k.png",
                                   close_fig=True):

    adj = np.asarray(adj)
    assert (len(adj.shape) == 2)

    G = nx.from_numpy_array(adj)

    N = adj.shape[0]
    omegaNeighborsBar = [0] * N

    omegaBar = omega - np.mean(omega)

    degrees = list(dict(G.degree()).values())

    for i in range(N):
        neighbors = [n for n in G[i]]
        omegaNeighborsBar[i] = np.mean(omegaBar[neighbors])

    fig, ax = plt.subplots(2, figsize=(8, 5))

    ax[0].plot(omegaBar, degrees, "ro")
    ax[1].plot(omegaBar, omegaNeighborsBar, "ko")

    ax[0].set_ylabel(r"$k_i$", fontsize=16)
    ax[1].set_ylabel(r"$\langle \omega_j \rangle$", fontsize=16)
    ax[1].set_xlabel(r"$\omega_i$", fontsize=16)

    for i in range(2):
        ax[i].tick_params(labelsize=15)

    plt.tight_layout()
    plt.savefig(filename)

    if close_fig:
        plt.close()


def plot_phases(phases, extent,
                ax=None,
                cmap="afmhot",
                xlabel=None,
                ylabel=None,
                filename="fig.png",
                close_fig=True):

    savefig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        savefig = True

    im = ax.imshow(phases.T,
                   origin="lower",
                   extent=extent,
                   aspect="auto",
                   cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if savefig and close_fig:
        plt.savefig(filename, dpi=150)
        plt.close()
# -------------------------------------------------------------------


def plot_lyaps(x, y,
               filename="f.png",
               color="k",
               xlabel=None,
               ylabel=None,
               label=None,
               xlog=False,
               ylog=False,
               xlim=None,
               ylim=None,
               ax=None,
               close_fig=True,
               **kwargs):
    """
    plot y vs x with given x and y labels
    """

    plt.style.use('ggplot')

    savefig = False
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 4))
        savefig = True

    ax.plot(x, y, lw=1, color=color, label=label, **kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if label is not None:
        ax.legend(frameon=False, loc="upper right")

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")

    ax.margins(x=0.01, y=0.01)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if savefig:
        fig.savefig(filename, dpi=150)

    if close_fig and savefig:
        plt.close()
