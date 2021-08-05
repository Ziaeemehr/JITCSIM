from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab as plt


def plot_order(x, y, filename, xlabel=None, ylabel=None, close_fig=True):
    """
    plot y vs x with given x and y labels
    """

    plt.style.use('ggplot')
    fig, ax = plt.subplots(1, figsize=(6, 4))
    ax.plot(x, y, lw=1, color='b')

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.margins(x=0.01)
    ax.set_ylim(0, 1.1)
    fig.savefig(filename, dpi=150)
    if close_fig:
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
