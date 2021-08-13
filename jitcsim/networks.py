import numpy as np
import networkx as nx
# import pylab as plt
# from copy import copy
# from jitcsim.utility import is_symmetric


class make_network:
    ''' 
    make different graphs ans return their adjacency matrices
    as a 1 dimensional double vector in stl library
    '''

    def __init__(self, seed=None):
        self.G = 0

        if seed:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = None

    def complete(self, N):
        ''' 
        make complete all to all adjacency matrix
        
        Parameters
        ----------

        N : int
            number of nodes

        Return : A
            2D int numpy array, adjacency matrix 
        
        '''

        self.N = N
        self.G = nx.complete_graph(N)
        A = nx.to_numpy_array(self.G, dtype=int)

        return A

    def erdos_renyi(self, N, p, directed=False):
        ''' 
        make Erdos Renyi network 

        Parameters
        ----------

        N : int
            number of nodes
        p : float
            Probability for edge creation.
        directed : (bool, optional (default=False)) 
            If `True`, this function returns a directed adjacency matrix.
        Return : A
            2D int numpy array, adjacency matrix 
        
        '''

        self.N = N
        self.G = nx.erdos_renyi_graph(N, p, directed=directed, seed=self.seed)
        A = nx.to_numpy_array(self.G, dtype=int)

        return A

    def barabasi(self, N, m):
        ''' 
        Return random network using Barabási-Albert preferential attachment model.
        A graph of n nodes is grown by attaching new nodes each with m edges that are preferentially attached to existing nodes with high degree.

        This is `networkx.barabasi_albert_graph` module.

        Parameters
        -----------
        n : int
            Number of nodes
        m : int
            Number of edges to attach from a new node to existing nodes
        Return : A
            2D int numpy array, adjacency matrix 

        '''

        self.N = N
        self.G = nx.barabasi_albert_graph(N, m, seed=self.seed)
        A = nx.to_numpy_matrix(self.G, dtype=int)

        return A

    def fgc(self,
            N,
            k,
            omega,
            gamma=0.4):
        """
        Frequency Gap-conditioned (FGC) network
        :param: N [int] the number of oscillators in the system
        :param: k [int] degree of the network
        :param: gamma [float] minimal frequency gap
        """

        # the number of links in the network
        L = N*k //2

        # the natural frequencies follow a uniform distribution
        # if omega is None:
        #     omega = np.random.uniform(low=0, high=1, size=N)

        # initialize the adjacency matrix
        A = np.zeros((N, N), dtype=int)

        # construct FGC random network
        counter = 0
        num_trial = 0
        while counter < L:

            num_trial += 1
            i, j = np.random.choice(range(N), size=2, replace=False)
            if (abs(omega[i]-omega[j]) > gamma) and (A[i][j] == 0):
                A[i][j] = 1
                counter += 1
                num_trial = 0

            if (num_trial > 10000):
                print("adding edge stuck!")
                exit(0)

        # G = nx.from_numpy_array(A)
        # assert (nx.is_connected(G))
        # assert(is_symmetric(A))

        return A
        #---------------------------------------------------------------#
