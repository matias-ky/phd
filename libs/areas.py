
from numba import njit, prange
import numpy as np
from scipy.sparse import csr_matrix
from skimage import measure


@njit
def unstable_nodes(grid_list, Z_c, Z_c_percentage=0.8):
    """
    Identify unstable nodes in a list of grids.

    Parameters
    ----------
    grid_list : list of numpy.ndarray
        List of grids.
    Z_c : float
        Parameter for Z_c extraction.
    Z_c_percentage : float, optional
        Threshold percentage for considering nodes unstable, by default 0.8.

    Returns
    -------
    numpy.ndarray
        Array of grids with identified unstable nodes.

    Notes
    -----
    This function iterates through a list of grids and identifies unstable nodes
    based on the specified threshold percentage (`Z_c_percentage`). The identified
    nodes are marked as 1, and nodes exceeding the Z_c parameter are marked as 2.

    Examples
    --------
    >>> grid_list = [np.zeros((10, 10), dtype=np.float32)]
    >>> unstable_grids = (np.array(grid_list), Zc, Zc_porcentaje=0.8)
    """

    N = len(grid_list[0])
    D = 2

    unstable_grids = np.zeros((len(grid_list), N+2, N+2))

    for i in prange(len(grid_list)):
        B = grid_list[i]
        S = np.zeros((N+2, N+2))

        for j in range(1, N+1):
            for k in range(1, N+1):
                Z = B[j, k] - (1/(2*D))*(B[j+1, k] +
                                         B[j-1, k]+B[j, k+1]+B[j, k-1])
                if abs(Z) > Z_c_percentage:
                    S[j, k] = 1
                if abs(Z) > Z_c:
                    S[j, k] = 2

        unstable_grids[i] = S

    return unstable_grids


def avalanche_areas_func(area_list):
    """
    Calculate avalanche areas from a list of area matrices.

    Parameters
    ----------
    area_list : list of numpy.ndarray
        List of area matrices.

    Returns
    -------
    tuple
        A tuple containing:
            - total_avalanche_areas : List of total avalanche areas.
            - per_avalanche_areas : List of avalanche areas per avalanche.

    Notes
    -----
    This function processes a list of area matrices and calculates the total
    avalanche areas as well as the areas per individual avalanche.

    Examples
    --------
    >>> area_list = [np.zeros((10, 10), dtype=np.float32)]
    >>> total_areas, per_avalanche_areas = avalanche_areas_func(area_list)
    """

    total_avalanche_areas = []
    per_avalanche_areas = []
    current_avalanche = []

    for matrix in area_list:
        if (matrix > 0).any():
            current_avalanche.append(matrix)
        else:
            total_avalanche_areas.append(sum(current_avalanche))
            per_avalanche_areas.append(current_avalanche)
            current_avalanche = []

    total_avalanche_areas = [
        i for i in total_avalanche_areas if np.sum(i) != 0]
    per_avalanche_areas = [i for i in per_avalanche_areas if np.sum(i) != 0]

    return total_avalanche_areas, per_avalanche_areas


@njit
def calculate_covered_areas(avalanche_areas):
    """
    Calculate covered areas from a list of avalanche areas matrices.

    Parameters
    ----------
    avalanche_areas : list of numpy.ndarray
        List of avalanche areas matrices.

    Returns
    -------
    list
        List of covered areas.

    Notes
    -----
    This function processes a list of avalanche areas matrices and calculates
    the covered areas for each matrix.

    Examples
    --------
    >>> area_list = [np.zeros((10, 10), dtype=np.float32)]
    >>> covered_areas = calculate_covered_areas(area_list)
    """

    covered_areas = []

    for matrix in avalanche_areas:
        area = 0
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                if matrix[i][j] != 0:
                    area += 1
        covered_areas.append(area)

    return covered_areas


def node_count_in_avalanche(avalanche_areas_per_avalanche):
    """
    Calculate the number of nodes in each avalanche.

    Parameters
    ----------
    avalanche_areas_per_avalanche : list of numpy.ndarray
        List of area matrices per avalanche.

    Returns
    -------
    list
        List of node counts for each avalanche.

    Notes
    -----
    This function processes a list of area matrices per avalanche and calculates
    the number of nodes in each avalanche.

    Examples
    --------
    >>> area_list = [np.zeros((10, 10), dtype=np.float32)]
    >>> node_counts = node_count_in_avalanche(area_list)
    """

    node_counts_in_avalanches = []

    for matrices in avalanche_areas_per_avalanche:
        nodes = np.count_nonzero(np.array(matrices))
        node_counts_in_avalanches.append(nodes)

    return node_counts_in_avalanches


def unstable_nodes_before_avalanche(unstable_grids, e_lib):
    """
    Identify unstable nodes before an avalanche.

    Parameters
    ----------
    unstable_grids : numpy.ndarray
        Array of grids with identified unstable nodes.
    e_lib : list
        List of energy values at each iteration.

    Returns
    -------
    list
        List of grids with unstable nodes before an avalanche.

    Notes
    -----
    This function processes an array of grids with identified unstable nodes
    and a corresponding list of energy values (`e_lib`). It identifies the grids
    with unstable nodes before an avalanche occurs.

    Examples
    --------
    >>> unstable_grids = np.zeros((10, 10, 10), dtype=np.float32)
    >>> energy_list = [0.1, 0.2, 0.3, 0.0, 0.5]
    >>> unstable_nodes = unstable_nodes_before_avalanche(unstable_grids, energy_list)
    """
    unstable_nodes_before = []

    for i in range(len(e_lib)-1):
        if e_lib[i] == 0 and e_lib[i+1] != 0:
            unstable_nodes_before.append(unstable_grids[i])

    return unstable_nodes_before


def cluster_list_func(avalanche_areas):
    """
    Generate a list of matrices with labeled clusters for each avalanche area.

    Parameters
    ----------
    avalanche_areas : list of numpy.ndarray
        List of avalanche areas matrices.

    Returns
    -------
    list
        List of clusters represented as Compressed Sparse Row matrices.

    Notes
    -----
    This function processes a list of avalanche areas matrices and generates a
    list of clusters for each avalanche area.

    Examples
    --------
    >>> area_list = [np.zeros((10, 10), dtype=np.float32)]
    >>> clusters = cluster_list_func(area_list)
    """

    return [csr_matrix(measure.label(area > 0)) for area in avalanche_areas]


def number_of_clusters(total_cluster_list):
    """
    Calculate the number of clusters for each total cluster matrix.

    Parameters
    ----------
    total_cluster_list : list of numpy.ndarray
        List of total cluster matrices.

    Returns
    -------
    list
        List of the maximum cluster number for each total cluster matrix.

    Notes
    -----
    This function processes a list of total cluster matrices and calculates the
    maximum cluster number for each matrix.

    Examples
    --------
    >>> total_clusters = [np.zeros((10, 10), dtype=np.float32)]
    >>> cluster_numbers = number_of_clusters(total_clusters)
    """

    cluster_numbers = [cluster_matrix.max()
                       for cluster_matrix in total_cluster_list]

    return cluster_numbers
