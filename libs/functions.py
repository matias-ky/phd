# Here I will put the functions

import numpy as np
import random
from numba import jit
from time import time
from libs.areas import *


@jit(nopython=True)
def set_seed(seed):
    """
    Set the seed for Numba jit decorated 'random' function calls.

    Parameters
    ----------
    seed : int
        Seed value for random number generation.

    Returns
    -------
    None

    Raises
    ------
    None

    Notes
    -----
    This function sets the seed value for random number generation when using Numba jit decorated functions.
    It ensures that the random numbers generated by Numba jit are reproducible.

    Examples
    --------
    >>> set_seed(42)
    """
    np.random.seed(seed)
    random.seed(seed)


@jit(nopython=True)
def lu_ham_standard(B, N, Zc, iterations):
    """
    Apply cellular automaton rules to a grid.

    Parameters
    ----------
    B : numpy.ndarray
        Initial grid.
    N : int
        Size of the grid.
    Zc : float
        Parameter Zc.
    iterations : int
        Number of iterations.

    Returns
    -------
    tuple
        A tuple containing:
            - e_lib : List of energy values at each iteration.
            - e_tot : List of total energy values at each iteration.
            - final_grid : numpy.ndarray : Final grid state.
            - grid_states : List of grid states at each iteration.
            - area_states : List of corresponding area states.

    Raises
    ------
    None

    Notes
    -----
    This function applies cellular automaton rules to a grid represented by
    matrix `B`. It iterates for a specified number of times (`iterations`)
    and updates the grid state according to the specified rules.

    Examples
    --------
    >>> initial_grid = np.zeros((10, 10), dtype=np.float32)
    >>> energy_list, total_energy_list, final_grid, grid_states, area_states = cellular_automaton(initial_grid, N=62, Zc=1, iterations=1000)
    """

    C = np.zeros((N+2, N+2), dtype=np.float32)  # Initialize C matrix
    M = np.zeros((N+2, N+2), dtype=np.float32)  # Initialize M matrix
    D = 2  # Dimension
    s = 2*D+1  # Calculate 's'
    grid_list = [B]  # Initialize list to store grids
    area_list = [B]  # Initialize list to store areas
    e_lib = []  # Initialize list to store e values
    e_tot = []  # Initialize list to store total energy values
    zc_s_over_s = Zc / s  # Calculate zc_s_over_s
    two_d_s_over_s_zc = ((2*D) / s) * Zc  # Calculate two_d_s_over_s_zc

    keep_going = True
    i = 0

    while keep_going:
        i += 1
        e = 0  # Initialize energy variable

        # Iterate through the grid
        for j in range(1, N+1):
            for k in range(1, N+1):
                Z = B[j, k] - (1/(2*D)) * (B[j+1, k] +
                                           B[j-1, k] + B[j, k+1] + B[j, k-1])
                abs_z = abs(Z)

                # Check if condition is met for energy update
                if abs_z > Zc:
                    C[j, k] = C[j, k] - two_d_s_over_s_zc
                    C[j+1, k] = C[j+1, k] + zc_s_over_s
                    C[j-1, k] = C[j-1, k] + zc_s_over_s
                    C[j, k+1] = C[j, k+1] + zc_s_over_s
                    C[j, k-1] = C[j, k-1] + zc_s_over_s
                    M[j, k] = 1
                    M[j+1, k] = 1
                    M[j-1, k] = 1
                    M[j, k+1] = 1
                    M[j, k-1] = 1
                    g = two_d_s_over_s_zc * ((2*abs_z/Zc) - 1) * Zc
                    e = e + g
                else:
                    continue

        if e > 0:  # Update the grid if energy is positive
            B += C
            C = np.zeros((N+2, N+2), dtype=np.float32)
        else:  # Randomly update a cell if energy is non-positive
            if i > iterations: # This guarantees that it ends after 0 energy release
                keep_going = False
            k_prime = [random.randint(1, N+1), random.randint(1, N+1)]
            delta_B = random.random() - 0.2
            B[k_prime[0], k_prime[1]] = B[k_prime[0], k_prime[1]] + delta_B

        # Set boundary conditions
        B[0, :] = 0
        B[:, 0] = 0
        B[N+1, :] = 0
        B[:, N+1] = 0
        M[0, :] = 0
        M[:, 0] = 0
        M[N+1, :] = 0
        M[:, N+1] = 0

        # Append energy values to lists
        e_lib.append(e)
        e_tot.append(np.sum(np.square(B)))
        grid_list.append(B.copy())
        area_list.append(M)

        M = np.zeros((N+2, N+2), dtype=np.float32)  # Reset M matrix

    del grid_list[0]  # Delete initial grid from the list
    del area_list[0]  # Delete initial area from the list

    return e_lib, e_tot, B, grid_list, area_list  # Return the lists and final grid


@jit(nopython=True)
def lu_ham_deterministic(B, Z_c, N_i, eps, D_nc):
    """
    Apply cellular automaton rules to a grid for a deterministic global parameter model.

    Parameters
    ----------
    B : numpy.ndarray
        Initial grid.
    Z_c : float
        Parameter for Z_c extraction.
    N_i : int
        Number of iterations.
    eps : float
        Global redistribution parameter.
    D_nc : float
        Non-conservative redistribution parameter.

    Returns
    -------
    tuple
        A tuple containing:
            - e_lib : List of normalized energy values at each iteration.
            - e_tot : List of total energy values at each iteration.
            - final_grid : numpy.ndarray : Final grid state.
            - grid_states : List of grid states at each iteration.
            - area_states : List of corresponding area states.

    Notes
    -----
    This function applies cellular automaton rules to a grid represented by
    matrix `B` based on a deterministic global parameter model. It iterates for a specified
    number of times (`N_i`) and updates the grid state according to the specified rules.

    Examples
    --------
    >>> initial_grid = np.zeros((10, 10), dtype=np.float32)
    >>> energy_list, total_energy_list, final_grid, grid_states, area_states = lu_ham_deterministic(initial_grid, Z_c=1, N_i=1000, eps=0.1, D_nc=0.2)
    """

    N = len(B)
    C = np.zeros((N, N))
    M = np.zeros((N+2, N+2), dtype=np.float32)
    e_0 = 4*(Z_c**2)/5
    e_lib = []  # List to store normalized energy values in each collapse
    e_tot = []  # List to store total energy values
    grid_states = [B]  # List to store grid states
    area_states = [B]  # List to store area states

    keep_going = True
    k = 0

    while keep_going:
        k += 1
        e = 0
        # Z = np.empty((N, N))
        r_0 = random.uniform(D_nc, 1)

        for i in range(1, N-1):  # Non-conservative redistribution
            for j in range(1, N-1):
                Z = B[i, j] - (1/4)*(B[i-1, j]+B[i, j-1]+B[i+1, j]+B[i, j+1])
                if abs(Z) > Z_c:
                    C[i, j] = C[i, j]-(4/5)*Z_c
                    C[i-1, j] = C[i-1, j]+(r_0/5)*Z_c
                    C[i+1, j] = C[i+1, j]+(r_0/5)*Z_c
                    C[i, j-1] = C[i, j-1]+(r_0/5)*Z_c
                    C[i, j+1] = C[i, j+1]+(r_0/5)*Z_c
                    M[i, j] = 1
                    M[i+1, j] = 1
                    M[i-1, j] = 1
                    M[i, j+1] = 1
                    M[i, j-1] = 1

                    B_ii = B[i-1, j] + B[i, j-1] + B[i+1, j] + B[i, j+1]
                    # g = -(4/5)*(((B_ii/2)*r_0/Z_c) + ((r_0**2)/5) - (2*B[i, j]/Z_c) + 4/5)*(Z_c**2)
                    
                    # There is a problem with B_ii but it is the correct formula
                    abs_z = abs(Z)
                    g = (4/5) * Z_c * ((2*abs_z/Z_c) - 1) * Z_c + ((1 - r_0)/(2*Z_c)) * B_ii + (1 - r_0**2)/5
                    e = e+g

        if e > 0:  # If there was a collapse e>0, then update.
            for i in range(1, N-1):
                for j in range(1, N-1):
                    B[i, j] = B[i, j]+C[i, j]  # Update the field
                    C[i, j] = 0
        else:  # Global Driving
            if k > N_i: # This guarantees that it ends after 0 energy release
                keep_going = False
            for i in range(1, N-1):  # Redistribution
                for j in range(1, N-1):
                    B[i, j] = B[i, j]*(1+eps)

        e_lib.append(e/e_0)
        e_tot.append(np.sum(np.square(B)))
        grid_states.append(B.copy())
        area_states.append(M)
        M = np.zeros((N+2, N+2), dtype=np.float32)

    del grid_states[0]
    del area_states[0]
    # Return the lists and final grid
    return e_lib, e_tot, B, grid_states, area_states

@jit(nopython=True)
def dgd_random_redistribution(B, N, Zc, iterations, eps=0.001):
    """
    Deterministic Global Driving with Random Redistribution.
    """

    C = np.zeros((N+2, N+2), dtype=np.float32)  # Initialize C matrix
    M = np.zeros((N+2, N+2), dtype=np.float32)  # Initialize M matrix
    D = 2  # Dimension
    s = 2*D+1  # 5 for 2D lattice
    grid_list = [B]  # Initialize list to store grids
    area_list = [B]  # Initialize list to store areas
    e_lib = []  # Initialize list to store e values
    e_tot = []  # Initialize list to store total energy values
    zc_s_over_s = Zc / s  # Zc/5 for 2D lattice
    two_d_s_over_s_zc = ((2*D) / s) * Zc  # 4/5 for 2D lattice

    r1 = random.uniform(0, 1)
    r2 = random.uniform(r1, 1)
    r3 = random.uniform(r2, 1)
    r4 = 1 - r1 - r2 - r3

    keep_going = True
    i = 0

    while keep_going:
        i += 1
        e = 0  # Initialize energy variable

        # Iterate through the grid
        for j in range(1, N+1):
            for k in range(1, N+1):
                Z = B[j, k] - (1/(2*D)) * (B[j+1, k] +
                                           B[j-1, k] + B[j, k+1] + B[j, k-1])
                abs_z = abs(Z)

                # Check if condition is met for energy update
                if abs_z > Zc:
                    C[j, k] = C[j, k] - two_d_s_over_s_zc
                    C[j+1, k] = C[j+1, k] + (4/5) * r1 * zc_s_over_s
                    C[j-1, k] = C[j-1, k] + (4/5) * r2 * zc_s_over_s
                    C[j, k+1] = C[j, k+1] + (4/5) * r3 * zc_s_over_s
                    C[j, k-1] = C[j, k-1] + (4/5) * r4 * zc_s_over_s
                    M[j, k] = 1
                    M[j+1, k] = 1
                    M[j-1, k] = 1
                    M[j, k+1] = 1
                    M[j, k-1] = 1
                    g = two_d_s_over_s_zc * ((2*abs_z/Zc) - 1) * Zc
                    e = e + g
                else:
                    continue

        if e > 0:  # Update the grid if energy is positive
            B += C
            C = np.zeros((N+2, N+2), dtype=np.float32)
        else:  # Randomly update a cell if energy is non-positive
            if i > iterations: # This guarantees that it ends after 0 energy release
                keep_going = False
            for j in range(1, N+1):
                for k in range(1, N+1):
                    B[j, k] = B[j, k] * (1 + eps)

        # Set boundary conditions
        B[0, :] = 0
        B[:, 0] = 0
        B[N+1, :] = 0
        B[:, N+1] = 0
        M[0, :] = 0
        M[:, 0] = 0
        M[N+1, :] = 0
        M[:, N+1] = 0

        # Append energy values to lists
        e_lib.append(e)
        e_tot.append(np.sum(np.square(B)))
        grid_list.append(B.copy())
        area_list.append(M)

        M = np.zeros((N+2, N+2), dtype=np.float32)  # Reset M matrix

    del grid_list[0]  # Delete initial grid from the list
    del area_list[0]  # Delete initial area from the list

    return e_lib, e_tot, B, grid_list, area_list  # Return the lists and final grid

def simulacion_completa(B, N, Zc, iterations, model):
    retry = 1

    for _ in range(retry):

        try:
            start_time_tot = time()

            e_lib_tot = []
            e_tot_tot = []
            B_tot = []
            areas_cubiertas_tot = []
            number_of_nodes_at_peak_tot = []
            cantidad_de_nodos_en_avalanchas_tot = []
            # lista_de_clusters_tot = []
            number_of_clusters_tot = []
            cluster_sizes_tot = []

            chunks_range = 2000

            for iter_tot in range(chunks_range):
                # start_time_iter = time()
                # print("Arranca iteracion " + str(iter_tot))

                # CLÁSICO
                # start_time = time()
                if model == "standard":
                    e_lib, e_tot, B, lista_de_grillas, lista_de_areas = lu_ham_standard(B, N, Zc, iterations)
                # print("--- %.4f seconds --- e_lib" % (time() - start_time))

                # DETERMINISTA
                # start_time = time()
                if model == "deterministic":
                    e_lib, e_tot, B, lista_de_grillas, lista_de_areas = lu_ham_deterministic(
                        B, Z_c=Zc, N_i=iterations, eps=0.001, D_nc=0.1)
                # print("--- %.4f seconds --- e_lib" % (time() - start_time))

                # RANDOM REDISTRIBUTION
                # start_time = time()
                if model == "random_redistribution":
                    e_lib, e_tot, B, lista_de_grillas, lista_de_areas = dgd_random_redistribution(
                        B, N, Zc, iterations, eps=0.001)
                # print("--- %.4f seconds --- e_lib" % (time() - start_time))

                # # start_time = time()
                # # grillas_con_nodos_inestables = nodos_inestables(np.array(lista_de_grillas), Zc,
                # #                                                 Zc_porcentaje=0.8)
                # # print("--- %.4f seconds --- grillas" % (time() - start_time))

                # start_time = time()
                areas_de_avalanchas, areas_de_avalanchas_por_avalancha = avalanche_areas_func(
                    lista_de_areas)
                # print("--- %.4f seconds --- areas de avalanchas" %
                #     (time() - start_time))

                # start_time = time()
                areas_cubiertas = calculate_covered_areas(
                    List(areas_de_avalanchas))
                # print("--- %.4f seconds --- areas cubiertas" %
                #     (time() - start_time))

                # # Create a typed list from areas_de_avalanchas
                # typed_areas_de_avalanchas = List()
                # for area_de_avalancha in areas_de_avalanchas:
                #     typed_areas_de_avalanchas.append(area_de_avalancha)

                # # Call the function with the typed list
                # areas_cubiertas = calcular_areas_cubiertas(
                #     typed_areas_de_avalanchas)

                # Number of nodes at the avalanche peak
                number_of_nodes_at_peak = node_count_in_avalanches_peak(areas_de_avalanchas_por_avalancha)

                # start_time = time()
                cantidad_de_nodos_en_avalanchas = node_count_in_avalanche(
                    areas_de_avalanchas_por_avalancha)
                # print("--- %.4f seconds --- nodos en avalanchas" %
                #     (time() - start_time))

                # # start_time = time()
                # # nodos_inestasbles_antes = nodos_inest_antes_de_avalanchar(
                # #     grillas_con_nodos_inestables)
                # # print("--- %.4f seconds --- nodos inest antes" % (time() - start_time))

                # start_time = time()
                # lista_de_clusters = [csr_matrix(find_clusters(
                #     area_de_avalancha)) for area_de_avalancha in areas_de_avalanchas]
                # print("--- %.4f seconds --- hoshen kopelman" %
                #       (time() - start_time))

                # start_time = time()
                lista_de_clusters = [csr_matrix(measure.label(
                    area_de_avalancha > 0)) for area_de_avalancha in areas_de_avalanchas]
                # print("--- %.4f seconds --- hoshen kopelman" %
                #     (time() - start_time))

                # Number of clusters per avalanche areas
                number_of_clusters = [cluster_matrix.max()
                                        for cluster_matrix in lista_de_clusters]
                
                # Cluster sizes for all avalanches
                cluster_sizes = [item for cluster_matrix in lista_de_clusters for item in np.unique(cluster_matrix.data, return_counts=True)[
                    1][0:cluster_matrix.max()]]


                e_lib_tot = e_lib_tot + e_lib
                e_tot_tot = e_tot_tot + e_tot
                B_tot.extend(B)
                areas_cubiertas_tot = areas_cubiertas_tot + areas_cubiertas
                number_of_nodes_at_peak_tot = number_of_nodes_at_peak_tot + \
                    number_of_nodes_at_peak
                cantidad_de_nodos_en_avalanchas_tot = cantidad_de_nodos_en_avalanchas_tot + \
                    cantidad_de_nodos_en_avalanchas
                # lista_de_clusters_tot += lista_de_clusters
                number_of_clusters_tot += number_of_clusters
                cluster_sizes_tot += cluster_sizes
                del lista_de_grillas
                del lista_de_areas
                del areas_de_avalanchas
                del areas_de_avalanchas_por_avalancha
                del lista_de_clusters

                # print("--- %.4f seconds --- iter" % (time() - start_time_iter))
                if iter_tot%(chunks_range/10) == 0:
                    print("Complete " + str(100*iter_tot/chunks_range) + "%" + " --- %.4f seconds ---" % (time() - start_time_tot))

            # print("--- %.4f seconds ---" % (time() - start_time_tot))
            break

        except:
            pass

    return e_lib_tot, e_tot_tot, B_tot, areas_cubiertas_tot, number_of_nodes_at_peak_tot, cantidad_de_nodos_en_avalanchas_tot, number_of_clusters_tot, cluster_sizes_tot