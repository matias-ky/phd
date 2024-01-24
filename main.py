# -*- coding: utf-8 -*-

#%%
# Import necessary libraries and functions
from libs.functions import *
from libs.areas import *
import numpy as np
from time import time
import os

# Set the program seed for consistency over the simulations
set_seed(7)

#%%
# Load data for N = 62
current_directory = os.getcwd()
data_N62 = np.load(current_directory + "/B_final_N62_Zc1.npz")
B_N62 = data_N62["B"]
B_N62 = B_N62.astype(np.float32)

#%%
# Set simulation parameters
N = 62
Zc = 1
iterations = 1

# Compile Numba jit function for cellular automaton
_, _, _, _, _ = lu_ham_standard(
    B_N62, N, Zc, iterations)
_, _, _, _, _ = lu_ham_deterministic(
    B_N62, Z_c=1, N_i=iterations, eps=0.001, D_nc=0.1)

#%%
# Run simulation with increased iterations
iterations = 10000

# Standard
start_time = time()
e_lib_st, e_tot_st, B_st, grid_list_st, area_list_st = lu_ham_standard(
    B_N62, N, Zc, iterations)
print("--- %s seconds ---" % (time() - start_time))

# Deterministic
start_time = time()
e_lib_det, e_tot_det, B_det, grid_list_det, area_list_det = lu_ham_deterministic(
    B_N62, Z_c=1, N_i=iterations, eps=0.001, D_nc=0.1)
print("--- %s seconds ---" % (time() - start_time))

#%%
# Avalanche active distribution nodes
total_avalanche_areas_st, per_avalanche_areas_st = avalanche_areas_func(
    area_list_st)

total_avalanche_areas_det, per_avalanche_areas_det = avalanche_areas_func(
    area_list_det)

#%%
# Avalanche covered areas as the sum of active nodes
avalanche_covered_areas_st = calculate_covered_areas(
    List(total_avalanche_areas_st))

avalanche_covered_areas_det = calculate_covered_areas(
    List(total_avalanche_areas_det))

#%%
# Number of active nodes per avalanche
number_of_nodes_per_avalanche_st = node_count_in_avalanche(
    per_avalanche_areas_st)

number_of_nodes_per_avalanche_det = node_count_in_avalanche(
    per_avalanche_areas_det)

#%%
# Cluster in avalanche areas
cluster_list_st = [csr_matrix(measure.label(
    avalanche_area > 0)) for avalanche_area in total_avalanche_areas_st]

cluster_list_det = [csr_matrix(measure.label(
    avalanche_area > 0)) for avalanche_area in total_avalanche_areas_det]

#%%
# Number of clusters per avalanche areas
number_of_clusters_st = [cluster_matrix.max()
                         for cluster_matrix in cluster_list_st]

number_of_clusters_det = [cluster_matrix.max()
                          for cluster_matrix in cluster_list_det]

#%%
# Cluster sizes for all avalanches
cluster_sizes_st = [item for cluster_matrix in cluster_list_st for item in np.unique(cluster_matrix.data, return_counts=True)[
    1][0:cluster_matrix.max()]]

cluster_sizes_det = [item for cluster_matrix in cluster_list_det for item in np.unique(cluster_matrix.data, return_counts=True)[
    1][0:cluster_matrix.max()]]
