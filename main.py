# -*- coding: utf-8 -*-

# %%
# Import necessary libraries and functions
from libs.functions import *
from libs.areas import *
from libs.analysis import *
from libs.utils import *
import numpy as np
from time import time
import os

# Set the program seed for consistency over the simulations
set_seed(7)

# %%
# Load data for N = 62
current_directory = os.getcwd()
data_N62 = np.load(current_directory + "/phd/B_final_N62_Zc1.npz")
B_N62 = data_N62["B"]
B_N62 = B_N62.astype(np.float32)

# %%
# Set simulation parameters
N = 62
Zc = 1
iterations = 1

# Compile Numba jit function for cellular automaton
_, _, _, _, _ = lu_ham_standard(
    B_N62, N, Zc, iterations)
_, _, _, _, _ = lu_ham_deterministic(
    B_N62, Z_c=1, N_i=iterations, eps=0.001, D_nc=0.1)

# %%
# Run simulation with increased iterations
iterations = 100000

# Standard
start_time = time()
e_lib_st, e_tot_st, B_st, grid_list_st, area_list_st = lu_ham_standard(
    B_N62, N, Zc, iterations)
print("LU&H Standard: --- %s seconds ---" % (time() - start_time))

# Deterministic
start_time = time()
e_lib_det, e_tot_det, B_det, grid_list_det, area_list_det = lu_ham_deterministic(
    B_N62, Z_c=1, N_i=iterations, eps=0.001, D_nc=0.1)
print("LU&H Deterministic --- %s seconds ---" % (time() - start_time))

# %%
# Avalanche active distribution nodes
start_time = time()
total_avalanche_areas_st, per_avalanche_areas_st = avalanche_areas_func(
    area_list_st)
print("avalanche_areas_func Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
total_avalanche_areas_det, per_avalanche_areas_det = avalanche_areas_func(
    area_list_det)
print("avalanche_areas_func Deterministic --- %s seconds ---" % (time() - start_time))
# %%
# Avalanche covered areas as the sum of active nodes
start_time = time()
avalanche_covered_areas_st = calculate_covered_areas(total_avalanche_areas_st)
print("calculate_covered_areas Standard: --- %s seconds ---" %
      (time() - start_time))

start_time = time()
avalanche_covered_areas_det = calculate_covered_areas(total_avalanche_areas_det)
print("calculate_covered_areas Deterministic --- %s seconds ---" %
      (time() - start_time))
# %%
# Number of active nodes per avalanche
start_time = time()
number_of_nodes_per_avalanche_st = node_count_in_avalanche(
    per_avalanche_areas_st)
print("node_count_in_avalanche Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
number_of_nodes_per_avalanche_det = node_count_in_avalanche(
    per_avalanche_areas_det)
print("node_count_in_avalanche Deterministic --- %s seconds ---" % (time() - start_time))
# %%
# Cluster in avalanche areas
start_time = time()
cluster_list_st = [csr_matrix(measure.label(
    avalanche_area > 0)) for avalanche_area in total_avalanche_areas_st]
print("cluster_list_st Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
cluster_list_det = [csr_matrix(measure.label(
    avalanche_area > 0)) for avalanche_area in total_avalanche_areas_det]
print("cluster_list_det Deterministic --- %s seconds ---" % (time() - start_time))
# %%
# Number of clusters per avalanche areas
start_time = time()
number_of_clusters_st = [cluster_matrix.max()
                         for cluster_matrix in cluster_list_st]
print("number_of_clusters_st Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
number_of_clusters_det = [cluster_matrix.max()
                          for cluster_matrix in cluster_list_det]
print("number_of_clusters_det Deterministic --- %s seconds ---" % (time() - start_time))
# %%
# Cluster sizes for all avalanches
start_time = time()
cluster_sizes_st = [item for cluster_matrix in cluster_list_st for item in np.unique(cluster_matrix.data, return_counts=True)[
    1][0:cluster_matrix.max()]]
print("cluster_sizes_st Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
cluster_sizes_det = [item for cluster_matrix in cluster_list_det for item in np.unique(cluster_matrix.data, return_counts=True)[
    1][0:cluster_matrix.max()]]
print("cluster_sizes_det Deterministic --- %s seconds ---" % (time() - start_time))
# %%
fit=True

# A
start_time = time()
xe, ye, fit_ye = distribution_to_plot(avalanche_covered_areas_st)
distribution_plot(xe, ye, fit_ye, "A",
                  scale="log", fit=fit, save=False)
print("distribution_to_plot Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(avalanche_covered_areas_det)
distribution_plot(xe, ye, fit_ye, "A",
                  scale="log", fit=fit, save=False)
print("distribution_to_plot Deterministic --- %s seconds ---" % (time() - start_time))
# %%
# Number of nodes at the avalanche peak
start_time = time()
number_of_nodes_at_peak_st = node_count_in_avalanches_peak(per_avalanche_areas_st)
print("node_count_in_avalanches_peak Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
number_of_nodes_at_peak_det = node_count_in_avalanches_peak(per_avalanche_areas_det)
print("node_count_in_avalanches_peak Deterministic --- %s seconds ---" % (time() - start_time))
# %%
# A*
start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_nodes_at_peak_st)
distribution_plot(xe, ye, fit_ye, "A^{*}",
                  scale="log", fit=fit, save=False)
print("distribution_to_plot Standard: --- %s seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_nodes_at_peak_det)
distribution_plot(xe, ye, fit_ye, "A^{*}",
                  scale="log", fit=fit, save=False)
print("distribution_to_plot Deterministic --- %s seconds ---" % (time() - start_time))
# %%
# D (fractal index)
start_time = time()
fractal_index_st = []
for matrix in total_avalanche_areas_st:
    fractal_index_st.append(fractal_index(matrix))
print("fractal_index Standard: --- %s seconds ---" % (time() - start_time))

plt.hist(fractal_index_st, bins=20, density=False)
plt.xlabel(r"$"+"D_{st}"+"$", size=20)
plt.ylabel(r"PDF($"+"D_{st}"+"$)", size=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(c='black', alpha=0.3)
plt.tight_layout()
plt.show()

print("Mean Fractal Index St: " + str(np.mean(fractal_index_st)))

start_time = time()
fractal_index_det = []
for matrix in total_avalanche_areas_det:
    fractal_index_det.append(fractal_index(matrix))
print("fractal_index Deterministic: --- %s seconds ---" % (time() - start_time))

plt.hist(fractal_index_det, bins=20, density=False)
plt.xlabel(r"$"+"D_{det}"+"$", size=20)
plt.ylabel(r"PDF($"+"D_{det}"+"$)", size=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(c='black', alpha=0.3)
plt.tight_layout()
plt.show()

print("Mean Fractal Index det: " + str(np.mean(fractal_index_det)))
# %%