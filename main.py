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
data_N62 = np.load(current_directory + "/initial_grids/B_N62_Zc1.npz")
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
iterations = 20000

# Standard
start_time = time()
e_lib_st, e_tot_st, B_st, grid_list_st, area_list_st = lu_ham_standard(
    B_N62, N, Zc, iterations)
print("LU&H Standard: --- %.4f seconds ---" % (time() - start_time))

# Deterministic
start_time = time()
e_lib_det, e_tot_det, B_det, grid_list_det, area_list_det = lu_ham_deterministic(
    B_N62, Z_c=1, N_i=iterations, eps=0.001, D_nc=0.1)
print("LU&H Deterministic --- %.4f seconds ---" % (time() - start_time))

# %%

# Avalanche active distribution nodes
start_time = time()
total_avalanche_areas_st, per_avalanche_areas_st = avalanche_areas_func(
    area_list_st)
print("avalanche_areas_func Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
total_avalanche_areas_det, per_avalanche_areas_det = avalanche_areas_func(
    area_list_det)
print("avalanche_areas_func Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# Avalanche covered areas as the sum of active nodes
start_time = time()
avalanche_covered_areas_st = calculate_covered_areas(total_avalanche_areas_st)
print("calculate_covered_areas Standard: --- %.4f seconds ---" %
      (time() - start_time))

start_time = time()
avalanche_covered_areas_det = calculate_covered_areas(total_avalanche_areas_det)
print("calculate_covered_areas Deterministic --- %.4f seconds ---" %
      (time() - start_time))
# %%

# Number of active nodes per avalanche
start_time = time()
number_of_nodes_per_avalanche_st = node_count_in_avalanche(
    per_avalanche_areas_st)
print("node_count_in_avalanche Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
number_of_nodes_per_avalanche_det = node_count_in_avalanche(
    per_avalanche_areas_det)
print("node_count_in_avalanche Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# Cluster in avalanche areas
start_time = time()
cluster_list_st = [csr_matrix(measure.label(
    avalanche_area > 0)) for avalanche_area in total_avalanche_areas_st]
print("cluster_list_st Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
cluster_list_det = [csr_matrix(measure.label(
    avalanche_area > 0)) for avalanche_area in total_avalanche_areas_det]
print("cluster_list_det Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# Number of clusters per avalanche areas
start_time = time()
number_of_clusters_st = [cluster_matrix.max()
                         for cluster_matrix in cluster_list_st]
print("number_of_clusters_st Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
number_of_clusters_det = [cluster_matrix.max()
                          for cluster_matrix in cluster_list_det]
print("number_of_clusters_det Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# Cluster sizes for all avalanches
start_time = time()
cluster_sizes_st = [item for cluster_matrix in cluster_list_st for item in np.unique(cluster_matrix.data, return_counts=True)[
    1][0:cluster_matrix.max()]]
print("cluster_sizes_st Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
cluster_sizes_det = [item for cluster_matrix in cluster_list_det for item in np.unique(cluster_matrix.data, return_counts=True)[
    1][0:cluster_matrix.max()]]
print("cluster_sizes_det Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

fit=True

# A
start_time = time()
xe, ye, fit_ye = distribution_to_plot(avalanche_covered_areas_st)
distribution_plot(xe, ye, fit_ye, "A",
                  scale="log", fit=fit, save=True)
print("distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(avalanche_covered_areas_det)
distribution_plot(xe, ye, fit_ye, "A",
                  scale="log", fit=fit, save=True)
print("distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# Number of nodes at the avalanche peak
start_time = time()
number_of_nodes_at_peak_st = node_count_in_avalanches_peak(per_avalanche_areas_st)
print("node_count_in_avalanches_peak Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
number_of_nodes_at_peak_det = node_count_in_avalanches_peak(per_avalanche_areas_det)
print("node_count_in_avalanches_peak Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# A*
start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_nodes_at_peak_st)
distribution_plot(xe, ye, fit_ye, "A^{*}",
                  scale="log", fit=fit, save=True)
print("distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_nodes_at_peak_det)
distribution_plot(xe, ye, fit_ye, "A^{*}",
                  scale="log", fit=fit, save=True)
print("distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# D (fractal index)
start_time = time()
fractal_index_st = []
for matrix in total_avalanche_areas_st:
    fractal_index_st.append(fractal_index(matrix))
print("fractal_index Standard: --- %.4f seconds ---" % (time() - start_time))

plt.hist(fractal_index_st, bins=20, density=False)
plt.xlabel(r"$"+"D_{st}"+"$", size=20)
plt.ylabel(r"PDF($"+"D_{st}"+"$)", size=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(c='black', alpha=0.3)
# plt.tight_layout()
plt.show()

print("Mean Fractal Index St: " + str(np.mean(fractal_index_st)))

start_time = time()
fractal_index_det = []
for matrix in total_avalanche_areas_det:
    fractal_index_det.append(fractal_index(matrix))
print("fractal_index Deterministic: --- %.4f seconds ---" % (time() - start_time))

plt.hist(fractal_index_det, bins=20, density=False)
plt.xlabel(r"$"+"D_{det}"+"$", size=20)
plt.ylabel(r"PDF($"+"D_{det}"+"$)", size=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(c='black', alpha=0.3)
# plt.tight_layout()
plt.show()

print("Mean Fractal Index det: " + str(np.mean(fractal_index_det)))
# %%

# Lista con índices donde estan los 0 de e_soc_st.
lim_a_st = np.where(np.array(e_lib_st) == 0)[0]

# Lista con índices donde estan los 0 de e_soc_det.
lim_a_det = np.where(np.array(e_lib_det) == 0)[0]

start_time = time()
T_st = duraciones(lim_a_st)
print("T Standard --- %.4f seconds ---" % (time() - start_time))

start_time = time()
T_det = duraciones(lim_a_det)
print("T Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
E_st, P_st, tes_st, t_ac_st, E_ac_st, t_rel_st, E_rel_st, t_ac_pesado_st, t_rel_pesado_st = energia_picos(lim_a_st, e_lib_st)
print("E, P, tes Standard --- %.4f seconds ---" % (time() - start_time))

start_time = time()
E_det, P_det, tes_det, t_ac_det, E_ac_det, t_rel_det, E_rel_det, t_ac_pesado_det, t_rel_pesado_det = energia_picos(lim_a_det, e_lib_det)
print("E, P, tes Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# Tiempo entre picos Standard
start_time = time()
t_P_st = tiempo_entre_picos(tes_st)
print("t_P tiempo_entre_picos Standard --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre picos Deterministic
start_time = time()
t_P_det = tiempo_entre_picos(tes_det)
print("t_P tiempo_entre_picos Deterministic --- %.4f seconds ---" % (time() - start_time))

# Tiempo ente fin e inicio de avalanchas Standard
start_time = time()
t_fi_st = tiempo_fin_inicio(lim_a_st)
print("t_fi tiempo_fin_inicio Standard --- %.4f seconds ---" % (time() - start_time))

# Tiempo ente fin e inicio de avalanchas Deterministic
start_time = time()
t_fi_det = tiempo_fin_inicio(lim_a_det)
print("t_fi tiempo_fin_inicio Deterministic --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre inicios de avalanchas Standard
start_time = time()
t_ii_st = tiempo_inicio_inicio(lim_a_st)
print("t_ii tiempo_inicio_inicio Standard --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre inicios de avalanchas Deterministic
start_time = time()
t_ii_det = tiempo_inicio_inicio(lim_a_det)
print("t_ii tiempo_inicio_inicio Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# E
start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_st)
distribution_plot(xe, ye, fit_ye, "E_{st}",
                  scale="log", fit=fit, save=True)
print("E distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_det)
distribution_plot(xe, ye, fit_ye, "E_{det}",
                  scale="log", fit=fit, save=True)
print("E distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# P
start_time = time()
xe, ye, fit_ye = distribution_to_plot(P_st)
distribution_plot(xe, ye, fit_ye, "P_{st}",
                  scale="log", fit=fit, save=True)
print("P distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(P_det)
distribution_plot(xe, ye, fit_ye, "P_{det}",
                  scale="log", fit=fit, save=True)
print("P distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# T
start_time = time()
xe, ye, fit_ye = distribution_to_plot(T_st)
distribution_plot(xe, ye, fit_ye, "T_{st}",
                  scale="log", fit=fit, save=True)
print("T distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(T_det)
distribution_plot(xe, ye, fit_ye, "T_{det}",
                  scale="log", fit=fit, save=True)
print("T distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# t_P
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_P_st)
distribution_plot(xe, ye, fit_ye, "t_{P_{st}}",
                  scale="log", fit=fit, save=True)
print("t_P distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_P_det)
distribution_plot(xe, ye, fit_ye, "t_{P_{det}}",
                  scale="log", fit=fit, save=True)
print("t_P distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# t_fi
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_fi_st)
distribution_plot(xe, ye, fit_ye, "t_{{fi}_{st}}",
                  scale="semilog", fit=fit, save=True)
print("t_fi distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_fi_det)
distribution_plot(xe, ye, fit_ye, "t_{{fi}_{det}}",
                  scale="semilog", fit=fit, save=True)
print("t_fi distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# t_ii
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ii_st)
distribution_plot(xe, ye, fit_ye, "t_{{ii}_{st}}",
                  scale="log", fit=fit, save=True)
print("t_ii distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ii_det)
distribution_plot(xe, ye, fit_ye, "t_{{ii}_{det}}",
                  scale="log", fit=fit, save=True)
print("t_ii distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# t_ac
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_st)
distribution_plot(xe, ye, fit_ye, "t_{{ac}_{st}}",
                  scale="log", fit=fit, save=True)
print("t_ac distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_det)
distribution_plot(xe, ye, fit_ye, "t_{{ac}_{det}}",
                  scale="log", fit=fit, save=True)
print("t_ac distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# t_rel
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_st)
distribution_plot(xe, ye, fit_ye, "t_{{rel}_{st}}",
                  scale="log", fit=fit, save=True)
print("t_rel distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_det)
distribution_plot(xe, ye, fit_ye, "t_{{rel}_{det}}",
                  scale="log", fit=fit, save=True)
print("t_rel distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# E_ac
start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_ac_st)
distribution_plot(xe, ye, fit_ye, "E_{{ac}_{st}}",
                  scale="log", fit=fit, save=True)
print("E_ac distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_ac_det)
distribution_plot(xe, ye, fit_ye, "E_{{ac}_{det}}",
                  scale="log", fit=fit, save=True)
print("E_ac distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# E_rel
start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_rel_st)
distribution_plot(xe, ye, fit_ye, "E_{{rel}_{st}}",
                  scale="log", fit=fit, save=True)
print("E_rel distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_rel_det, normal=True)
# distribution_plot(xe, ye, fit_ye, "E_{rel_det}",
#                   scale="log", fit=fit, save=True)
plt.plot(xe, ye, "o", mfc="none", label="E_rel_det", markersize=9)
plt.show()
print("E_rel distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# t_ac_pesado
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_pesado_st, normal=True)
# distribution_plot(xe, ye, fit_ye, "t_{ac_pesado_st}",
#                   scale="log", fit=fit, save=True)
plt.plot(xe, ye, "o", mfc="none", label="t_ac_pesado_st", markersize=9)
plt.show()
print("t_ac_pesado distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_pesado_det, normal=True)
# distribution_plot(xe, ye, fit_ye, "t_{ac_pesado_det}",
#                   scale="log", fit=fit, save=True)
plt.plot(xe, ye, "o", mfc="none", label="t_ac_pesado_det", markersize=9)
plt.show()
print("t_ac_pesado distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# t_rel_pesado
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_pesado_st, normal=True)
# distribution_plot(xe, ye, fit_ye, "t_{rel_pesado_st}",
#                   scale="log", fit=fit, save=True)
plt.plot(xe, ye, "o", mfc="none", label="t_rel_pesado_st", markersize=9)
plt.show()
print("t_rel_pesado distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_pesado_det, normal=True)
# distribution_plot(xe, ye, fit_ye, "t_{rel_pesado_det}",
#                   scale="log", fit=fit, save=True)
plt.plot(xe, ye, "o", mfc="none", label="t_rel_pesado_det", markersize=9)
plt.show()
print("t_rel_pesado distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))
# %%

# Eventos Extremos
start_time = time()
T_ext_st, E_ext_st, P_ext_st, t_P_ext_st, t_fi_ext_st, t_ii_ext_st, t_ac_ext_st, E_ac_ext_st, t_rel_ext_st, E_rel_ext_st, t_ac_pesado_ext_st, t_rel_pesado_ext_st = eventos_extremos(e_lib_st, 10)
print("Extreme Events Standard --- %.4f seconds ---" % (time() - start_time))