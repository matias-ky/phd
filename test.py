# -*- coding: utf-8 -*-

# Import necessary libraries and functions
from libs.functions import *
from libs.areas import *
from libs.analysis import *
from libs.utils import *
import numpy as np
from time import time
import os
import logging

# Set logging level
logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=TIME_EXECUTION)

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

# TODO: Fix bug with lu_ham_standard compilation function. If not commented, it will make B matrix not to be in SOC state.
# Compile Numba jit function for cellular automaton
# _, _, _, _, _ = lu_ham_standard(
#     B_N62, N, Zc, iterations)
# _, _, _, _, _ = lu_ham_deterministic(
#     B_N62, Z_c=1, N_i=iterations, eps=0.001, D_nc=0.1)

# %%

# Run simulation with increased iterations (for each X iterations, there are Y chunks, so total will be X*Y)
# For example use 10 iterations and 2000 chunks, so total will be 20000 iterations
# However, simulation ends only when energy release is 0, so the number will mostly be higher than 20000
iterations = 100

# Standard
start_time = time()
e_lib_st, e_tot_st, B_st, avalanche_covered_areas_st, number_of_nodes_at_peak_st, number_of_nodes_per_avalanche_st, number_of_clusters_st, cluster_sizes_st, fractal_index_st = simulacion_completa(B_N62, N, Zc, iterations, "standard")
print("LU&H Standard: --- %.4f seconds ---" % (time() - start_time))
print(len(e_lib_st))
# plt.plot(e_tot_st)
# plt.show()

# Deterministic
start_time = time()
e_lib_det, e_tot_det, B_det, avalanche_covered_areas_det, number_of_nodes_at_peak_det, number_of_nodes_per_avalanche_det, number_of_clusters_det, cluster_sizes_det, fractal_index_det = simulacion_completa(B_N62, N, Zc, iterations, "deterministic")
print("LU&H Deterministic: --- %.4f seconds ---" % (time() - start_time))
print(len(e_lib_det))
# plt.plot(e_tot_det)
# plt.show()

# Deterministic Global Driving with Random Redistribution
start_time = time()
e_lib_dgdrr, e_tot_dgdrr, B_dgdrr, avalanche_covered_areas_dgdrr, number_of_nodes_at_peak_dgdrr, number_of_nodes_per_avalanche_dgdrr, number_of_clusters_dgdrr, cluster_sizes_dgdrr, fractal_index_dgdrr = simulacion_completa(B_N62, N, Zc, iterations, "random_redistribution")
print("LU&H Deterministic Global Driving with Random Redistribution: --- %.4f seconds ---" % (time() - start_time))
print(len(e_lib_dgdrr))
# plt.plot(e_tot_dgdrr)
# plt.show()

# %%

fit = True

# Fractal Index (D)
xe, ye, fit_ye = distribution_to_plot(fractal_index_st, normal=True)
distribution_plot(xe, ye, None, "D_{st}",
                  scale=None, fit=False, save=True)

logging.info("Mean Fractal Index St: " + str(np.mean(fractal_index_st)))

xe, ye, fit_ye = distribution_to_plot(fractal_index_det, normal=True)
distribution_plot(xe, ye, None, "D_{det}",
                  scale=None, fit=False, save=True)

logging.info("Mean Fractal Index det: " + str(np.mean(fractal_index_det)))

xe, ye, fit_ye = distribution_to_plot(fractal_index_dgdrr, normal=True)
distribution_plot(xe, ye, None, "D_{dgdrr}",
                    scale=None, fit=False, save=True)


# Number of clusters distribution
logging.info("Number of clusters distribution")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_clusters_st, normal=True)
distribution_plot(xe, ye, fit_ye, "NC_{st}",
                  scale=None, fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_clusters_det, normal=True)
distribution_plot(xe, ye, fit_ye, "NC_{det}",
                  scale=None, fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_clusters_dgdrr, normal=True)
distribution_plot(xe, ye, fit_ye, "NC_{dgdrr}",
                    scale=None, fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# Cluster Sizes Distribution
logging.info("Cluster Sizes Distribution (CS)")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(cluster_sizes_st)
distribution_plot(xe, ye, fit_ye, "CS_{st}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(cluster_sizes_det)
distribution_plot(xe, ye, fit_ye, "CS_{det}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(cluster_sizes_dgdrr)
distribution_plot(xe, ye, fit_ye, "CS_{dgdrr}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# A
logging.info("A")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(avalanche_covered_areas_st)
distribution_plot(xe, ye, fit_ye, "A_{st}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(avalanche_covered_areas_det)
distribution_plot(xe, ye, fit_ye, "A_{det}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(avalanche_covered_areas_dgdrr)
distribution_plot(xe, ye, fit_ye, "A_{dgdrr}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# A*
logging.info("A^{*}")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_nodes_at_peak_st)
distribution_plot(xe, ye, fit_ye, "A^{*}_{st}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_nodes_at_peak_det)
distribution_plot(xe, ye, fit_ye, "A^{*}_{det}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(number_of_nodes_at_peak_dgdrr)
distribution_plot(xe, ye, fit_ye, "A^{*}_{dgdrr}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# %%

# Lista con índices donde estan los 0 de e_soc_st.
lim_a_st = np.where(np.array(e_lib_st) == 0)[0]

# Lista con índices donde estan los 0 de e_soc_det.
lim_a_det = np.where(np.array(e_lib_det) == 0)[0]

# Lista con índices donde estan los 0 de e_soc_dgdrr.
lim_a_dgdrr = np.where(np.array(e_lib_dgdrr) == 0)[0]

start_time = time()
T_st = duraciones(lim_a_st)
time_execution_logger.log(TIME_EXECUTION, "T Standard --- %.4f seconds ---" % (time() - start_time))

start_time = time()
T_det = duraciones(lim_a_det)
time_execution_logger.log(TIME_EXECUTION, "T Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
T_dgdrr = duraciones(lim_a_dgdrr)
time_execution_logger.log(TIME_EXECUTION, "T DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

start_time = time()
E_st, P_st, tes_st, t_ac_st, E_ac_st, t_rel_st, E_rel_st, t_ac_pesado_st, t_rel_pesado_st = energia_picos(lim_a_st, e_lib_st)
time_execution_logger.log(TIME_EXECUTION, "E, P, tes Standard --- %.4f seconds ---" % (time() - start_time))

start_time = time()
E_det, P_det, tes_det, t_ac_det, E_ac_det, t_rel_det, E_rel_det, t_ac_pesado_det, t_rel_pesado_det = energia_picos(lim_a_det, e_lib_det)
time_execution_logger.log(TIME_EXECUTION, "E, P, tes Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
E_dgdrr, P_dgdrr, tes_dgdrr, t_ac_dgdrr, E_ac_dgdrr, t_rel_dgdrr, E_rel_dgdrr, t_ac_pesado_dgdrr, t_rel_pesado_dgdrr = energia_picos(lim_a_dgdrr, e_lib_dgdrr)
time_execution_logger.log(TIME_EXECUTION, "E, P, tes DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))
# %%

# Tiempo entre picos Standard
start_time = time()
t_P_st = tiempo_entre_picos(tes_st)
time_execution_logger.log(TIME_EXECUTION, "t_P tiempo_entre_picos Standard --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre picos Deterministic
start_time = time()
t_P_det = tiempo_entre_picos(tes_det)
time_execution_logger.log(TIME_EXECUTION, "t_P tiempo_entre_picos Deterministic --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre picos DGD Random Redistribution
start_time = time()
t_P_dgdrr = tiempo_entre_picos(tes_dgdrr)
time_execution_logger.log(TIME_EXECUTION, "t_P tiempo_entre_picos DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# Tiempo ente fin e inicio de avalanchas Standard
start_time = time()
t_fi_st = tiempo_fin_inicio(lim_a_st)
time_execution_logger.log(TIME_EXECUTION, "t_fi tiempo_fin_inicio Standard --- %.4f seconds ---" % (time() - start_time))

# Tiempo ente fin e inicio de avalanchas Deterministic
start_time = time()
t_fi_det = tiempo_fin_inicio(lim_a_det)
time_execution_logger.log(TIME_EXECUTION, "t_fi tiempo_fin_inicio Deterministic --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre fin e inicio de avalanchas DGD Random Redistribution
start_time = time()
t_fi_dgdrr = tiempo_fin_inicio(lim_a_dgdrr)
time_execution_logger.log(TIME_EXECUTION, "t_fi tiempo_fin_inicio DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre inicios de avalanchas Standard
start_time = time()
t_ii_st = tiempo_inicio_inicio(lim_a_st)
time_execution_logger.log(TIME_EXECUTION, "t_ii tiempo_inicio_inicio Standard --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre inicios de avalanchas Deterministic
start_time = time()
t_ii_det = tiempo_inicio_inicio(lim_a_det)
time_execution_logger.log(TIME_EXECUTION, "t_ii tiempo_inicio_inicio Deterministic --- %.4f seconds ---" % (time() - start_time))

# Tiempo entre inicios de avalanchas DGD Random Redistribution
start_time = time()
t_ii_dgdrr = tiempo_inicio_inicio(lim_a_dgdrr)
time_execution_logger.log(TIME_EXECUTION, "t_ii tiempo_inicio_inicio DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))
# %%

# E
logging.info("E")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_st)
distribution_plot(xe, ye, fit_ye, "E_{st}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_det)
distribution_plot(xe, ye, fit_ye, "E_{det}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_dgdrr)
distribution_plot(xe, ye, fit_ye, "E_{dgdrr}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# P
logging.info("P")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(P_st)
distribution_plot(xe, ye, fit_ye, "P_{st}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "P distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(P_det)
distribution_plot(xe, ye, fit_ye, "P_{det}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "P distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(P_dgdrr)
distribution_plot(xe, ye, fit_ye, "P_{dgdrr}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "P distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# T
logging.info("T")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(T_st)
distribution_plot(xe, ye, fit_ye, "T_{st}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "T distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(T_det)
distribution_plot(xe, ye, fit_ye, "T_{det}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "T distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(T_dgdrr)
distribution_plot(xe, ye, fit_ye, "T_{dgdrr}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "T distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# t_P
logging.info("t_P")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_P_st)
distribution_plot(xe, ye, fit_ye, "t_{P_{st}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_P distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_P_det)
distribution_plot(xe, ye, fit_ye, "t_{P_{det}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_P distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# TODO: Try non-log scale for t_P_dgdrr distribution
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_P_dgdrr)
distribution_plot(xe, ye, fit_ye, "t_{P_{dgdrr}}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_P distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# t_fi
logging.info("t_fi")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_fi_st, semilog=True)
distribution_plot(xe, ye, fit_ye, "t_{{fi}_{st}}",
                  scale="semilog", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_fi distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_fi_det, semilog=True)
distribution_plot(xe, ye, fit_ye, "t_{{fi}_{det}}",
                  scale="semilog", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_fi distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_fi_dgdrr, semilog=True)
distribution_plot(xe, ye, fit_ye, "t_{{fi}_{dgdrr}}",
                    scale="semilog", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_fi distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# t_ii
logging.info("t_ii")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ii_st)
distribution_plot(xe, ye, fit_ye, "t_{{ii}_{st}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ii distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ii_det)
distribution_plot(xe, ye, fit_ye, "t_{{ii}_{det}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ii distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

# TODO: Try non-log scale for t_P_dgdrr distribution
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ii_dgdrr)
distribution_plot(xe, ye, fit_ye, "t_{{ii}_{dgdrr}}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ii distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# t_ac
logging.info("t_ac")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_st)
distribution_plot(xe, ye, fit_ye, "t_{{ac}_{st}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ac distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_det)
distribution_plot(xe, ye, fit_ye, "t_{{ac}_{det}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ac distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_dgdrr)
distribution_plot(xe, ye, fit_ye, "t_{{ac}_{dgdrr}}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ac distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# t_rel
logging.info("t_rel")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_st)
distribution_plot(xe, ye, fit_ye, "t_{{rel}_{st}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_rel distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_det)
distribution_plot(xe, ye, fit_ye, "t_{{rel}_{det}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_rel distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_dgdrr)
distribution_plot(xe, ye, fit_ye, "t_{{rel}_{dgdrr}}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_rel distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# E_ac
logging.info("E_ac")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_ac_st)
distribution_plot(xe, ye, fit_ye, "E_{{ac}_{st}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E_ac distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_ac_det)
distribution_plot(xe, ye, fit_ye, "E_{{ac}_{det}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E_ac distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_ac_dgdrr)
distribution_plot(xe, ye, fit_ye, "E_{{ac}_{dgdrr}}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E_ac distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# E_rel
logging.info("E_rel")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_rel_st)
distribution_plot(xe, ye, fit_ye, "E_{{rel}_{st}}",
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E_rel distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_rel_det)
distribution_plot(xe[1:], ye[1:], fit_ye[1:], "E_{{rel}_{det}}", 
                  scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E_rel distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(E_rel_dgdrr)
distribution_plot(xe, ye, fit_ye, "E_{{rel}_{dgdrr}}",
                    scale="log", fit=fit, save=True)
time_execution_logger.log(TIME_EXECUTION, "E_rel distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# t_ac_pesado
logging.info("t_ac_pesado")
start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_pesado_st, normal=True)
distribution_plot(xe[1:], ye[1:], fit_ye, "t_{{{ac}_{{w}_{st}}}}", fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ac_pesado distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_pesado_det, normal=True)
distribution_plot(xe[1:], ye[1:], fit_ye, "t_{{{ac}_{{w}_{det}}}}", fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ac_pesado distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_ac_pesado_dgdrr, normal=True)
distribution_plot(xe[1:], ye[1:], fit_ye, "t_{{{ac}_{{w}_{dgdrr}}}}", fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_ac_pesado distribution_to_plot DGD Random Redistribution --- %.4f seconds ---" % (time() - start_time))

# t_rel_pesado
logging.info("t_rel_pesado")
start_time = time()
distribution_plot(xe[1:], ye[1:], fit_ye, "t_{{{rel}_{{w}_{st}}}}", fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_rel_pesado distribution_to_plot Standard: --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_pesado_det, normal=True)
distribution_plot(xe[:-1], ye[:-1], fit_ye, "t_{{{rel}_{{w}_{det}}}}", fit=False, save=True)
time_execution_logger.log(TIME_EXECUTION, "t_rel_pesado distribution_to_plot Deterministic --- %.4f seconds ---" % (time() - start_time))

start_time = time()
xe, ye, fit_ye = distribution_to_plot(t_rel_pesado_dgdrr, normal=True)
distribution_plot(xe[1:], ye[1:], fit_ye, "t_{{{rel}_{{w}_{dgdrr}}}}", fit=False, save=True)

logging.info("Number of avalanches analyzed Standard: " + str(len(T_st)))
logging.info("Number of avalanches analyzed Deterministic: " + str(len(T_det)))
logging.info("Number of avalanches analyzed DGD Random Redistribution: " + str(len(T_dgdrr)))