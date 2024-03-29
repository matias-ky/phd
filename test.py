# -*- coding: utf-8 -*-

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
e_lib_st, e_tot_st, B_st, areas_cubiertas_st, cantidad_de_nodos_en_avalanchas_st, lista_de_clusters_st = simulacion_completa(B_N62, N, Zc, iterations)
print("LU&H Standard: --- %s seconds ---" % (time() - start_time))
print(len(e_lib_st))