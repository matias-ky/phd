# -*- coding: utf-8 -*-

# Import necessary libraries and functions
from libs.functions import *
import numpy as np
from time import time
import os

# Load data for N = 62
current_directory = os.getcwd()
data_N62 = np.load(current_directory + "/B_final_N62_Zc1.npz")
B_N62 = data_N62["B"]
B_N62 = B_N62.astype(np.float32)

# Set simulation parameters
N = 62
Zc = 1
iterations = 1

# Compile Numba jit function for cellular automaton
e_lib, e_tot, B, grid_list, area_list = cellular_automaton_lu_ham_standard(
    B_N62, N, Zc, iterations)

# Run simulation with increased iterations
iterations = 10000

start_time = time()
e_lib, e_tot, B, grid_list, area_list = cellular_automaton_lu_ham_standard(
    B_N62, N, Zc, iterations)
print("--- %s seconds ---" % (time() - start_time))
