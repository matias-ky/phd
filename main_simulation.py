# -*- coding: utf-8 -*-
"""
Main Simulation and Analysis Script
Refactored version of test.py with clean, modular, and reusable functions.
"""

import os
import re
import logging
import datetime
import time as sys_time
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# Import project-specific simulation and analysis libraries
from libs.functions import *
from libs.areas import *
from libs.analysis import *
from libs.utils import *

# Define simulation parameters
SEED = 7
GRID_SIZE = 128
ZC = 1
ITERATIONS = 1000
FIT_DISTRIBUTIONS = True

# Active models selection
ACTIVE_MODELS = ["standard", "deterministic"]

# Models configuration
MODELS_CONFIG = {
    "standard": {
        "name": "Standard",
        "label": "st",
        "marker": "o",
        "linestyle": "--",
    },
    "deterministic": {
        "name": "Deterministic",
        "label": "det",
        "marker": "s",
        "linestyle": "-",
    },
    "random_redistribution": {
        "name": "Random Redistribution",
        "label": "dgdrr",
        "marker": "^",
        "linestyle": "-.",
    },
    "stochastic_threshold": {
        "name": "Stochastic Threshold",
        "label": "dgdst",
        "marker": "d",
        "linestyle": ":",
    }
}

# Base directory for the current run
RUN_DIR = None


def get_next_run_dir(base_dir="runs"):
    """Return the next available run directory for today's date."""
    date_folder = datetime.date.today().strftime("%d_%m_%Y")
    date_dir = os.path.join(base_dir, date_folder)
    os.makedirs(date_dir, exist_ok=True)

    existing_indices = []
    for entry in os.listdir(date_dir):
        path = os.path.join(date_dir, entry)
        if os.path.isdir(path):
            match = re.match(r"^run_(\d{2})$", entry)
            if match:
                existing_indices.append(int(match.group(1)))

    next_index = max(existing_indices) + 1 if existing_indices else 0
    run_dir = os.path.join(date_dir, f"run_{next_index:02d}")
    return run_dir


class SimulationResult:
    """Container for all simulation outputs and processed metrics."""
    def __init__(self, outputs):
        (
            self.e_lib,
            self.e_tot,
            self.B,
            self.avalanche_covered_areas,
            self.number_of_nodes_at_peak,
            self.number_of_nodes_per_avalanche,
            self.number_of_clusters,
            self.cluster_sizes,
            self.fractal_index,
            self.fractal_indexes_at_peak
        ) = outputs

        # Process derived avalanche metrics
        # Find indices where e_lib is 0 (avalanche boundaries)
        self.lim_a = np.where(np.array(self.e_lib) == 0)[0]
        
        # Durations
        self.T = duraciones(self.lim_a)
        
        # Energy peaks analysis
        (
            self.E,
            self.P,
            self.tes,
            self.t_ac,
            self.E_ac,
            self.t_rel,
            self.E_rel,
            self.t_ac_pesado,
            self.t_rel_pesado
        ) = energia_picos(self.lim_a, self.e_lib)
        
        # Time differences
        self.t_P = tiempo_entre_picos(self.tes)
        self.t_fi = tiempo_fin_inicio(self.lim_a)
        self.t_ii = tiempo_inicio_inicio(self.lim_a)


def run_all_simulations(B_initial, N, Zc, iterations):
    """Runs all model simulations according to ACTIVE_MODELS and returns a dict of SimulationResult."""
    results = {}
    
    # 1. Run Standard Model
    if "standard" in ACTIVE_MODELS:
        logging.info("Running Standard Model simulation...")
        t0 = sys_time.time()
        outputs_st = simulacion_completa(B_initial, N, Zc, iterations, "standard")
        logging.info("Standard Model completed in %.4f seconds. Steps: %d", sys_time.time() - t0, len(outputs_st[0]))
        results["standard"] = SimulationResult(outputs_st)
    else:
        logging.info("Skipping Standard Model simulation (not in active models).")
    
    # 2. Run Deterministic Model
    if "deterministic" in ACTIVE_MODELS:
        logging.info("Running Deterministic Model simulation...")
        t0 = sys_time.time()
        outputs_det = simulacion_completa(B_initial, N, Zc, iterations, "deterministic")
        logging.info("Deterministic Model completed in %.4f seconds. Steps: %d", sys_time.time() - t0, len(outputs_det[0]))
        results["deterministic"] = SimulationResult(outputs_det)
    else:
        logging.info("Skipping Deterministic Model simulation (not in active models).")
    
    # 3. Run Random Redistribution Model
    if "random_redistribution" in ACTIVE_MODELS:
        logging.info("Running Random Redistribution Model simulation...")
        t0 = sys_time.time()
        outputs_dgdrr = simulacion_completa(B_initial, N, Zc, iterations, "random_redistribution")
        logging.info("Random Redistribution Model completed in %.4f seconds. Steps: %d", sys_time.time() - t0, len(outputs_dgdrr[0]))
        results["random_redistribution"] = SimulationResult(outputs_dgdrr)
    else:
        logging.info("Skipping Random Redistribution Model simulation (not in active models).")
    
    # 4. Stochastic Threshold
    if "stochastic_threshold" in ACTIVE_MODELS:
        if "random_redistribution" in results:
            logging.info("Setting Stochastic Threshold model results (identical to Random Redistribution)...")
            results["stochastic_threshold"] = results["random_redistribution"]
        else:
            logging.warning("Stochastic Threshold is active but Random Redistribution was skipped. Skipping Stochastic Threshold.")
    else:
        logging.info("Skipping Stochastic Threshold model (not in active models).")
    
    return results


def format_title_with_suffix(title, suffix):
    """Formats title with key suffix nested correctly for LaTeX parsing in matplotlib."""
    if title.endswith("}"):
        return f"{title[:-1]}_{{{suffix}}}}}"
    else:
        return f"{title}_{{{suffix}}}"


def plot_and_log_distribution(data, key_suffix, title, normal=False, semilog=False, scale=None, fit=False):
    """Helper to compute, plot, and log a single distribution."""
    start_time = sys_time.time()
    xe, ye, fit_ye = distribution_to_plot(data, normal=normal, semilog=semilog)
    formatted_title = format_title_with_suffix(title, key_suffix)
    distribution_plot(
        xe,
        ye,
        fit_ye,
        formatted_title,
        scale=scale,
        fit=fit,
        save=True,
        base_dir=os.path.join(RUN_DIR, "plots")
    )
    time_execution_logger.log(
        TIME_EXECUTION, 
        f"Distribution {title} for {key_suffix} plotted in --- {sys_time.time() - start_time:.4f} seconds ---"
    )
    return xe, ye, fit_ye


def plot_comparison(datasets, xlabel, ylabel, filename, scale="linear"):
    """Generates a comparison plot containing multiple models."""
    plt.figure()
    for label, config in MODELS_CONFIG.items():
        if label not in datasets:
            continue
        xe, ye = datasets[label]
        plt.plot(
            xe, ye, 
            linestyle=config["linestyle"], 
            marker=config["marker"], 
            label=config["name"]
        )
    
    if scale == "log":
        plt.yscale("log")
        plt.xscale("log")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plots inside the run directory
    pdf_dir = os.path.join(RUN_DIR, "plots", "pdf")
    png_dir = os.path.join(RUN_DIR, "plots", "png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    plt.savefig(os.path.join(pdf_dir, f"{filename}.pdf"))
    plt.savefig(os.path.join(png_dir, f"{filename}.png"), dpi=300)
    plt.close()


def main():
    global RUN_DIR
    
    # Generate run directory using today's date and sequential numbering
    RUN_DIR = get_next_run_dir()
    os.makedirs(os.path.join(RUN_DIR, "plots", "pdf"), exist_ok=True)
    os.makedirs(os.path.join(RUN_DIR, "plots", "png"), exist_ok=True)
    
    # Setup logging to file inside the run directory
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    log_file = os.path.join(RUN_DIR, "execution.log")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setLevel(logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure time execution logger to write to the log file and console as well
    time_execution_logger.handlers = []
    time_execution_logger.addHandler(file_handler)
    time_execution_logger.addHandler(console_handler)
    time_execution_logger.setLevel(TIME_EXECUTION)
    
    logging.info("Starting run folder %s", RUN_DIR)
    logging.info("Active models: %s", ", ".join(ACTIVE_MODELS))
    logging.info("Skipped models: %s", ", ".join([k for k in MODELS_CONFIG.keys() if k not in ACTIVE_MODELS]))
    
    # Set consistency seed
    set_seed(SEED)
    
    # Load initial grid
    current_directory = os.getcwd()
    grid_path = os.path.join(current_directory, f"initial_grids/B_N{GRID_SIZE}_Zc1.npz")
    logging.info("Loading initial grid from %s", grid_path)
    data = np.load(grid_path)
    B_initial = data["B"].astype(np.float32)
    
    # Run active model simulations
    results = run_all_simulations(B_initial, GRID_SIZE, ZC, ITERATIONS)
    
    # If no models were run, exit early
    if not results:
        logging.info("No active models executed. Exiting.")
        return
        
    # --- 1. Fractal Index (D) Analysis ---
    logging.info("Analyzing Fractal Index (D)...")
    d_data = {}
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        res = results[key]
        xe, ye, _ = plot_and_log_distribution(res.fractal_index, model["label"], "D", normal=True)
        d_data[key] = (xe, ye)
        
        logging.info("Mean Fractal Index %s: %f", model["label"], np.mean(res.fractal_index))
        logging.info("Max Fractal Index %s: %f", model["label"], np.max(res.fractal_index))
        logging.info("--> Most Frequent Fractal Index %s (Mode): %f", model["label"], xe[np.argmax(ye)])
        
    if d_data:
        plot_comparison(d_data, r"$D$", r"PDF($D$)", "plot_D")
    
    # --- 2. Fractal Index at Peak (D*) Analysis ---
    logging.info("Analyzing Fractal Index at Peak (D*)...")
    d_star_data = {}
    for key in ["standard", "deterministic", "random_redistribution"]:
        if key not in results:
            continue
        model = MODELS_CONFIG[key]
        res = results[key]
        xe, ye, _ = plot_and_log_distribution(res.fractal_indexes_at_peak, model["label"], "D^{*}", normal=True)
        d_star_data[key] = (xe, ye)
        
        logging.info("Mean Fractal Index at peak %s: %f", model["label"], np.mean(res.fractal_indexes_at_peak))
        logging.info("Max Fractal Index at peak %s: %f", model["label"], np.max(res.fractal_indexes_at_peak))
        logging.info("--> Most Frequent Fractal Index at peak %s (Mode): %f", model["label"], xe[np.argmax(ye)])
        
    if d_star_data:
        plot_comparison(d_star_data, r"$D^{*}$", r"PDF($D^{*}$)", "plot_D^{*}")
    
    # --- 3. Number of Clusters Distribution (NC) ---
    logging.info("Analyzing Number of Clusters (NC) Distribution...")
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        plot_and_log_distribution(results[key].number_of_clusters, model["label"], "NC", normal=True)
        
    # --- 4. Cluster Sizes Distribution (CS) ---
    logging.info("Analyzing Cluster Sizes (CS) Distribution...")
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        plot_and_log_distribution(results[key].cluster_sizes, model["label"], "CS", scale="log", fit=FIT_DISTRIBUTIONS)
        
    # --- 5. Avalanche Covered Areas (A) ---
    logging.info("Analyzing Avalanche Covered Areas (A)...")
    a_data = {}
    fits_a = {}
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        res = results[key]
        xe, ye, fit_ye = plot_and_log_distribution(res.avalanche_covered_areas, model["label"], "A", scale="log", fit=FIT_DISTRIBUTIONS)
        a_data[key] = (xe, ye)
        fits_a[key] = fit_ye
        
    # Replicate custom plot_A with specific fits
    if "deterministic" in results or "random_redistribution" in results:
        plt.figure()
        if "deterministic" in results:
            plt.plot(a_data["deterministic"][0], a_data["deterministic"][1], marker='s', label='Deterministic')
            plt.plot(a_data["deterministic"][0], fits_a["deterministic"], linestyle='--', label='Deterministic Fit')
        if "random_redistribution" in results:
            plt.plot(a_data["random_redistribution"][0], a_data["random_redistribution"][1], marker='^', label='Random Redistribution')
            plt.plot(a_data["random_redistribution"][0], fits_a["random_redistribution"], linestyle='-.', label='Random Redistribution Fit')
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel(r"$A$")
        plt.ylabel(r"PDF($A$)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "plots", "pdf", "plot_A.pdf"))
        plt.savefig(os.path.join(RUN_DIR, "plots", "png", "plot_A.png"), dpi=300)
        plt.close()
    
    # --- 6. Number of Nodes at Peak (A*) ---
    logging.info("Analyzing Number of Nodes at Peak (A*)...")
    a_star_data = {}
    fits_a_star = {}
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        res = results[key]
        xe, ye, fit_ye = plot_and_log_distribution(res.number_of_nodes_at_peak, model["label"], "A^{*}", scale="log", fit=FIT_DISTRIBUTIONS)
        a_star_data[key] = (xe, ye)
        fits_a_star[key] = fit_ye
        
    # Replicate custom plot_A* with specific fits
    if "deterministic" in results or "random_redistribution" in results:
        plt.figure()
        if "deterministic" in results:
            plt.plot(a_star_data["deterministic"][0], a_star_data["deterministic"][1], marker='s', label='Deterministic')
            plt.plot(a_star_data["deterministic"][0], fits_a_star["deterministic"], linestyle='--', label='Deterministic Fit')
        if "random_redistribution" in results:
            plt.plot(a_star_data["random_redistribution"][0], a_star_data["random_redistribution"][1], marker='^', label='Random Redistribution')
            plt.plot(a_star_data["random_redistribution"][0], fits_a_star["random_redistribution"], linestyle='-.', label='Random Redistribution Fit')
        plt.yscale("log")
        plt.xscale("log")
        plt.xlabel(r"$A^{*}$")
        plt.ylabel(r"PDF($A^{*}$)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RUN_DIR, "plots", "pdf", "plot_A^{*}.pdf"))
        plt.savefig(os.path.join(RUN_DIR, "plots", "png", "plot_A^{*}.png"), dpi=300)
        plt.close()
    
    # --- 7. Distribution Plotting Loop for Derived Metrics ---
    metrics_to_plot = [
        # (field_name, title_prefix, scale_type, normal_flag, semilog_flag, fit_flag)
        ("E", "E", "log", False, False, FIT_DISTRIBUTIONS),
        ("P", "P", "log", False, False, FIT_DISTRIBUTIONS),
        ("T", "T", "log", False, False, FIT_DISTRIBUTIONS),
        ("t_P", "t_{P}", "log", False, False, FIT_DISTRIBUTIONS),
        ("t_fi", "t_{{fi}}", "semilog", False, True, FIT_DISTRIBUTIONS),
        ("t_ii", "t_{{ii}}", "log", False, False, FIT_DISTRIBUTIONS),
        ("t_ac", "t_{{ac}}", "log", False, False, FIT_DISTRIBUTIONS),
        ("t_rel", "t_{{rel}}", "log", False, False, FIT_DISTRIBUTIONS),
        ("E_ac", "E_{{ac}}", "log", False, False, FIT_DISTRIBUTIONS),
    ]
    
    for field_name, title_prefix, scale, normal, semilog, fit in metrics_to_plot:
        logging.info("Analyzing %s...", field_name)
        for key, model in MODELS_CONFIG.items():
            if key not in results:
                continue
            data = getattr(results[key], field_name)
            plot_and_log_distribution(data, model["label"], title_prefix, normal=normal, semilog=semilog, scale=scale, fit=fit)
            
    # E_rel (requires special slicing for deterministic model in original code)
    logging.info("Analyzing E_rel...")
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        res = results[key]
        xe, ye, fit_ye = distribution_to_plot(res.E_rel)
        formatted_title = format_title_with_suffix("E_{{rel}}", model['label'])
        if key == "deterministic":
            # Apply identical slicing as original code
            distribution_plot(
                xe[1:],
                ye[1:],
                fit_ye[1:],
                formatted_title,
                scale="log",
                fit=FIT_DISTRIBUTIONS,
                save=True,
                base_dir=os.path.join(RUN_DIR, "plots")
            )
        else:
            distribution_plot(
                xe,
                ye,
                fit_ye,
                formatted_title,
                scale="log",
                fit=FIT_DISTRIBUTIONS,
                save=True,
                base_dir=os.path.join(RUN_DIR, "plots")
            )
            
    # t_ac_pesado & t_rel_pesado (both use normal=True, fit=False)
    logging.info("Analyzing t_ac_pesado & t_rel_pesado...")
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        res = results[key]
        
        # t_ac_pesado (slices xe[1:], ye[1:])
        xe, ye, fit_ye = distribution_to_plot(res.t_ac_pesado, normal=True)
        formatted_title_ac = format_title_with_suffix("t_{{{ac}_{{w}}}}", model['label'])
        distribution_plot(
            xe[1:],
            ye[1:],
            fit_ye,
            formatted_title_ac,
            fit=False,
            save=True,
            base_dir=os.path.join(RUN_DIR, "plots")
        )
        
        # t_rel_pesado
        xe_r, ye_r, fit_ye_r = distribution_to_plot(res.t_rel_pesado, normal=True)
        formatted_title_rel = format_title_with_suffix("t_{{{rel}_{{w}}}}", model['label'])
        if key == "deterministic":
            # Slices differently for deterministic in original code
            distribution_plot(
                xe_r[:-1],
                ye_r[:-1],
                fit_ye_r,
                formatted_title_rel,
                fit=False,
                save=True,
                base_dir=os.path.join(RUN_DIR, "plots")
            )
        else:
            distribution_plot(
                xe_r[1:],
                ye_r[1:],
                fit_ye_r,
                formatted_title_rel,
                fit=False,
                save=True,
                base_dir=os.path.join(RUN_DIR, "plots")
            )
            
    # --- Summary Logging ---
    for key, model in MODELS_CONFIG.items():
        if key not in results:
            continue
        logging.info("Number of avalanches analyzed %s: %d", model["name"], len(results[key].T))


if __name__ == "__main__":
    main()
