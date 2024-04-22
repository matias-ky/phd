# -*- coding: utf-8 -*-

# Here I will put plot functions and other useful functions.

import matplotlib.pyplot as plt
import numpy as np
import logging


def distribution_plot(x_values, y_values, fitted_values, variable, scale=None, fit=True, save=False):
    """
    Generate a distribution plot.

    Parameters
    ----------
    x_values : numpy.ndarray
        X values for the distribution.
    y_values : numpy.ndarray
        Y values for the distribution.
    fitted_values : numpy.ndarray
        Fitted Y values.
    variable : str
        Variable name.
    scale : str, optional
        Scaling type ('log', 'semilog'), by default None.
    fit : bool, optional
        Include the fit in the plot, by default True.
    save : bool, optional
        Save the plot as a PDF and PNG, by default False (not Save).

    Returns
    -------
    None

    Notes
    -----
    This function generates a distribution plot for given data.

    Examples
    --------
    >>> x_data = np.linspace(0, 10, 100)
    >>> y_data = np.sin(x_data)
    >>> fitted_data = np.cos(x_data)
    >>> distribution_plot(x_data, y_data, fitted_data, variable='x', scale='log', fit=True, save=False)
    """
    # plt.figure(figsize=(8, 6))
    plt.figure()

    # plt.style.use("ggplot")  # Define the plot style
    plt.rcParams['axes.facecolor'] = 'w'

    plt.plot(x_values, y_values, "o", mfc="none",
             label="Distribution", markersize=9)
    plt.legend(shadow=True, edgecolor="black")

    if fit:
        plt.plot(x_values, fitted_values, "--", label="Fit", linewidth=3)
        plt.legend(shadow=True, edgecolor="black")

    if scale == "log":
        plt.yscale("log")
        plt.xscale("log")
    if scale == "semilog":
        plt.yscale("log")

    # if scale == "log":
    #     plt.xlim(
    #         (np.min(x_values)/1.5, x_values[np.max(np.nonzero(y_values))]*10))
    #     plt.ylim(
    #         (y_values[np.max(np.nonzero(y_values))]/5, np.max(y_values)*10))

    plt.xlabel(r"$"+variable+"$", size=20)
    plt.ylabel(r"PDF($"+variable+"$)", size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(c='black', alpha=0.3)
    plt.rcParams.update({'font.size': 13})
    plt.tight_layout()

    if save:
        plt.savefig("plots/pdf/plot_"+variable+"_fit.pdf")
        plt.savefig("plots/png/plot_"+variable+"_fit.png", dpi=300)
    else:
        plt.show()

    plt.close()

# Define a custom logging level
TIME_EXECUTION = 15  # You can choose any value between logging.DEBUG (10) and logging.INFO (20)

# Add the custom logging level to the logging module
logging.addLevelName(TIME_EXECUTION, "TIME_EXECUTION")

# Create a custom logger for time execution
time_execution_logger = logging.getLogger("time_execution")
# time_execution_logger.setLevel(TIME_EXECUTION)
time_execution_handler = logging.StreamHandler()
time_execution_formatter = logging.Formatter("%(message)s")
time_execution_handler.setFormatter(time_execution_formatter)
time_execution_logger.addHandler(time_execution_handler)
