# -*- coding: utf-8 -*-

# Here I will put plot functions and other useful functions.

import matplotlib.pyplot as plt
import numpy as np


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

    plt.style.use("ggplot")  # Define the plot style
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

    if scale == "log":
        plt.xlim(
            (np.min(x_values)/1.5, x_values[np.max(np.nonzero(y_values))]*10))
        plt.ylim(
            (y_values[np.max(np.nonzero(y_values))]/5, np.max(y_values)*10))

        # Add manual borders
        plt.axvline(np.min(x_values)/1.45, color="black")  # Left
        plt.axvline(x_values[np.max(np.nonzero(y_values))]
                    * 9.7, color="black")  # Right
        plt.axhline(y_values[np.max(np.nonzero(y_values))] /
                    4.9, color="black")  # Bottom
        plt.axhline(np.max(y_values)*8.8, color="black")  # Top

    plt.xlabel(r"$"+variable+"$", size=20)
    plt.ylabel(r"PDF($"+variable+"$)", size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(c='black', alpha=0.3)
    plt.rcParams.update({'font.size': 13})
    plt.tight_layout()

    if save:
        plt.savefig("plot_"+variable+"_fit.pdf")
        plt.savefig("plot_"+variable+"_fit.png", dpi=300)

    plt.show()
