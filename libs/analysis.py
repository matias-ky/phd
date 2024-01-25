# -*- coding: utf-8 -*-

# Here I will put the analysis functions

import numpy as np


def distribution_to_plot(E, normal=False):
    """
    Generate distribution data for plotting.

    Parameters
    ----------
    E : array_like
        Input array of energy values.
    normal : bool, optional
        If True, a normal distribution is used; if False, a power-law distribution is assumed.
        Default is False.

    Returns
    -------
    tuple
        Tuple containing the x-axis values, y-axis values, and fitted y-axis values for plotting.

    Notes
    -----
    This function generates distribution data for plotting based on the input array of energy values (`E`).
    It can assume either a power-law or normal distribution depending on the value of the `normal` parameter.

    Parameters
    ----------
    E : array_like
        Input array of energy values.
    normal : bool, optional
        If True, a normal distribution is used; if False, a power-law distribution is assumed.
        Default is False.

    Returns
    -------
    tuple
        Tuple containing the x-axis values, y-axis values, and fitted y-axis values for plotting.

    Examples
    --------
    >>> energy_values = np.random.rand(100)
    >>> x_vals, y_vals, fit_vals = distribution_to_plot(energy_values, normal=False)
    """
    hist_E = []
    if not normal:
        exponent = np.log(np.max(E)) / np.log(10)
        hist_E = np.histogram(
            E, bins=10 ** np.linspace(0, np.log10(10**exponent), 50), density=True)
    else:
        hist_E = np.histogram(E, bins=50, density=True)
    xe = hist_E[1]
    xe = xe[:-1]  # Removes the last value from a list or array
    ye = hist_E[0]

    if not normal:
        xe_log = []
        ye_log = []
        for i in range(len(ye)):
            if ye[i] == 0:
                continue
            else:
                xe_log.append(np.log(xe[i]))
                ye_log.append(np.log(ye[i]))

        me, be = np.polyfit(xe_log[2:20], ye_log[2:20], 1, cov=True)[0]
        _, cov = np.polyfit(xe_log[2:20], ye_log[2:20], 1, cov=True)
        me_error, be_error = np.sqrt(np.diag(cov))
        fit = np.poly1d([me, be])

    fit_ye = []
    if not normal:
        for i in range(len(xe)):
            fit_ye.append(np.exp(be)*(xe[i]**me))

    return xe, ye, fit_ye
