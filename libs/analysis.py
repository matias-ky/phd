# -*- coding: utf-8 -*-

# Here I will put the analysis functions

import numpy as np
from numba import jit, njit


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
        print(f"The power-law exponent is {me} and error is {me_error}")

    fit_ye = []
    if not normal:
        for i in range(len(xe)):
            fit_ye.append(np.exp(be)*(xe[i]**me))

    return xe, ye, fit_ye

@jit(nopython=True)
def duraciones(lim_a):
    T = []
    for i in range(len(lim_a)-1):
        if lim_a[i + 1] - lim_a[i] == 1:
            continue
        else:
            T.append(lim_a[i + 1] - lim_a[i])
    return T

# @jit(nopython=True)
# @njit
def energia_picos(lim_a, e_soc):
    E = []
    P = []
    tes = []
    t_ac = []
    E_ac = []
    t_rel = []
    E_rel = []
    t_ac_pesado = []
    t_rel_pesado = []
    i = 0
    while i < len(lim_a)-1:
        if lim_a[i + 1] - lim_a[i] == 1:
            i = i + 1
            continue
        else:
            m = e_soc[lim_a[i]+1:lim_a[i+1]]
            E.append(np.sum(m))
            P.append(np.max(m))
            for j in range(0, len(m)):
                if m[j-1] == np.max(m):
                    t_ac.append(j)
                    t_ac_pesado.append(j/float(len(m)))
                    t_rel.append(len(m)-j)
                    t_rel_pesado.append((len(m)-j)/float(len(m)))
                    E_ac.append(np.sum(m[0:j]))
                    E_rel.append(np.sum(m[j:-1]))
                    aux = j + lim_a[i] + 1
                    break
            tes.append(aux)
            i = i + 1
            if i % 50000 == 0:
                print(i)
    return E, P, tes, t_ac, E_ac, t_rel, E_rel, t_ac_pesado, t_rel_pesado