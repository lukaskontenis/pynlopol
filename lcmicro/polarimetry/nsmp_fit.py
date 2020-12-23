
"""Nonlinear Stokes-Mueller polarimetry (NSMP).

This module contains NSMP fitting routines.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from lklib.plot import export_figure
from lklib.string import get_human_val_str

from lcmicro.proc import load_pipo
from lcmicro.polarimetry.nsmp_sim import simulate_pipo


def unwrap_angle(angle, period=np.pi):
    """Unwrap angle to value to [-period/2, period/2] range."""
    return angle - np.round(angle/period)*period


def plot_pipo_fit(
        data, fit_model=None, fit_par=None, fit_data=None,
        show_fig=False, new_fig=True,
        export_fig=False, fig_file_name=None, **kwargs):
    if fit_model != 'zcq':
        raise(Exception("Only zcq fit model is currently supported"))
    if fit_par is None:
        raise(Exception("No fit parameters given"))
    if len(fit_par) != 2:
        raise(Exception("Incorrect number of fit parameters"))

    if fit_model == 'zcq':
        ampl = fit_par[0]
        delta = fit_par[1]
        delta_period = 60/180*np.pi
        symmetry_str = 'd3'

    if fit_data is None:
        fit_data = ampl*simulate_pipo(symmetry_str=symmetry_str, delta=delta)

    res = data - fit_data
    err = np.mean(np.sqrt(res**2))

    ampl_str = get_human_val_str(ampl)
    delta_str = get_human_val_str(unwrap_angle(delta, period=delta_period)/np.pi*180, num_sig_fig=3)
    err_str = get_human_val_str(err)

    if new_fig:
        plt.figure(figsize=[10, 5])
    else:
        plt.clf()

    plt.subplot(1, 3, 1)
    plt.imshow(data, cmap='gray')
    plt.title('Data')
    plt.subplot(1, 3, 2)
    plt.imshow(fit_data, cmap='gray')
    plt.title('Fit model ''{:s}'', A = {:s}, δ = {:s}°'.format(fit_model, ampl_str, delta_str))
    plt.subplot(1, 3, 3)
    plt.imshow(res, cmap='coolwarm')
    plt.title('Residuals, err = {:s}'.format(err_str))

    if export_fig:
        print("Exporting figure...")
        if fig_file_name is None:
            fig_file_name = 'pipo_fit.png'

        export_figure(fig_file_name, resize=False)

    if show_fig:
        plt.show()


def print_fit_par(fit_model=None, fit_result=None):
    print("=== Fit result ===")
    if fit_model == 'zcq':
        ampl = fit_result.x[0]
        delta = fit_result.x[1]
        delta_period = 60/180*np.pi

    err = fit_result.fun[0]

    ampl_str = get_human_val_str(ampl)
    delta_str = get_human_val_str(unwrap_angle(delta, period=delta_period)/np.pi*180, num_sig_fig=3)
    err_str = get_human_val_str(err)

    print('Fit model: {:s}'.format(fit_model))
    print('Parameters:')
    print('\tA = {:s}'.format(ampl_str))
    print('\tδ = {:s}°'.format(delta_str))
    print('RMS residual error: {:s}\n'.format(err_str))


def pipo_fitfun(par, xdata, data, fit_model='zcq', print_progress=False, plot_progress=False):
    ampl = par[0]
    delta = par[1]
    symmetry_str = fit_model
    fit_data = ampl*simulate_pipo(symmetry_str=symmetry_str, delta=delta)

    if np.any(np.isnan(fit_data)):
        print("NaN in fit model")

    res = data - fit_data
    err = np.mean(np.sqrt(res**2))

    if plot_progress or print_progress:
        ampl_str = get_human_val_str(ampl)
        delta_str = get_human_val_str(unwrap_angle(delta)/np.pi*180, num_sig_fig=3)
        err_str = get_human_val_str(err)

    if plot_progress:
        plot_pipo_fit(data, fit_model=fit_model, fit_par=par, new_fig=False)
        plt.draw()
        plt.pause(0.001)

    if print_progress:
        print("A = {:s}, δ = {:s}°, err = {:s}".format(ampl_str, delta_str, err_str))
    else:
        print('.', end='')

    return err


def fit_pipo(pipo_arr=None, file_name=None, fit_model='zcq', plot_progress=False):
    if pipo_arr is None:
        pipo_arr = load_pipo(file_name)

    guess_par = [np.max(pipo_arr), 0]

    fit_cfg = {
        'fit_model': fit_model,
        'plot_progress': plot_progress
    }

    plot_progress = fit_cfg.get('plot_progress', False)

    if plot_progress:
        plt.figure(figsize=[12, 5])

    print("Fitting data", end='')
    fit_result = least_squares(pipo_fitfun, guess_par, args=(0, pipo_arr), kwargs=fit_cfg)
    print("Done")

    if plot_progress:
        plt.close()

    plot_pipo_fit(pipo_arr, fit_par=fit_result.x, show_fig=True, export_fig=True, **fit_cfg)

    print_fit_par(fit_model='zcq', fit_result=fit_result)
