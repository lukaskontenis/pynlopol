
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
from scipy.interpolate import interpn
import time

from lklib.plot import export_figure
from lklib.string import get_human_val_str

from lcmicro.proc import load_pipo
from lcmicro.polarimetry.nsmp_sim import simulate_pipo
from lcmicro.polarimetry.report import plot_pipo

def unwrap_angle(angle, period=np.pi, plus_minus_range=True):
    """Unwrap angle to value to [-period/2, period/2] range."""
    if plus_minus_range:
        return angle - np.round(angle/period)*period
    else:
        return angle - np.floor(angle/period)*period


def plot_pipo_fit(
        data, fit_model=None, fit_par=None, fit_data=None,
        show_fig=False, new_fig=True,
        export_fig=False, fig_file_name=None, **kwargs):
    if fit_model not in ['zcq', 'c6v']:
        raise(Exception("Unsupported fitting model"))
    if fit_par is None:
        raise(Exception("No fit parameters given"))
    if (fit_model == 'zcq' and len(fit_par) != 2) or \
        (fit_model == 'c6v' and len(fit_par) != 3):
        raise(Exception("Incorrect number of fit parameters"))

    zzz = None
    if fit_model == 'zcq':
        ampl = fit_par[0]
        delta = fit_par[1]
        delta_period = 60/180*np.pi
        symmetry_str = 'd3'
    elif fit_model == 'c6v':
        ampl = fit_par[0]
        delta = fit_par[1]
        zzz = fit_par[2]
        delta_period = 180/180*np.pi
        symmetry_str = 'c6v'

    if fit_data is None:
        fit_data = ampl*simulate_pipo(symmetry_str=symmetry_str, delta=delta, zzz=zzz)

    res = data - fit_data
    err = np.mean(np.sqrt(res**2))

    ampl_str = get_human_val_str(ampl, suppress_suffix='m')
    zzz_str = get_human_val_str(zzz, num_sig_fig=3, suppress_suffix='m')
    delta_str = get_human_val_str(unwrap_angle(delta, period=delta_period)/np.pi*180, num_sig_fig=3, suppress_suffix='m')
    err_str = get_human_val_str(err)

    if new_fig:
        plt.figure(figsize=[12, 5])
    else:
        plt.clf()

    plt.subplot(1, 3, 1)
    plt.imshow(data, cmap='gray')
    plt.title('Data')
    plt.subplot(1, 3, 2)
    plt.imshow(fit_data, cmap='gray')
    if fit_model == 'zcq':
        plt.title('Fit model ''{:s}''\nA = {:s}, δ = {:s}°'.format(fit_model, ampl_str, delta_str))
    else:
        plt.title('Fit model ''{:s}''\nA = {:s}, R = {:s}, δ = {:s}°'.format(fit_model, ampl_str, zzz_str, delta_str))
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

    zzz = None
    if fit_model == 'zcq':
        ampl = fit_result.x[0]
        delta = fit_result.x[1]
        delta_period = 60/180*np.pi
    elif fit_model == 'c6v':
        ampl = fit_result.x[0]
        delta = fit_result.x[1]
        zzz = fit_result.x[2]
        delta_period = 180/180*np.pi

    err = fit_result.fun[0]

    ampl_str = get_human_val_str(ampl, suppress_suffix='m')
    zzz_str = get_human_val_str(zzz, num_sig_fig=3, suppress_suffix='m')
    delta_str = get_human_val_str(unwrap_angle(delta, period=delta_period)/np.pi*180, num_sig_fig=3)
    err_str = get_human_val_str(err)

    print('Fit model: {:s}'.format(fit_model))
    print('Parameters:')
    print('\tA = {:s}'.format(ampl_str))
    print('\tδ = {:s}°'.format(delta_str))
    if zzz is not None:
        print('\tzzz = {:s}°'.format(zzz_str))
    print('RMS residual error: {:s}\n'.format(err_str))


def pipo_fitfun(
        par, xdata, data, fit_model='zcq',
        fit_accel=None,
        print_progress=False, plot_progress=False):
    symmetry_str = fit_model

    zzz = None
    if fit_model in ['zcq', 'c6v']:
        ampl = par[0]
        delta = par[1]
        delta_period = 60/180*np.pi

    if fit_model == 'c6v':
        zzz = par[2]
        delta_period = 180/180*np.pi

    delta = unwrap_angle(delta, delta_period, plus_minus_range=False)

    if fit_accel is not None:
        try:
            mapind = int(interpn(
                fit_accel['pargrid'], fit_accel['mapinds'], [delta, zzz],
                method='nearest', bounds_error=True))
        except ValueError:
            print('val error')
        fit_data = ampl*fit_accel['maps'][mapind]
    else:
        fit_data = ampl*simulate_pipo(symmetry_str=symmetry_str, delta=delta, zzz=zzz)

    if np.any(np.isnan(fit_data)):
        print("NaN in fit model")

    res = data - fit_data
    err = np.mean(np.sqrt(res**2))

    if plot_progress or print_progress:
        ampl_str = get_human_val_str(ampl, suppress_suffix='m')
        zzz_str = get_human_val_str(zzz, num_sig_fig=3, suppress_suffix='m')
        delta_str = get_human_val_str(unwrap_angle(delta)/np.pi*180, num_sig_fig=3)
        err_str = get_human_val_str(err)

    if plot_progress:
        plot_pipo_fit(data, fit_model=fit_model, fit_par=par, new_fig=False)
        plt.draw()
        plt.pause(0.001)

    if print_progress:
        msg = "A = {:s}, δ = {:s}°".format(ampl_str, delta_str)
        if fit_model == 'c6v':
            msg += ", R = {:s}".format(zzz_str)
        msg += ", err = {:s}".format(err_str)
        if fit_accel is not None:
            msg += ", mapind = {:d}".format(mapind)
        print(msg)
    else:
        print('.', end='', flush=True)

    return err


def fit_pipo(
        pipo_arr=None, file_name=None, fit_model='zcq', plot_progress=False,
        use_fit_accel=False,
        **kwargs):

    if pipo_arr is None:
        pipo_arr = load_pipo(file_name)

    if fit_model not in ['zcq', 'c6v']:
        raise Exception("Unsupported fittig model")

    if use_fit_accel and fit_model != 'c6v':
        raise Exception("Fit acceleration only supported for c6v")

    if fit_model == 'zcq':
        guess_par = [np.max(pipo_arr), 0]
    elif fit_model == 'c6v':
        guess_par = [np.max(pipo_arr), 0, 1.5]

    if plot_progress:
        plt.figure(figsize=[12, 5])

    tstart = time.time()

    fit_accel = None
    diff_step = None
    if use_fit_accel:
        delta_min = 0/180*np.pi
        delta_max = 180/180*np.pi
        num_delta = 25
        delta_step = (delta_max - delta_min)/num_delta
        delta_arr = np.linspace(delta_min, delta_max, num_delta)

        zzz_min = 1
        zzz_max = 2
        num_zzz = 25
        zzz_step = (zzz_max - zzz_min)/num_zzz
        zzz_arr = np.linspace(zzz_min, zzz_max, num_zzz)

        mapinds = np.reshape(np.arange(0, num_delta*num_zzz), [num_delta, num_zzz])
        accel_file_name = 'fit_accel_c6v.npy'
        try:
            maps = np.load(accel_file_name)
            print("Loaded fit accel data from '{:s}'".format(accel_file_name))
        except:
            maps = np.ndarray([num_delta*num_zzz, 8, 8])
            mapind = 0
            for delta_ind, delta in enumerate(delta_arr):
                for zzz_ind, zzz in enumerate(zzz_arr):
                    print("Map {:d} of {:d}".format(mapind + 1, len(maps)))
                    mapind = mapinds[delta_ind, zzz_ind]
                    maps[mapind, :, :] = simulate_pipo(symmetry_str=fit_model, delta=delta, zzz=zzz)
            print("Saving fit accel data to '{:s}'".format(accel_file_name))
            np.save(accel_file_name, maps)

        fit_accel = {}
        fit_accel['pargrid'] = (delta_arr, zzz_arr)
        fit_accel['mapinds'] = mapinds
        fit_accel['maps'] = maps

        if fit_model == 'zcq':
            diff_step = [0.1, 2*delta_step]
        elif fit_model == 'c6v':
            diff_step = [0.1, 2*delta_step, zzz_step]

    fit_cfg = {
        'fit_model': fit_model,
        'plot_progress': plot_progress,
        'print_progress': False,
        'fit_accel': fit_accel,
    }

    print("Fitting data", end='')

    fit_result = least_squares(pipo_fitfun, guess_par, args=(0, pipo_arr), diff_step=diff_step, kwargs=fit_cfg)

    print("Done")
    num_eval = fit_result.nfev + fit_result.njev
    print("Number of fit evaluations: {:d}".format(num_eval))
    print("Fitting time: {:.2f} s".format(time.time() - tstart))

    if plot_progress:
        plt.close()

    print_fit_par(fit_model=fit_model, fit_result=fit_result)

    plot_pipo_fit(pipo_arr, fit_par=fit_result.x, **fit_cfg, **kwargs)
