
"""Nonlinear Stokes-Mueller polarimetry (NSMP).

This module contains NSMP fitting routines.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interpn

from lklib.string import get_human_val_str, arr_summary_str
from lklib.util import handle_general_exception, unwrap_angle, ask_yesno
from lklib.fileread import check_file_exists

from lcmicro.proc import load_pipo
from lcmicro.polarimetry.report import plot_pipo_fit_img, plot_pipo_fit_1point
from lcmicro.polarimetry.nsmp_sim import simulate_pipo
from lcmicro.polarimetry.fitdata import get_parmap


def get_default_fitdata_filename():
    """Get default fitdata file name."""
    return "fitdata.npy"


def get_default_fitaccel_filename():
    """Get default fit accelerator filte name."""
    return 'fit_accel_c6v.npy'


def print_fit_result_1point(fit_result, fit_model=None):
    """Print PIPO fit results for a single point."""
    print("=== Fit results ===")

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
    delta_str = get_human_val_str(
        unwrap_angle(delta, period=delta_period)/np.pi*180, num_sig_fig=3)
    err_str = get_human_val_str(err)

    print('Fit model: {:s}'.format(fit_model))
    print('Parameters:')
    print('\tA = {:s}'.format(ampl_str))
    print('\tδ = {:s}°'.format(delta_str))
    if zzz is not None:
        print('\tzzz = {:s}'.format(zzz_str))
    print('RMS residual error: {:s}\n'.format(err_str))


def print_fit_result_img(fitdata):
    """Print PIPO fit results for an image."""
    print("=== Fit results ===")

    data_type = fitdata['data_type']
    print("Data type: " + data_type)

    fit_model = fitdata['model']

    fit_mask = fitdata['mask']
    num_row, num_col = np.shape(fit_mask)
    num_px = num_row*num_col
    num_pts_to_fit = fitdata['num_pts_to_fit']
    if data_type == 'img':
        print("Image size: {:d}x{:d}".format(num_row, num_col))
        print("Fit threshold: {:d} c.".format(fitdata['mask_thr']))
        num_thr = np.sum(fit_mask)
        print("Number of px. above threshold: {:d}, {:.0f}%".format(
            num_thr, num_thr/num_px*100))

        if num_pts_to_fit < num_thr:
            print("Number of px. fitted: {:d}, {:.0f}%".format(
                num_pts_to_fit, num_pts_to_fit/num_px*100))

    zzz = None
    if fit_model in ['zcq', 'c6v']:
        ampl = get_parmap(fitdata, 'x', 0)
        delta = get_parmap(fitdata, 'x', 1)

    if fit_model == 'c6v':
        zzz = get_parmap(fitdata, 'x', 2)

    err = get_parmap(fitdata, 'err', 0)

    print('Fit model: {:s}'.format(fit_model))
    print('Parameters:')
    print('\tA: ' + arr_summary_str(ampl, suppress_suffix='m'))
    print('\tδ (°): ' + arr_summary_str(delta/np.pi*180, num_sig_fig=3))
    if zzz is not None:
        print('\tzzz: ' + arr_summary_str(
            ampl, num_sig_fig=3, suppress_suffix='m'))

    print('RMS residual error: ' + arr_summary_str(err))
    print('\n')


def pipo_fitfun(
        par, xdata, data, fit_model='zcq',
        fit_accel=None,
        print_progress=False, plot_progress=False, vlvl=1):
    """Fit optimization function for PIPO."""
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
        fit_data = ampl*simulate_pipo(
            symmetry_str=symmetry_str, delta=delta, zzz=zzz)

    if np.any(np.isnan(fit_data)):
        print("NaN in fit model")

    res = data - fit_data
    err = np.mean(np.sqrt(res**2))

    if plot_progress or print_progress or vlvl >= 2:
        ampl_str = get_human_val_str(ampl, suppress_suffix='m')
        zzz_str = get_human_val_str(zzz, num_sig_fig=3, suppress_suffix='m')
        delta_str = get_human_val_str(
            unwrap_angle(delta)/np.pi*180, num_sig_fig=3)
        err_str = get_human_val_str(err)

    if plot_progress:
        plot_pipo_fit_1point(
            data, fit_model=fit_model, fit_par=par, new_fig=False)
        plt.draw()
        plt.pause(0.001)

    if vlvl >= 2:
        msg = "A = {:s}, δ = {:s}°".format(ampl_str, delta_str)
        if fit_model == 'c6v':
            msg += ", R = {:s}".format(zzz_str)
        msg += ", err = {:s}".format(err_str)
        if fit_accel is not None:
            msg += ", mapind = {:d}".format(mapind)
        print(msg)
    elif vlvl >= 1:
        print('.', end='', flush=True)

    return err


def get_fit_accel(fit_model='c6v'):
    """Get fit accelerator."""
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
    except Exception:
        maps = np.ndarray([num_delta*num_zzz, 8, 8])
        mapind = 0
        for delta_ind, delta in enumerate(delta_arr):
            for zzz_ind, zzz in enumerate(zzz_arr):
                print("Map {:d} of {:d}".format(mapind + 1, len(maps)))
                mapind = mapinds[delta_ind, zzz_ind]

                maps[mapind, :, :] = simulate_pipo(
                    symmetry_str=fit_model, delta=delta, zzz=zzz)

        print("Saving fit accel data to '{:s}'".format(accel_file_name))
        np.save(accel_file_name, maps)

    fit_accel = {}
    fit_accel['pargrid'] = (delta_arr, zzz_arr)
    fit_accel['mapinds'] = mapinds
    fit_accel['maps'] = maps

    diff_step = [0.1, 3*delta_step, 3*zzz_step]

    return fit_accel, diff_step


def fit_pipo_1point(
        pipo_arr=None, file_name=None, fit_model='zcq',
        plot_progress=False, print_results=True, plot_fig=True,
        use_fit_accel=False, fit_accel=None, diff_step=None, vlvl=1,
        **kwargs):
    """Fit PIPO using single-point data."""
    if pipo_arr is None:
        pipo_arr = load_pipo(file_name, binsz=None)

    if fit_model not in ['zcq', 'c6v']:
        raise Exception("Unsupported fittig model")

    if use_fit_accel and fit_model != 'c6v':
        raise Exception("Fit acceleration only supported for c6v")

    max_ampl = np.max(pipo_arr)
    if fit_model == 'zcq':
        guess_par = [max_ampl, 0]
        bounds = [
            [0,             -np.pi],
            [max_ampl*1.5,  np.pi]]
    elif fit_model == 'c6v':
        guess_par = [max_ampl, 0, 1.5]
        bounds = [
            [0,             0,      1],
            [max_ampl*1.5,  np.pi,  2]]

    if plot_progress:
        plt.figure(figsize=[12, 5])

    tstart = time.time()

    if use_fit_accel:
        if fit_accel is None:
            fit_accel, diff_step = get_fit_accel()

    fit_cfg = {
        'fit_model': fit_model,
        'plot_progress': plot_progress,
        'print_progress': False,
        'fit_accel': fit_accel,
        'vlvl': vlvl
    }

    if vlvl >= 1:
        print("Fitting data", end='')

    fit_result = least_squares(
        pipo_fitfun, guess_par, args=(0, pipo_arr), diff_step=diff_step,
        bounds=bounds, kwargs=fit_cfg)

    if vlvl >= 1:
        print("Done")

    if vlvl >= 2:
        num_eval = fit_result.nfev + fit_result.njev
        print("Number of fit evaluations: {:d}".format(num_eval))
        print("Fitting time: {:.2f} s".format(time.time() - tstart))

    if plot_progress:
        plt.close()

    if print_results:
        print_fit_result_1point(fit_model=fit_model, fit_result=fit_result)

    if plot_fig:
        plot_pipo_fit_1point(
            pipo_arr, fit_par=fit_result.x, **fit_cfg, **kwargs)

    return fit_result


def fit_pipo_img(
        pipo_arr, fit_model='zcq', plot_progress=False,
        use_fit_accel=False, max_fit_pts=None,
        print_results=True, plot_results=True,
        vlvl=1,
        **kwargs):
    """Fit PIPO using image data."""
    t_start = time.time()

    if fit_model not in ['zcq', 'c6v']:
        raise Exception("Unsupported fittig model")

    if use_fit_accel and fit_model != 'c6v':
        raise Exception("Fit acceleration only supported for c6v")

    num_row, num_col = np.shape(pipo_arr)[0:2]

    if use_fit_accel:
        fit_accel, diff_step = get_fit_accel()
    else:
        fit_accel = diff_step = None

    auto_fit_mask = kwargs.get('auto_fit_mask', True)
    if auto_fit_mask:
        cnt_thr = 50
        print("Calculating fit mask, count threshold is {:d}".format(cnt_thr))
        fit_mask = np.ndarray([num_row, num_col], dtype=bool)
        fit_mask.fill(False)
        for ind_row in range(num_row):
            for ind_col in range(num_col):
                if np.sum(pipo_arr[ind_row, ind_col, :, :]) >= cnt_thr:
                    fit_mask[ind_row, ind_col] = True

    num_fit_pts = np.sum(fit_mask)
    num_pts_to_fit = num_fit_pts

    if max_fit_pts and num_fit_pts > max_fit_pts:
        print("Dataset contains {:d} fittable points, but max_fit_points is "
              "{:d}, truncating".format(num_fit_pts, max_fit_pts))
        num_pts_to_fit = max_fit_pts

    ind_fit = 0
    fit_result = []
    t_last_prog_update = time.time()
    t_fit_start = time.time()
    for ind_row in range(num_row):
        for ind_col in range(num_col):
            if not fit_mask[ind_row, ind_col]:
                continue
            if ind_fit == num_pts_to_fit:
                break

            t_now = time.time()
            if t_now - t_last_prog_update > 0.5:
                t_last_prog_update = t_now
                elapsed_time = t_now - t_fit_start
                fit_rate = ind_fit/elapsed_time
                est_time_remaining = (num_pts_to_fit - ind_fit)/fit_rate
                msg = "Fitting point {:d} of {:d}. Elapsed time: {:.0f} s, " \
                    "remaining {:.0f} s".format(
                        ind_fit+1, num_pts_to_fit, elapsed_time, est_time_remaining)

                print(msg)

            pipo_arr1 = pipo_arr[ind_row, ind_col, :, :]
            try:
                fit_result1 = fit_pipo_1point(
                    pipo_arr1, fit_model=fit_model,
                    plot_progress=plot_progress, use_fit_accel=use_fit_accel,
                    print_results=False, plot_fig=False,
                    fit_accel=fit_accel, diff_step=diff_step,
                    vlvl=0,
                    **kwargs)
            except Exception:
                print("\nFitting failed")
                if vlvl >= 2:
                    handle_general_exception(Exception)
                fit_result1 = None

            fit_result.append(fit_result1)
            ind_fit += 1

    elapsed_time = time.time() - t_fit_start
    fit_rate = num_pts_to_fit/elapsed_time
    print("Fitting completed in {:.1f} s at {:.1f} pts/s rate".format(elapsed_time, fit_rate))

    fitdata = {}
    fitdata['data_type'] = 'img'  # 'img', '1point'
    fitdata['model'] = 'c6v'
    fitdata['mask'] = fit_mask
    fitdata['mask_thr'] = 50
    fitdata['num_pts_to_fit'] = num_pts_to_fit
    fitdata['result'] = fit_result

    print("Saving intermediate fit data to 'fitdata.npy'...")
    np.save('fitdata.npy', [fitdata])

    if print_results:
        print_fit_result_img(fitdata)

    if plot_results:
        plot_pipo_fit_img(fitdata, pipo_arr=pipo_arr)

    return fitdata


def fit_pipo(
        pipo_arr=None, file_name=None,
        binsz=None, cropsz=None,
        show_input=False, **kwargs):
    """Fit a PIPO model to data.

    Fit a PIPO model to a single PIPO map or a 4D array for an image where each
    point contains a per-pixel PIPO array. The data can be provided as an array
    or a file name containing the array.

    Pixels can be binned for faster or more accurate fitting, setting binsz to
    'all' will sum all image data into a single PIPO array.

    The image can be cropped by setting cropsz to:
        [from_row, to_row, from_col, to_col], in pixels

    Cropping will also speed up fitting and can be combined with binsz='all' to
    yield a single PIPO map fit for a given area.

    A mask will be applied to suppress fitting of pixels that have low singal,
    by defaut the threshold is <50 counts total over all polarization states.

    By setting show_input to True a total count image will be shown before
    fitting, and the execution will be paused until the image is closed. This
    is useful to verify that the correct data is loaded before committing to
    a long fit.

    Args:
        pipo_arr (ndarray): PIPO data to fit the model to
        file_name (str): file name of the PIPO dataset
        binsz (int/'all'): number of pixels to bin for fitting
        cropsz (4-tuple): image are to use for fitting
        show_input (bool): show the input total count image before fitting and
            pause

    Returns:
        fitdata dict containing fit parameters and results
    """
    fitdata_filename = get_default_fitdata_filename()
    if check_file_exists(fitdata_filename) and not ask_yesno(
            "A " + fitdata_filename + " file containing the fit results "
            "already exists and will be overwritten. Do you want to continue?",
            default='no'):
        print("Terminating fitter")
        return None

    fitaccel_filename = get_default_fitaccel_filename()
    if kwargs.get('use_fit_accel', False) \
            and not check_file_exists(fitaccel_filename):
        print("Fit accelertion is enabled, but " + fitaccel_filename +
              " was not found. Acceleration file generation will be "
              "implemented in the future, until then this file is required.")
        return None

    if show_input:
        pipo_arr = load_pipo(file_name, binsz=None, cropsz=cropsz)
        print("Showing input image, close the figure window to continue...")
        plt.imshow(np.sum(np.sum(pipo_arr, 2), 2))
        plt.show()
        pipo_arr = None

    if pipo_arr is None:
        pipo_arr = load_pipo(file_name, binsz=binsz, cropsz=cropsz)

    if len(np.shape(pipo_arr)) == 4:
        return fit_pipo_img(pipo_arr, **kwargs)
    else:
        return fit_pipo_1point(pipo_arr, **kwargs)
