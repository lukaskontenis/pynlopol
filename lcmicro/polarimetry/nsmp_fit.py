
"""Nonlinear Stokes-Mueller polarimetry (NSMP).

This module contains NSMP fitting routines.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2021 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interpn

from lklib.string import get_human_val_str
from lklib.util import handle_general_exception, unwrap_angle, ask_yesno, \
    find_closest
from lklib.fileread import check_file_exists

from lcmicro.proc import load_pipo
from lcmicro.polarimetry.report import plot_pipo_fit_img, plot_pipo_fit_1point
from lcmicro.polarimetry.nsmp_sim import simulate_pipo
from lcmicro.polarimetry.fitdata import FitData
from lcmicro.polarimetry.imgfitdata import ImgFitData, ImgFitConfig


def get_default_fitdata_filename():
    """Get default fitdata file name."""
    return "fitdata.npy"


def get_default_fitaccel_filename():
    """Get default fit accelerator filte name."""
    return 'fit_accel_c6v.npy'


def pipo_fitfun(
        par, xdata, data, fit_model='c6v',
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


def pipo_c6v_fun(par):
    """C6v PIPO equation from from Golaraei2020, Eq. (1).

    The C6 equation is used with the xyz parameter is set to 0.
    """
    return pipo_c6_fun(np.append(par, 0))


def pipo_c6_fun(par):
    """C6 PIPO equation from Golaraei2020, Eq. (1).

    The equation is used in PIPONATOR for fitting with ampl, background and
    xyz. Background is not used here.
    """
    # TODO: fix this
    psg_states = np.arange(0.0, 180.0, 22.5)/180*np.pi
    psa_states = np.arange(0.0, 180.0, 22.5)/180*np.pi
    fit_data = np.ndarray([len(psg_states), len(psa_states)])

    ampl = par[0]
    delta = par[1]
    zzz = par[2]
    xyz = par[3]

    # TODO: implement optional background fitting
    bckg = 0
    for ind_psg, psg in enumerate(psg_states):
        for ind_psa, psa in enumerate(psa_states):
            fit_data[ind_psa, ind_psg] = ampl*(
                np.sin(psa-delta)*np.sin(2*(psg-delta)) +
                np.cos(psa-delta)*np.sin(psg-delta)**2 +
                zzz*np.cos(psa-delta)*np.cos(psg-delta)**2 +
                2*xyz*np.cos(psg-delta)*np.sin(psg-psa))**2 + bckg

    return fit_data


def pipo_c6v_fitfun(
        par, xdata, data, fit_model='c6v', fitfun_name='c6_ag', fit_accel=None,
        plot_progress=False, vlvl=1):
    """Fit optimization function for C6 PIPO."""
    ampl = par[0]
    delta = par[1]
    zzz = par[2]
    delta_period = 180/180*np.pi

    delta = unwrap_angle(delta, delta_period, plus_minus_range=False)

    if fitfun_name == 'c6_ag':
        fit_data = pipo_c6v_fun(par)
    elif fitfun_name == 'fullsim':
        fit_data = ampl*simulate_pipo(symmetry_str='c6v', delta=delta, zzz=zzz)
    elif fitfun_name == 'accelmap':
        try:
            mapind = int(interpn(
                fit_accel['pargrid'], fit_accel['mapinds'], [delta, zzz],
                method='nearest', bounds_error=True))
        except ValueError:
            print('val error')
        fit_data = ampl*fit_accel['maps'][mapind]

    if np.any(np.isnan(fit_data)):
        print("NaN in fit model")

    res = data - fit_data
    err = np.mean(np.sqrt(res**2))

    if plot_progress:
        plot_pipo_fit_1point(
            data, fit_model='c6v', fit_par=par, fit_data=fit_data,
            new_fig=False)
        plt.draw()
        plt.pause(0.001)

    if plot_progress:
        print("\nIter: " + print_par(['ampl', 'delta', 'zzz'], par) +
              ", rmse={:.3f}".format(err))
    elif vlvl >= 1:
        print('.', end='', flush=True)

    return err


def get_fit_accel(fit_model='c6v'):
    """Get fit accelerator."""
    delta_min = 0/180*np.pi
    delta_max = 180/180*np.pi
    num_delta = 50
    delta_step = (delta_max - delta_min)/num_delta
    delta_arr = np.linspace(delta_min, delta_max, num_delta)

    if fit_model == 'zcq':
        mapinds = np.reshape(np.arange(0, num_delta), [num_delta])
        accel_file_name = 'fit_accel_d3.npy'
        try:
            maps = np.load(accel_file_name)
            print("Loaded fit accel data from '{:s}'".format(accel_file_name))
        except Exception:
            maps = np.ndarray([num_delta, 8, 8])
            mapind = 0
            for delta_ind, delta in enumerate(delta_arr):
                print("Map {:d} of {:d}".format(mapind + 1, len(maps)))
                mapind = mapinds[delta_ind]

                maps[mapind, :, :] = simulate_pipo(
                    symmetry_str=fit_model, delta=delta)

            print("Saving fit accel data to '{:s}'".format(accel_file_name))
            np.save(accel_file_name, maps)

        fit_accel = {}
        fit_accel['pargrid'] = (delta_arr)
        fit_accel['mapinds'] = mapinds
        fit_accel['maps'] = maps

        diff_step = [0.1, 2*delta_step]

    elif fit_model == 'c6v':
        zzz_min = 1
        zzz_max = 2
        num_zzz = 10
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

        diff_step = [0.1, 2*delta_step, 2*zzz_step]

    return fit_accel, diff_step


def verify_fit_accel(fit_accel=None):
    """Verify fit accelerator array values."""
    delta_arr = fit_accel['pargrid'][0]
    zzz_arr = fit_accel['pargrid'][1]
    err = np.ndarray([len(delta_arr), len(zzz_arr)])
    for ind_zzz, zzz in enumerate(zzz_arr):
        for ind_delta, delta in enumerate(delta_arr):
            accel_arr = fit_accel['maps'][ind_delta*len(delta_arr) + ind_zzz]
            true_arr = simulate_pipo(symmetry_str='c6v', delta=delta, zzz=zzz)
            err[ind_delta, ind_zzz] = np.sqrt(np.mean((accel_arr - true_arr)**2))

    plt.imshow(err)
    plt.show()


def fit_pipo_1point(
        pipo_arr=None, file_name=None, fit_model='zcq', fitfun_name=None,
        map_guess=True, use_fit_accel=False, fit_accel=None, diff_step=None,
        plot_progress=False, print_results=True, plot_fig=True,
        true_par=None, vlvl=1,
        **kwargs):
    """Fit PIPO using single-point data."""
    t_start = time.time()

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
        guess_par = [max_ampl, 0, 1]
        bounds = [
            [0,             0,      1],
            [max_ampl*100,  np.pi,  2]]

    if plot_progress:
        plt.figure(figsize=[12, 5])

    if map_guess or use_fit_accel:
        if fit_accel is None:
            fit_accel, diff_step = get_fit_accel(fit_model)

        test_fit_accel = False
        if test_fit_accel:
            verify_fit_accel(fit_accel)

    fit_cfg = {
        'fit_model': fit_model,
        'plot_progress': plot_progress,
        'print_progress': False,
        'fit_accel': fit_accel,
        'vlvl': vlvl
    }

    if vlvl >= 1:
        print("Fitting data", end='')
    elif vlvl >= 2:
        print("Fitting data")

    # Use a sparse map covering all expected parameter ranges to find a good
    # guess value
    if map_guess:
        accel_maps = fit_accel['maps']
        num_maps = accel_maps.shape[0]
        err = np.ndarray([num_maps])

        # Calculate total squared errors between the given PIPO dataset and all
        # pre-generated fit PIPO datasets. Sice we're only looking for the
        # closest match, the formula for the error isn't important.
        for ind in range(num_maps):
            accel_map = accel_maps[ind, :, :]
            err[ind] = np.sum((pipo_arr/np.max(pipo_arr) - accel_map/np.max(accel_map))**2)

        pargrid = fit_accel['pargrid']

        if fit_model == 'zcq':
            # Find the index of the fit dataset that is most similar to the
            # given datasaet
            guess_inds = np.unravel_index(np.argmin(err), pargrid.shape)

            # The guess parameters are then the ones used to generated that
            # dataset
            guess_par = [max_ampl, pargrid[guess_inds[0]]]
            diff_step = [0.001, 0.01/180*np.pi]

        elif fit_model == 'c6v':
            guess_inds = np.unravel_index(
                np.argmin(err), [len(pargrid[0]), len(pargrid[1])])
            guess_par = [max_ampl,
                         pargrid[0][guess_inds[0]],
                         pargrid[1][guess_inds[1]]]
            diff_step = [0.001, 0.01/180*np.pi, 0.001]

        if vlvl >= 2 and true_par is not None:
            for ind, guess_par1 in enumerate(guess_par):
                if ind == 0:
                    # Skip amplitude
                    continue

                if find_closest(pargrid[ind-1], true_par[ind]) != guess_inds[ind-1]:
                    print("Parameter map guess could have been better")

        fit_cfg = {
            'fit_model': fit_model,
            'plot_progress': plot_progress,
            'vlvl': vlvl
        }

    if fit_model == 'c6v':
        if fitfun_name is None:
            fitfun_name = 'c6v_ag'
    elif fit_model == 'zcq':
        if fitfun_name is None:
            fitfun_name = 'nsmpsim'

    if fitfun_name == 'c6v_ag':
        fitfun = pipo_c6v_fitfun
        guess_pipo_arr = pipo_c6v_fun(guess_par)
        # plot_pipo_fit_1point(pipo_arr, fit_data=guess_pipo_arr)
        guess_par[0] = guess_par[0]*np.mean(np.max(pipo_arr)/np.max(guess_pipo_arr))
    elif fitfun_name == 'nsmpsim':
        fitfun = pipo_fitfun
    else:
        raise Exception("Fit function '" + fitfun_name + "' not defined for '"
                        + fit_model + "' model")

    if False and vlvl >= 2:
        true_pipo_arr = pipo_c6v_fun(true_par)
        true_pipo_arr *= np.max(pipo_arr)/np.max(true_pipo_arr)
        plot_pipo_fit_1point(pipo_arr, fit_data=true_pipo_arr)
        plt.show()

    guess_par_bound_check_lo = np.array(bounds[0]) > np.array(guess_par)
    guess_par_bound_check_hi = np.array(bounds[1]) < np.array(guess_par)
    if guess_par_bound_check_lo.any():
        if vlvl >= 1:
            print("Guess beyond lower bound, setting to lower band")

        guess_par[guess_par_bound_check_lo] = bounds[0][guess_par_bound_check_lo]

    if guess_par_bound_check_hi.any():
        if vlvl >= 1:
            print("Guess beyond lower bound, setting to upper band")

        guess_par[guess_par_bound_check_hi] = bounds[1][guess_par_bound_check_hi]

    if plot_progress:
        print("\nGuess parameters: "
              + print_par(['ampl', 'delta', 'zzz'], guess_par))

    fit_result = least_squares(
        fitfun, guess_par, args=(0, pipo_arr), diff_step=diff_step,
        bounds=bounds, kwargs=fit_cfg)

    fit_duration = time.time() - t_start
    fitdata = FitData(fit_result)
    fitdata.set_fit_model(fit_model)
    fitdata.set_fitfun_name(fitfun_name)
    fitdata.set_fit_duration(fit_duration)

    if vlvl >= 1:
        print("Done")

    if plot_progress:
        plt.close()

    if print_results:
        fitdata.print()

    if plot_fig:
        fit_data = None
        if fitfun_name == 'c6v_ag':
            fit_data = pipo_c6v_fun(fit_result.x)

        plot_pipo_fit_1point(pipo_arr, fit_data=fit_data, fit_par=fit_result.x,
                             **fit_cfg, **kwargs)

    return fitdata


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

    fcfg = ImgFitConfig()
    fcfg.set_fit_model(fit_model)
    fcfg.set_max_fit_pts(max_fit_pts)

    num_row, num_col = np.shape(pipo_arr)[0:2]
    fcfg.set_sz([num_row, num_col])

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

        fcfg.set_mask(fit_mask)
        fcfg.set_mask_thr(cnt_thr)

    num_fit_pts = np.sum(fit_mask)
    num_pts_to_fit = num_fit_pts

    if max_fit_pts and num_fit_pts > max_fit_pts:
        print("Dataset contains {:d} fittable points, but max_fit_points is "
              "{:d}, truncating".format(num_fit_pts, max_fit_pts))
        num_pts_to_fit = max_fit_pts

    num_pts = np.prod(pipo_arr.shape[0:2])
    if num_fit_pts < num_pts:
        print("Fitting {:d} of {:d} points, that's {:.1f}%".format(num_fit_pts, num_pts, num_fit_pts/num_pts*100))
    else:
        print("Fitting every point")

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
                msg = "Fitting point {:d} of {:d}. Elapsed time: {:s}, " \
                    "remaining {:s}".format(
                        ind_fit+1, num_pts_to_fit, get_human_val_str(elapsed_time, is_time=True), get_human_val_str(est_time_remaining, is_time=True))

                print(msg)

            pipo_arr1 = pipo_arr[ind_row, ind_col, :, :]
            try:
                if vlvl >= 2:
                    vlvl1 = vlvl
                else:
                    vlvl1 = 0
                fit_result1 = fit_pipo_1point(
                    pipo_arr1, fit_model=fit_model,
                    plot_progress=plot_progress, use_fit_accel=use_fit_accel,
                    print_results=False, plot_fig=False,
                    fit_accel=fit_accel, diff_step=diff_step,
                    vlvl=vlvl1,
                    **kwargs)
            except Exception:
                print("\nFitting failed")
                if vlvl >= 1:
                    handle_general_exception(Exception)
                fit_result1 = None

            if not fit_result1.is_fit_success():
                print("Fit at point ({:d},{:d}) did not converge".format(ind_row, ind_col))

            fit_result.append(fit_result1)
            ind_fit += 1

    elapsed_time = time.time() - t_fit_start
    fit_rate = num_pts_to_fit/elapsed_time
    print("Fitting completed in {:.1f} s at {:.1f} pts/s rate".format(elapsed_time, fit_rate))

    imgfitdata = ImgFitData(fit_result)
    imgfitdata.cfg = fcfg
    imgfitdata.duration = elapsed_time

    print("Saving intermediate fit data to 'fitdata.npy'...")
    np.save('fitdata.npy', [imgfitdata])

    if print_results:
        imgfitdata.print()

    if plot_results:
        plot_pipo_fit_img(imgfitdata, pipo_arr=pipo_arr)

    return imgfitdata


def fit_pipo(
        pipo_arr=None, file_name=None,
        binsz=None, cropsz=None,
        ask_before_overwrite=True, show_input=False, **kwargs):
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
    if ask_before_overwrite and check_file_exists(fitdata_filename) and \
        not ask_yesno(
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
        #return None

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
