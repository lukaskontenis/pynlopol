
"""Nonlinear Stokes-Mueller polarimetry (NSMP) fit routines.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import interpn
import scipy.ndimage as ndimg

from lkcom.util import cap_in_range, handle_general_exception, unwrap_angle, \
    ask_yesno, find_closest
from lkcom.string import get_human_val_str
from lkcom.dataio import check_file_exists

from pynlomic.proc import load_pipo
from pynlopol.report import plot_pipo_fit_img, plot_pipo_fit_1point
from pynlopol.nsmp_sim import simulate_pipo
from pynlopol.fitdata import FitData
from pynlopol.imgfitdata import ImgFitData, ImgFitConfig


def get_default_fitdata_filename():
    """Get default fitdata file name."""
    return "fitdata.npy"


def get_default_fitaccel_filename():
    """Get default fit accelerator filte name."""
    return 'fit_accel_c6v.npy'


def get_default_fitfun_name(fit_model):
    """Get the default fit function name."""
    if fit_model == 'shg_c6v':
        return 'c6v_ag'
    elif fit_model == 'zcq':
        return 'nsmpsim'
    else:
        print("WARNING: no default fit function defined for fit model '{:}'".format(fit_model))


def pipo_fitfun(
        par, xdata, data, fit_model='c6v',
        fit_accel=None,
        print_progress=False, plot_progress=False, vlvl=1):
    """Fit optimization function for PIPO."""
    sim_par = {
        'symmetry_str': fit_model.split('_')[-1],
        'nlorder': {'shg': 2, 'thg': 3}[fit_model.split('_')[0]],
        'ampl': par[0],
        'delta': par[1],
    }

    # 'pset_name': 'pipo_8x8',
    # 'output_type': '1point'

    zzz = None
    if fit_model == 'zcq':
        delta_period = 60/180*np.pi
    elif fit_model == 'shg_c6v':
        sim_par['zzz'] = par[2]
        delta_period = 180/180*np.pi
    elif fit_model == 'thg_c6v':
        sim_par['zzzz'] = par[2]
        sim_par['xxxx'] = par[3]
        delta_period = 180/180*np.pi

    sim_par['delta'] = unwrap_angle(sim_par['delta'], delta_period, plus_minus_range=False)

    if fit_accel is not None:
        try:
            if fit_model == 'shg_c6v':
                mapind = int(interpn(
                    fit_accel['pargrid'], fit_accel['mapinds'], [sim_par['delta'], sim_par['zzz']],
                    method='nearest', bounds_error=True))
        except ValueError:
            print('val error')
        fit_data = sim_par['ampl']*fit_accel['maps'][mapind]
    else:
        fit_data = sim_par['ampl']*simulate_pipo(**sim_par)

    if np.any(np.isnan(fit_data)):
        print("NaN in fit model")

    res = data - fit_data
    err = np.mean(np.sqrt(res**2))

    if plot_progress or print_progress or vlvl >= 2:
        ampl_str = get_human_val_str(sim_par['ampl'], suppress_suffix='m')
        if fit_model == 'shg_c6v':
            zzz_str = get_human_val_str(sim_par['zzz'], num_sig_fig=3, suppress_suffix='m')
        elif fit_model == 'thg_c6v':
            zzzz_str = get_human_val_str(sim_par['zzzz'], num_sig_fig=3, suppress_suffix='m')
            xxxx_str = get_human_val_str(sim_par['xxxx'], num_sig_fig=3, suppress_suffix='m')
        delta_str = get_human_val_str(
            unwrap_angle(sim_par['delta'])/np.pi*180, num_sig_fig=3)
        err_str = get_human_val_str(err)

    if plot_progress:
        plot_pipo_fit_1point(
            data, fit_model=fit_model, fit_par=par, new_fig=False)
        plt.draw()
        plt.pause(0.001)

    if vlvl >= 2:
        msg = "A = {:s}, δ = {:s}°".format(ampl_str, delta_str)
        if fit_model == 'shg_c6v':
            msg += ", R = {:s}".format(zzz_str)
        elif fit_model == 'thg_c6v':
            msg += ", zzzz = {:s}, xxxx = {:s}".format(zzzz_str, xxxx_str)
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

    The Matlab equation string used in PIPONATOR is:
    'A.*abs((sin(y+r).*sin(2.*(x+r)) + cos(y+r).*(1.*sin(x+r).^2 + Xzzz.*cos(x+r).^2) + 2.*Xxyz.*cos(x+r).*sin((x+r)-(y+r))).^2) + K'
    """
    # TODO: fix this
    psg_states = np.arange(0.0, 180.0, 22.5)/180*np.pi
    psa_states = psg_states # np.arange(0.0, 180.0, 22.5)/180*np.pi
    fit_data = np.ndarray([len(psg_states), len(psa_states)])

    ampl = par[0]
    delta = par[1]
    zzz = par[2]
    xyz = par[3]

    sin_ad = np.empty_like(psa_states)
    cos_ad = np.empty_like(psa_states)

    for ind, psa in enumerate(psa_states):
       sin_ad[ind] = np.sin(psa-delta)
       cos_ad[ind] = np.cos(psa-delta)

    # TODO: implement optional background fitting
    bckg = 0
    for ind_psg, psg in enumerate(psg_states):
        sin_2gd = np.sin(2*(psg-delta))
        cos_gd2 = np.cos(psg-delta)**2
        sin_gd2 = np.sin(psg-delta)**2

        for ind_psa, psa in enumerate(psa_states):
            fit_data[ind_psa, ind_psg] = (
                sin_ad[ind_psa]*sin_2gd +
                cos_ad[ind_psa]*sin_gd2 +
                zzz*cos_ad[ind_psa]*cos_gd2)



            # fit_data[ind_psa, ind_psg] = ampl*(
            #     np.sin(psa-delta)*np.sin(2*(psg-delta)) +
            #     np.cos(psa-delta)*np.sin(psg-delta)**2 +
            #     zzz*np.cos(psa-delta)*np.cos(psg-delta)**2 +
            #     2*xyz*np.cos(psg-delta)*np.sin(psg-psa))**2 + bckg

    fit_data *= fit_data
    fit_data *= ampl

    return fit_data


def pipo_c6v_fitfun(
        par, xdata, data, fit_model='c6v', fitfun_name='c6_ag', fit_accel=None,
        print_progress=False, plot_progress=False, vlvl=1):
    """Fit optimization function for C6 PIPO.

    TODO: prin_progress does nothing, should merge functionality pipo_fitfun
    for this.
    """
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
        print("\nIter: ", par, ", rmse={:.3f}".format(err))
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
    fit_accel = None
    diff_step = None

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

    elif fit_model == 'shg_c6v':
        zzz_min = 1
        zzz_max = 2
        num_zzz = 20
        zzz_step = (zzz_max - zzz_min)/num_zzz
        zzz_arr = np.linspace(zzz_min, zzz_max, num_zzz)

        mapinds = np.reshape(np.arange(0, num_delta*num_zzz), [num_delta, num_zzz])
        accel_file_name = 'fit_accel_shg_c6v.npy'
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
    else:
        print("Fit acceleration for model '{:}' not available".format(fit_model))

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

def fit_pipo_1point(guess_strategy='map', fit_algorithm='scipy_least_quares', **kwargs):
    """Guess initial values and fit single-point PIPO data.

    Some models and data need multiple guesses to not get stuck in a local
    mininum. This function starts with multiple guess fits with lower fitting
    accuracy and then selects the best one for the final fit.

    """
    pipo_arr = kwargs.get('pipo_arr')

    if pipo_arr is None:
        raise Exception("PIPO data not provided")

    if fit_algorithm == 'scipy_minimize':
        from scipy.optimize import minimize, least_squares
        t_start = time()

        def err_func(par_arr):
            return np.sum(np.abs((pipo_arr - par_arr[0]*simulate_pipo(symmetry_str='c6v', nlorder=3, pset_name='pipo_8x8', delta=par_arr[1], zzzz=par_arr[2], xxxx=par_arr[3])).flatten()))

        t_start = time()
        fit_result = minimize(err_func, [np.max(pipo_arr), 0, 3, 3])
        # fit_result = minimize(err_func, [3.125E4, -2.45E-2, 3.08, 4.08])

        # fit_result = least_squares(err_func, [np.max(pipo_arr), 0, 3, 3])

        fit_duration = time() - t_start
        # print(fit_duration)
        # print(fit_result.x)
        fitdata = FitData(fit_result)
        fitdata.set_fit_model(kwargs.get('fit_model'))
        fitdata.set_fitfun_name(fit_algorithm)
        fitdata.set_fit_duration(fit_duration)

        return fitdata

    use_fit_accel = kwargs.get('use_fit_accel')
    fit_accel = kwargs.get('fit_accel')
    if guess_strategy == 'map':
        if use_fit_accel:
            if fit_accel is None:
                fit_accel, diff_step = get_fit_accel(fit_model)

            # test_fit_accel = False
            # if test_fit_accel:
            #     verify_fit_accel(fit_accel)
        else:
            print("Can't use map guessing without a fit accelerator")
            guess_strategy = 'multistart'

    max_ampl = np.max(pipo_arr)
    fit_model = kwargs.get('fit_model')
    if fit_model == 'zcq':
        guess_par = [max_ampl, 0]
        bounds = [
            [0,             -np.pi],
            [max_ampl*1.5,  np.pi]]
    elif fit_model == 'shg_c6v':
        #            A         δ  zzz
        guess_par = [max_ampl, 0, 1]
        bounds = [
            [0,             0,      0.1],
            [max_ampl*100,  np.pi,  10]]
    elif fit_model == 'thg_c6v':
        #            A         δ  zzzz  XXXX
        guess_par = kwargs.get('guess_par', [max_ampl, 0, 10,    5])
        bounds = [
            [0,             0,      0.1,    0.1],
            [max_ampl*100,  np.pi,  100,    100]]
    else:
        raise ValueError("Fit model '{:}' unsupported".format(fit_model))

    vlvl = kwargs.get('vlvl', 0)
    if vlvl >= 1:
        print("Determining guess values...")

    if guess_strategy == 'map':
        # Use a sparse map covering all expected parameter ranges to find a good
        # guess value
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

            if vlvl >= 1:
                print("Guess values: A={:.1f}M, δ={:.1f}°".format(
                    guess_par[0]*1E-6, guess_par[1]/np.pi*180))

        elif fit_model == 'shg_c6v':
            guess_inds = np.unravel_index(
                np.argmin(err), [len(pargrid[0]), len(pargrid[1])])
            guess_par = [max_ampl,
                         pargrid[0][guess_inds[0]],
                         pargrid[1][guess_inds[1]]]
            diff_step = [0.001, 0.01/180*np.pi, 0.001]
            # diff_step = [1, 1/180*np.pi, 0.1]
            x_tol = [1, 1/180*np.pi, 0.1]

            if vlvl >= 1:
                print("Guess values: A={:.1f}M, zzz={:.2f}, δ={:.1f}°".format(
                    guess_par[0]*1E-6, guess_par[1], guess_par[2]/np.pi*180))

        true_par = kwargs.get('true_par')
        if vlvl >= 2 and true_par is not None:
            for ind, guess_par1 in enumerate(guess_par):
                if ind == 0:
                    # Skip amplitude
                    continue

                if find_closest(pargrid[ind-1], true_par[ind]) != guess_inds[ind-1]:
                    print("Parameter map guess could have been better")

        return _fit_pipo_1point(guess_par=guess_par, bounds=bounds, **kwargs)

    elif guess_strategy == 'multistart':
        if fit_model != 'thg_c6v':
            raise RuntimeError("Multistart guess strategy only avaialbe for the THG C6v model")

        guess_set = []
        for delta in np.array([0, 60, 120])/180*np.pi:
            guess_set.append(np.array([max_ampl, delta, 3, 3]))

        kwargs['print_results'] = False
        kwargs['plot_fig'] = False
        kwargs['xtol'] = 0.1
        kwargs['vlvl'] = 0
        kwargs['bounds'] = np.array(bounds)
        guess_fit_data_arr = []
        for guess_ind, guess_par in enumerate(guess_set):
            print("Guess ind {:d} of {:d}".format(guess_ind+1, len(guess_set)))
            try:
                guess_fit_data_arr.append(_fit_pipo_1point(guess_par=guess_par, **kwargs))
            except Exception as excpt:
                print(excpt)

        opt_guess_ind = np.argmin([guess_fit_data.get_fit_err() for guess_fit_data in guess_fit_data_arr])
        opt_guess_par = guess_set[opt_guess_ind]
        kwargs['xtol'] = None
        kwargs['vlvl'] = 1
        kwargs['ftol'] = None
        return _fit_pipo_1point(guess_par=opt_guess_par, **kwargs)


def _fit_pipo_1point(
        pipo_arr=None, file_name=None, fit_model='zcq', fitfun_name=None,
        guess_par=None, true_par=None, bounds=None, xtol=0.0001,
        use_fit_accel=False, fit_accel=None, diff_step=None,
        plot_progress=False, print_results=True, plot_fig=True,
        vlvl=1,
        **kwargs):
    """Fit PIPO using single-point data."""
    t_start = time()

    if pipo_arr is None:
        pipo_arr = load_pipo(file_name, binsz=None)

    if fit_model not in ['zcq', 'shg_c6v', 'thg_c6v']:
        raise Exception("Unsupported fittig model")

    if use_fit_accel and fit_model != 'shg_c6v':
        raise Exception("Fit acceleration only supported for shg_c6v")

    if plot_progress:
        plt.figure(figsize=[12, 5])

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

    if not fitfun_name:
        fitfun_name = get_default_fitfun_name(fit_model)

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

        guess_par[guess_par_bound_check_hi] = np.array(bounds[1])[guess_par_bound_check_hi]

    if plot_progress:
        print("\nGuess parameters: ", guess_par)

    fit_result = least_squares(
        fitfun, guess_par, args=(0, pipo_arr), diff_step=diff_step,
        bounds=bounds, xtol=xtol, kwargs=fit_cfg)

    fit_duration = time() - t_start
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
        vlvl=1, resample=None,
        **kwargs):
    """Fit PIPO using image data."""
    t_start = time()

    if not np.array([inter in fit_model for inter in ['thg', 'shg']]).any():
        raise Exception("Nonlinear interaction (SHG, THG) not indicated")

    if fit_model not in ['zcq', 'shg_c6v', 'thg_c6v']:
        raise Exception("Unsupported fittig model")

    if use_fit_accel and fit_model != 'shg_c6v':
        raise Exception("Fit acceleration only supported for shg_c6v")

    fcfg = ImgFitConfig()
    fcfg.set_fit_model(fit_model)
    fcfg.set_fitfun_name(kwargs.get('fit_fun_name', get_default_fitfun_name(fit_model)))
    fcfg.set_max_fit_pts(max_fit_pts)

    if resample is not None:
        if not isinstance(resample, list):
            raise ValueError("'resample' has to be a list with two elements")
        num_row, num_col = np.shape(pipo_arr)[0:2]
        num_psa, num_psg = np.shape(pipo_arr)[2:4]
        pipo_arr2 = np.ndarray([resample[0], resample[1], num_psa, num_psg])
        resample_fac = np.min([resample[0]/num_row, resample[0]/num_col])
        for ind_psg in range(num_psg):
            for ind_psa in range(num_psa):
                pipo_arr2[:, :, ind_psa, ind_psg] = ndimg.zoom(
                    pipo_arr[:, :, ind_psa, ind_psg], resample_fac)

        pipo_arr = pipo_arr2

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

    with_hist_prog_update = fcfg.get_opt('with_hist_prog_update') # True
    hist_prog_update_period = fcfg.get_opt('hist_prog_update_period') # 10
    fit_smooth_kernel_sz = fcfg.get_opt('fit_smooth_kernel_sz') # 2

    ind_fit = 0
    fit_result = []
    t_last_prog_update = time()
    t_last_hist_prog_update = time()
    t_fit_start = time()
    for ind_row in range(num_row):
        for ind_col in range(num_col):
            if not fit_mask[ind_row, ind_col]:
                continue
            if ind_fit == num_pts_to_fit:
                break

            t_now = time()
            if t_now - t_last_prog_update > 0.5:
                t_last_prog_update = t_now
                elapsed_time = t_now - t_fit_start
                fit_rate = ind_fit/elapsed_time
                est_time_remaining = (num_pts_to_fit - ind_fit)/fit_rate
                msg = "Fitting point {:d} of {:d}. Elapsed time: {:s}, " \
                    "remaining {:s}".format(
                        ind_fit+1, num_pts_to_fit, get_human_val_str(elapsed_time, is_time=True), get_human_val_str(est_time_remaining, is_time=True))
                print(msg)

            if with_hist_prog_update and t_now - t_last_hist_prog_update > hist_prog_update_period:
                try:
                    print("Updating progress figure")
                    t_last_hist_prog_update = t_now
                    ax = kwargs.get('ratio_hist_ax', plt.gca())
                    ax.cla()
                    zzz_arr = [fr1.result.x[2] for fr1 in fit_result]
                    ax.hist(zzz_arr, bins=np.linspace(0.5, 3, 100))
                    plt.title("Fitting progress: {:d}/{:d} points".format(ind_fit+1, num_pts_to_fit))
                    ax.set_xlabel('R ratio')
                    ax.set_ylabel('Count')
                    ax.figure.canvas.draw()
                    ax.figure.canvas.show()
                    plt.draw()
                    plt.gcf().canvas.draw()
                    plt.pause(0.001)
                except Exception:
                    print("Could not draw histogram update figure")

            if fit_smooth_kernel_sz == 1:
                pipo_arr1 = pipo_arr[ind_row, ind_col, :, :]
            else:
                ind_row_from = cap_in_range(np.ceil(ind_row - fit_smooth_kernel_sz/2).astype('int'), [0, num_row])
                ind_row_to = cap_in_range(np.ceil(ind_row + fit_smooth_kernel_sz/2).astype('int'), [0, num_row])
                ind_col_from = cap_in_range(np.ceil(ind_col - fit_smooth_kernel_sz/2).astype('int'), [0, num_col])
                ind_col_to = cap_in_range(np.ceil(ind_col + fit_smooth_kernel_sz/2).astype('int'), [0, num_col])

                pipo_arr1 = np.mean(np.mean(pipo_arr[ind_row_from:ind_row_to, ind_col_from:ind_col_to, :, :], 0), 0)

            try:
                if vlvl >= 2:
                    vlvl1 = vlvl
                else:
                    vlvl1 = 0
                fit_result1 = fit_pipo_1point(
                    pipo_arr=pipo_arr1,
                    fit_model=fit_model,
                    plot_progress=plot_progress, use_fit_accel=use_fit_accel,
                    print_results=False, plot_fig=False,
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

    elapsed_time = time() - t_fit_start
    fit_rate = num_pts_to_fit/elapsed_time
    print("Fitting completed in {:.1f} s at {:.1f} pts/s rate".format(elapsed_time, fit_rate))

    imgfitdata = ImgFitData(fit_result)
    imgfitdata.cfg = fcfg
    imgfitdata.duration = elapsed_time
    imgfitdata.save()

    # print("Saving intermediate fit data to 'fitdata.npy'...")
    # np.save('fitdata.npy', [imgfitdata])

    if print_results:
        imgfitdata.print()

    if plot_results:
        plot_pipo_fit_img(imgfitdata, pipo_arr=pipo_arr)

    return imgfitdata


def bin_pixels(img, target_sz):
    """Bin image pixels to reduce resolution for fitting.

    NOTE: This is a quick and hacky function developed for the
    fit_pipo_img_alt_proto() test and should not be used elsewhere. It doesn't
    work for non-intenger bins, and there are better ways to do binning.
    """
    ind_arr1 = np.linspace(0, img.shape[0], target_sz[0]+1).astype('int')
    ind_arr2 = np.linspace(0, img.shape[1], target_sz[1]+1).astype('int')

    img_out = np.ndarray([*target_sz, *img.shape[2:]])
    img_out.fill(np.nan)
    for out_ind1 in range(target_sz[0]):
        in_ind1_from = ind_arr1[out_ind1]
        in_ind1_to = ind_arr1[out_ind1+1]
        for out_ind2 in range(target_sz[1]):
            in_ind2_from = ind_arr2[out_ind2]
            in_ind2_to = ind_arr2[out_ind2+1]
            img_out[out_ind1, out_ind2, :, :] = img[in_ind1_from:in_ind1_to, in_ind2_from:in_ind2_to, :, :].sum(0).sum(0)

    return img_out

def find_first_true_index(arr):
    """Find the first index of an element in an array that is True."""
    return np.unravel_index(np.where(arr.flatten())[0][0], arr.shape)

def fit_pipo_img_alt_proto(
        data_file_name='data.npy', bin_sz=100, fit_model='thg_c6v',
        fit_algorithm='scipy_minimize', fit_result_file_name='fit_results.npy',
        debug_plot=False):
    """Fit a PIPO image with THG data.

    NOTE: This is an alternative prototype fitting routine for testing THG C6v
    PIPO fitting which cannot be handled by the main fit_pipo_img() routine
    yet. Use the fit_pipo_img() routine whenever possible, this code will
    hopefully be intergrated into main once it is working.

    This routine has only been tested on 8x8 and 9x9 THG PIPO data and only
    with 'thg_c6v' model using 'scipy_minimize'. Input data has to be in the
    'data.npy' file located in the current dir, fit results will be stored in
    the same directory in a file called 'fit_results.npy'.
    """
    print("=== pynlopol ===")
    print("=== Nonlinear polarimetry fitting ===")
    print("This program will fit a nonlinear polarimetry model to the data "
          "provided")
    print("Fit model: ", fit_model)
    print("Algorithm: ", fit_algorithm)

    print("Loading data from file {:s}...".format(data_file_name))
    pipo_data = np.load(data_file_name)
    print("OK")

    [num_psg, num_psa] = np.shape(pipo_data)[2:]
    print("Polarimetric data format: PIPO {:d}x{:d}".format(num_psg, num_psa))
    if num_psg < 8 or num_psg > 9 or num_psa < 8 or num_psa > 9:
        print("WARNING: Unexpected data format. PIPO 8x8 and 9x9 are "
              "supported, but will try anyway.")

    total_img = pipo_data.sum(2).sum(2)
    img_shape = np.shape(total_img)
    num_px = np.prod(img_shape)
    total_counts = int(np.sum(total_img.flatten()))
    avg_counts_px = np.sum(total_img.flatten())/num_px

    print("Image size: ", img_shape)
    print("Total number of counts: {:d}".format(total_counts))
    print("Average counts per pixel: {:.1f}".format(avg_counts_px))

    print("Binning data for fitting to ({:d}, {:d})".format(bin_sz, bin_sz))

    if num_psg > 8 or num_psa > 8:
        print("PIPO data has more than 8 PSG and PSA states. Assuming the "
              "first 8 states cover the full 180 deg sweep and discarding "
              "additional states.")
    pipo_data_bin = bin_pixels(pipo_data, [bin_sz, bin_sz])[:, :, :8, :8]

    pipo_data_to_fit = pipo_data_bin
    fit_data_shape = pipo_data_to_fit.shape
    fit_data_num_px = np.prod(fit_data_shape[:2])
    num_px_to_fit = fit_data_num_px
    fit_result_arr = None
    if check_file_exists(fit_result_file_name):
        fit_result_arr = np.load(fit_result_file_name, allow_pickle=True)
        if fit_result_arr.shape == fit_data_shape[:2]:
            done_px_inds = fit_result_arr != None
            num_px_done = np.sum(done_px_inds)
            frac_done_px = num_px_done/fit_data_num_px
            if frac_done_px == 1.0:
                print("Found a fit data file that already contains a completed"
                      " fit result with the same bin resolution. Continuing "
                      "will overwrite the stored data and perform the fit "
                       "again.")
                input("Press any key to OVERWRITE the fit data and continue "
                      "with the fit...")
                fit_result_arr = None
            else:
                print("Found a fit data file that contains a partially "
                      "completed fit ({:.2f}%) at the same bin resolution. "
                      "Continuing will resume the fit by skipping the pixels "
                       "that are already done. If you want to start the fit "
                       "from the beginning, delete the fit data file.".format(
                           frac_done_px*100))
                input("Press any key to RESUME a partial fit...")
                num_px_to_fit -= num_px_done
        else:
            print("Found a fit data file, but the shape of the stored fit "
                  "data is different from the current. Continuing will "
                  "overwrite the stored fit data. This is normal if you are "
                  "running a new fit with a different bin setting. If you "
                  "don't want to overwrite the stored fit data, close the "
                  "terminal now.")
            input("Press any key to OVERWRITE the fit data and continue with "
                  "the fit...")
            fit_result_arr = None

    if fit_result_arr is None:
        fit_result_arr = np.ndarray([bin_sz, bin_sz], dtype=FitData)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(total_img))
    plt.title('Total count image\n({:d}, {:d}), log scale'.format(
        img_shape[0], img_shape[1]))

    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(pipo_data_bin.sum(2).sum(2)))
    plt.title('Binned image\n({:d}, {:d}), log scale'.format(bin_sz, bin_sz))

    print("Showing total count images")
    print("Close figure to start fitting")
    plt.show()

    num_px_done = 0
    num_row, num_col = pipo_data_to_fit.shape[0:2]
    t_start = time()
    for ind_row in range(num_row):
        for ind_col in range(num_col):
            if fit_result_arr[ind_row, ind_col] != None:
                continue
            t_elapsed = time() - t_start
            print("Pixel {:} of {:}, {:.2f}%, loc ({:}, {:})".format(
                num_px_done, num_px_to_fit, num_px_done/num_px_to_fit*100,
                ind_row, ind_col))
            if num_px_done > 5:
                print("Elapsed time: {:}. Remaining: {:}".format(
                    get_human_val_str(t_elapsed, is_time=True),
                    get_human_val_str(
                        t_elapsed/num_px_done * (num_px_to_fit - num_px_done),
                        is_time=True)))

            pipo_data1 = pipo_data_to_fit[ind_row, ind_col, :, :]
            fit_result = fit_pipo(
                pipo_arr=pipo_data1,
                ask_before_overwrite=False,
                fit_algorithm=fit_algorithm, fit_model=fit_model, use_fit_accel=False,
                print_results=False, plot_progress=False, show_fig=False)

            fit_result_arr[ind_row, ind_col] = fit_result
            num_px_done += 1
            np.save('fit_results.npy', fit_result_arr)

            if debug_plot:
                par_arr = fit_result.result.x
                plt.clf()
                plt.subplot(3, 1, 1)
                plt.imshow(pipo_data1)
                plt.colorbar()
                plt.subplot(3, 1, 2)
                pipo_best_fit = par_arr[0]*simulate_pipo(
                    symmetry_str='c6v', nlorder=3, pset_name='pipo_18x9',
                    delta=par_arr[1], zzzz=par_arr[2], xxxx=par_arr[3])
                plt.imshow(pipo_best_fit)
                plt.colorbar()
                plt.subplot(3, 1, 3)
                res_map = pipo_data1 - pipo_best_fit
                res_max = np.max(np.abs([np.min(res_map), np.max(res_map)]))
                total_err = np.sum(np.abs(res_map.flatten()))
                plt.imshow(res_map, vmin=-res_max, vmax=res_max, cmap='bwr')
                plt.colorbar()
                plt.title('total error={:.3f}Mcnt'.format(total_err*1E-6))

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

    fit_accel = False
    if kwargs.get('use_fit_accel'):
        fitaccel_filename = get_default_fitaccel_filename()
        fit_accel = get_fit_accel(kwargs.get('fit_model'))[0]

    if show_input:
        pipo_arr = load_pipo(file_name, binsz=binsz, cropsz=cropsz, **kwargs)
        print("Showing input image, close the figure window to continue...")
        plt.imshow(np.sum(np.sum(pipo_arr, 2), 2))
        plt.show()
        pipo_arr = None

    if pipo_arr is None:
        pipo_arr = load_pipo(file_name, binsz=binsz, cropsz=cropsz, **kwargs)

    ratio_hist_ax = kwargs.get('ratio_hist_ax')
    total_counts_ax = kwargs.get('total_counts_ax')
    if kwargs.get('plot_progress') and ratio_hist_ax is None and total_counts_ax is None:
        print("Fitting progress updates enabled, creating figure")
        plt.clf()

        try:
            if total_counts_ax is None:
                total_counts_ax = plt.subplot(1, 2, 1)
                kwargs['total_counts_ax'] = total_counts_ax
            else:
                plt.sca(total_counts_ax)

            plt.imshow(pipo_arr)
            plt.title('Total counts')
        except Exception:
            handle_general_exception("Could not make total counts panel")

        try:
            if ratio_hist_ax is None:
                ratio_hist_ax = plt.subplot(1, 2, 2)
                kwargs['ratio_hist_ax'] = ratio_hist_ax
            else:
                plt.sca(ratio_hist_ax)

            plt.xlim([0.5, 3])
            plt.xlabel('R ratio')
        except Exception:
            handle_general_exception("Could not make histogram panel")

        plt.draw()
        plt.pause(0.001)

    if len(np.shape(pipo_arr)) == 4:
        return fit_pipo_img(pipo_arr, fit_accel=fit_accel, **kwargs)
    else:
        return fit_pipo_1point(pipo_arr=pipo_arr, fit_accel=fit_accel, **kwargs)
