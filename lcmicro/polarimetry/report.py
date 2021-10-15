"""Polarimetry figure generation.

This module contains plotting routines for linear and nonlinear polarimetry.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import os
import numpy as np
import matplotlib.pyplot as plt


from lklib.util import unwrap_angle
from lklib.plot import export_figure, imshow_ex
from lklib.string import get_human_val_str
from lklib.fileread import list_files_by_pattern, check_file_exists

from lcmicro.proc import load_pipo
from lcmicro.polarimetry.nsmp_common import get_num_states, \
    get_nsmp_state_order
from lcmicro.polarimetry.nsmp_sim import simulate_pipo
from lcmicro.polarimetry.imgfitdata import ImgFitData


def plot_pipo(
        data, title_str=None, round_to_thr=True, thr=1E-3,
        pset_name='pipo_8x8',
        cmap='gray', tick_step=1, show_x_label=True, show_y_label=True,
        export_fig=False, fig_file_name=None, show_fig=False):
    """Plot a PIPO map.

    Args:
        data - PSGxPSA PIPO intensity array
        title_srt - Figure title string
        round_to_thr - Force PIPO array intensities below thr to zero
        thr - PIPO intensity threshold
        show_fig - Show figure
    """
    if round_to_thr:
        # Round PIPO intensities so that small values are zero in the figure
        # and do not distract from the significant values
        data = np.round(data/thr)*thr

    num_psg_states, num_psa_states = get_num_states(pset_name)
    psg_states, psa_states = get_nsmp_state_order(pset_name)

    # When adding x and y ticks for a small number of PSG and PSA states, it's
    # best tick each state and center the tick on the pixel. For a large number
    # of states, it's better to place ticks automatically on the angle x and y
    # axes by setting the image extent.
    if num_psg_states <= 10 or num_psa_states <= 10:
        extent = None
    else:
        extent = [float(x) for x in
                  [psg_states[0], psg_states[-1],
                   psa_states[0], psa_states[-1]]]

    # Plot PIPO map
    plt.imshow(data, origin='lower', cmap=cmap, extent=extent)

    # Add state labels
    plt.gca()
    if num_psg_states <= 10:
        # Tick every state, unless skip is set
        if tick_step is not 1:
            tick_vals = [psg_states[ind] for ind in np.arange(0, num_psg_states, tick_step)]
        else:
            tick_vals = psg_states

        plt.xticks(np.arange(0, num_psg_states, tick_step), tick_vals)

    if num_psa_states <= 10:
        if tick_step is not 1:
            tick_vals = [psa_states[ind] for ind in np.arange(0, num_psa_states, tick_step)]
        else:
            tick_vals = psa_states

        plt.yticks(np.arange(0, num_psa_states, tick_step), tick_vals)
    else:
        # Generate ticks automatically using 60-based 1, 2, 3, 6 step
        # multiples, e.g.:
        #   0, 10,  20,  30
        #   0, 20,  40,  60
        #   0, 30,  60,  90
        #   0, 60, 120, 180
        # Automatic ticking defaults to a 10-based 1, 2, 4, 5, 10, which does
        # not work well for angles
        plt.gca().xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 3, 6]))
        plt.gca().yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 3, 6]))

    if show_x_label:
        plt.xlabel('Input, deg')

    if show_y_label:
        plt.ylabel('Output, deg')

    if title_str is not None:
        plt.title(title_str)

    if export_fig:
        print("Exporting figure...")
        if fig_file_name is None:
            fig_file_name = 'pipo.png'

        export_figure(fig_file_name, resize=False)

    if show_fig:
        plt.show()


def plot_pipo_fit_1point(
        data, fit_model=None, fit_par=None, fit_data=None,
        show_fig=False, new_fig=True,
        export_fig=False, fig_file_name=None, **kwargs):
    """Plot PIPO fit result for a single point."""
    if fit_data is None and fit_model not in ['zcq', 'c6v']:
        raise Exception("Unsupported fitting model")

    if fit_data is None and fit_par is None:
        raise Exception("No fit parameters given")

    if fit_model and (fit_model == 'zcq' and len(fit_par) != 2) or \
            (fit_model == 'c6v' and len(fit_par) != 3):
        raise Exception("Incorrect number of fit parameters")

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
        fit_data = ampl*simulate_pipo(
            symmetry_str=symmetry_str, delta=delta, zzz=zzz)

    res = data - fit_data
    err = np.sqrt(np.mean(res**2))
    err_str = get_human_val_str(err)

    if fit_par is not None:
        ampl_str = get_human_val_str(ampl, suppress_suffix='m')
        zzz_str = get_human_val_str(zzz, num_sig_fig=3, suppress_suffix='m')
        delta_str = get_human_val_str(
            unwrap_angle(delta, period=delta_period)/np.pi*180,
            num_sig_fig=3, suppress_suffix='m')

    if new_fig:
        plt.figure(figsize=[12, 5])
    else:
        plt.clf()

    plt.subplot(1, 3, 1)
    plot_pipo(data, tick_step=2)
    total_cnt = data.sum()
    total_cnt_str = get_human_val_str(total_cnt)
    plt.title('Data\nTotal counts: ' + total_cnt_str)
    plt.subplot(1, 3, 2)
    plot_pipo(fit_data, tick_step=2, show_y_label=False)
    if fit_model == 'zcq':
        plt.title('Fit model ''{:s}''\nA = {:s}, δ = {:s}°'.format(
            fit_model, ampl_str, delta_str))
    elif fit_model == 'c6v':
        plt.title('Fit model ''{:s}''\nA = {:s}, R = {:s}, δ = {:s}°'.format(
            fit_model, ampl_str, zzz_str, delta_str))
    plt.subplot(1, 3, 3)
    plot_pipo(res, cmap='coolwarm', round_to_thr=False, tick_step=2, show_y_label=False)

    frac_err = err/total_cnt
    frac_err_str = "{:.2f}%".format(frac_err*100)
    plt.title('Residuals, rmse = {:s}'.format(err_str) + ', ' + frac_err_str)

    if export_fig:
        print("Exporting figure...")
        if fig_file_name is None:
            fig_file_name = 'pipo_fit.png'

        export_figure(fig_file_name, resize=False)

    if show_fig:
        plt.show()


def plot_pipo_fit_img(
        fitdata, pipo_arr=None,
        show_fig=True, new_fig=True,
        export_fig=False, fig_file_name=None, **kwargs):
    """Make a PIPO fit result figure for an image."""
    plt.figure(figsize=[10, 10])

    if pipo_arr is not None:
        ax = plt.subplot(2, 2, 1)
        total_cnt_img = np.sum(np.sum(pipo_arr, 2), 2)
        total_cnt_img[0, 0] = 0
        imshow_ex(
            total_cnt_img, bad_color='black', ax=ax, logscale=True, cmap='viridis',
            title_str='SHG intensity', with_hist=True)
    else:
        print("PIPO array not available, total count image will not be shown")

    fit_model = fitdata.get_fit_model()
    zzz = None
    if fit_model in ['zcq', 'c6v', 'c6']:
        ampl = fitdata.get_par()['ampl']
        delta = unwrap_angle(fitdata.get_par()['delta'])

    if fit_model in ['c6v', 'c6']:
        zzz = fitdata.get_par()['zzz']

    ax = plt.subplot(2, 2, 2)
    imshow_ex(
        ampl, ax=ax, logscale=False, cmap='viridis',
        title_str='Amplitude (counts)', with_hist=True)

    ax = plt.subplot(2, 2, 3)
    imshow_ex(
        delta, vmin=-90, vmax=90, ax=ax, logscale=False, cmap='hsv', title_str='delta (deg)',
        with_hist=True, is_angle=True)

    ax = plt.subplot(2, 2, 4)
    imshow_ex(
        zzz, ax=ax, logscale=False, cmap='plasma', title_str='zzz',
        vmin=0.5, vmax=3, min_vspan=0.2,
        with_hist=True)

    if export_fig:
        print("Exporting figure...")
        if fig_file_name is None:
            fig_file_name = 'pipo_fit.png'

        export_figure(fig_file_name, resize=False)

    if show_fig:
        plt.show()

def plot_piponator_fit():
    """Plot PIPONATOR fit results."""
    file_names = list_files_by_pattern('.', match_pattern=['CAK.bin'])

    if len(file_names) == 0:
        print("No PIPONATOR CAK/SKC files found")
        return

    if len(file_names) > 1:
        print("More than one set of CAK/SKC files found, using the first one")

    file_name_cak = file_names[0]
    file_name_skc = file_name_cak.split('CAK.bin')[0] + 'SKC.bin'

    if not check_file_exists(file_name_skc):
        print("SKC file not found")
        return

    cak_data = np.fromfile(file_name_cak, dtype='double')
    cak_data = cak_data.reshape([13, 128, 128])

    zzz = cak_data[0, :, :]
    delta = unwrap_angle(cak_data[1, :, :])
    backg = cak_data[2, :, :]
    rsqad = cak_data[3, :, :]
    xyz = cak_data[5, :, :]
    ampl = cak_data[8, :, :]

    fitdata = ImgFitData()
    fitdata.set_par({'ampl': ampl, 'backg': backg, 'delta': delta, 'zzz': zzz, 'xyz': xyz})
    fitdata.set_fit_err(rsqad, type='rsqad')
    fitdata.cfg.fit_model = 'c6'
    fitdata.set_mask(rsqad == 0)

    plot_pipo_fit_img(fitdata)

def make_pipo_fig(file_name):
    """Make a PIPO figure from a dataset."""
    pipo_arr = load_pipo(file_name)

    title_str = 'PIPO ' + os.path.basename(file_name)
    plot_pipo(pipo_arr, title_str=title_str, export_fig=True)
