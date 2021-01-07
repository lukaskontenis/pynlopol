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

from lcmicro.proc import load_pipo
from lcmicro.polarimetry.nsmp_common import get_num_states, \
    get_nsmp_state_order
from lcmicro.polarimetry.nsmp_sim import simulate_pipo
from lcmicro.polarimetry.fitdata import get_parmap


def plot_pipo(
        data, title_str=None, round_to_thr=True, thr=1E-3,
        pset_name='pipo_8x8',
        export_fig=False, fig_file_name=None, show_fig=True):
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
    plt.imshow(data, origin='lower', cmap='gray', extent=extent)

    # Add state labels
    plt.gca()
    if num_psg_states <= 10:
        # Tick every state
        plt.xticks(range(num_psg_states), psg_states)
    if num_psa_states <= 10:
        plt.yticks(range(num_psa_states), psa_states)
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

    plt.xlabel('Input, deg')
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
    if fit_model not in ['zcq', 'c6v']:
        raise Exception("Unsupported fitting model")
    if fit_par is None:
        raise Exception("No fit parameters given")
    if (fit_model == 'zcq' and len(fit_par) != 2) or \
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
    err = np.mean(np.sqrt(res**2))

    ampl_str = get_human_val_str(ampl, suppress_suffix='m')
    zzz_str = get_human_val_str(zzz, num_sig_fig=3, suppress_suffix='m')
    delta_str = get_human_val_str(
        unwrap_angle(delta, period=delta_period)/np.pi*180,
        num_sig_fig=3, suppress_suffix='m')
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
        plt.title('Fit model ''{:s}''\nA = {:s}, δ = {:s}°'.format(
            fit_model, ampl_str, delta_str))
    else:
        plt.title('Fit model ''{:s}''\nA = {:s}, R = {:s}, δ = {:s}°'.format(
            fit_model, ampl_str, zzz_str, delta_str))
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


def plot_pipo_fit_img(
        fitdata, pipo_arr=None,
        show_fig=True, new_fig=True,
        export_fig=False, fig_file_name=None, **kwargs):
    """Make a PIPO fit result figure for an image."""
    plt.figure(figsize=[10, 10])

    ax = plt.subplot(2, 2, 1)
    total_cnt_img = np.sum(np.sum(pipo_arr, 2), 2)
    total_cnt_img[0, 0] = 0
    imshow_ex(
        total_cnt_img, bad_color='black', ax=ax, logscale=True, cmap='viridis',
        title_str='SHG intensity', with_hist=True)

    fit_model = fitdata['model']
    zzz = None
    if fit_model in ['zcq', 'c6v']:
        ampl = get_parmap(fitdata, 'x', 0)
        delta = get_parmap(fitdata, 'x', 1)

    if fit_model == 'c6v':
        zzz = get_parmap(fitdata, 'x', 2)

    ax = plt.subplot(2, 2, 2)
    imshow_ex(
        ampl, ax=ax, logscale=False, cmap='viridis',
        title_str='Amplitude (counts)', with_hist=True)

    ax = plt.subplot(2, 2, 3)
    imshow_ex(
        delta, ax=ax, logscale=False, cmap='hsv', title_str='delta (deg)',
        with_hist=True, is_angle=True)

    ax = plt.subplot(2, 2, 4)
    imshow_ex(
        zzz, ax=ax, logscale=False, cmap='plasma', title_str='zzz',
        with_hist=True)

    if export_fig:
        print("Exporting figure...")
        if fig_file_name is None:
            fig_file_name = 'pipo_fit.png'

        export_figure(fig_file_name, resize=False)

    if show_fig:
        plt.show()


def make_pipo_fig(file_name):
    """Make a PIPO figure from a dataset."""
    pipo_arr = load_pipo(file_name)

    title_str = 'PIPO ' + os.path.basename(file_name)
    plot_pipo(pipo_arr, title_str=title_str, export_fig=True)
