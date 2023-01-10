
"""NSMP plotting routines.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lkcom.plot import export_figure, set_ticks_inside, align_outer_tick_labels

from pynlopol.nsmp_common import get_num_states, \
    get_nsmp_state_order


def plot_pipo(
        data, title_str=None, round_to_thr=True, thr=1E-3,
        pset_name='pipo_8x8', bare_plot_data=False,
        export_fig=False, fig_file_name=None, show_fig=True, **kwargs):
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

    # PIPO values at 0 deg and 180 deg are the same and are usually omitted in
    # the dataset for compactness and especially to avoid suggesting to scan
    # the same state twice when tyring to conform to the data format. For
    # display purposes, on the other hand, it is more useful to have duplicate
    # 0 deg and 180 deg to make the PIPO bubble symmetric. The PIPO dataset is
    # expanded here for this reason.
    duplicate_first_state = True
    if duplicate_first_state:
        data2 = np.ndarray(np.array(data.shape) + [1, 1])
        data2.fill(np.nan)
        data2[:-1, :-1] = data
        data2[:-1, -1] = data[:, 0]
        data2[-1, :-1] = data[0, :]
        data2[-1, -1] = data[0, 0]

        psg_states.append('180')
        psa_states.append('180')

        num_psg_states += 1
        num_psa_states += 1

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
    # TODO: contourf looks nicer, but the imshow method will likely work better
    # when the number of PIPO states is low
    plt.contourf(data, origin='lower', cmap='plasma', extent=extent)
    # plt.imshow(data, origin='lower', cmap='plasma', extent=extent)

    plt.axis('square')
    plt.xlim([0, 180])
    plt.ylim([0, 180])

    # Add state labels
    plt.gca()
    if num_psg_states <= 10:
        # Tick every state
        plt.xticks(range(num_psg_states), psg_states)
    if num_psa_states <= 10:
        plt.yticks(range(num_psa_states), psa_states)
    else:
        # Put ticks automatically at 45-deg intervals, e.g. 0, 45, 90, 135
        plt.gca().xaxis.set_major_locator(MaxNLocator(steps=[4.5]))
        plt.gca().yaxis.set_major_locator(MaxNLocator(steps=[4.5]))

        # TODO: in some cases ticks are better placed at base-60 intervals:
        #   0, 10,  20,  30
        #   0, 20,  40,  60
        #   0, 30,  60,  90
        #   0, 60, 120, 180
        # plt.gca().xaxis.set_major_locator(MaxNLocator(steps=[1, 2, 3, 6]))
        # plt.gca().yaxis.set_major_locator(MaxNLocator(steps=[1, 2, 3, 6]))

    plt.xlabel('Input, deg')
    plt.ylabel('Output, deg')

    if title_str is not None:
        plt.title(title_str)

    set_ticks_inside()
    align_outer_tick_labels(plt.gca())

    if bare_plot_data:
        plt.axis('off')

    if export_fig:
        print("Exporting figure...")
        if fig_file_name is None:
            fig_file_name = 'pipo.png'

        export_figure(fig_file_name, resize=False)

    if show_fig:
        plt.show()
