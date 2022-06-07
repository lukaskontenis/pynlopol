
"""NSMP plotting routines.

This script is part of pynolpol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from lkcom.plot import export_figure

from pynolpol.nsmp_common import get_num_states, \
    get_nsmp_state_order


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
    plt.imshow(data, origin='lower', cmap='plasma', extent=extent)

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
