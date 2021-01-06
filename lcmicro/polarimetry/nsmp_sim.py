
"""Nonlinear Stokes-Mueller polarimetry (NSMP).

This module contains NSMP simulation routines to make static and animated
SHG PIPO maps.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from lklib.util import round_to
from lklib.plot import export_figure
from lklib.image import make_gif

from lcmicro.polarimetry.gen_pol_state_sequence import gen_pol_state_sequence
from lcmicro.polarimetry.nsmp import get_nsm_matrix
from lcmicro.polarimetry.nsmp_common import get_nsvec
from lcmicro.polarimetry.polarimetry import get_stokes_vec, get_mueller_mat
from lcmicro.polarimetry.report import plot_pipo


def simulate_pipo(trunc_thr=None, pset_name='pipo_8x8', **kwargs):
    """Simulate SHG PIPO response of a sample.

    The code currenlty works for a pipo_8x8 state set only. Input and output
    states are assumed to be HLP. PSA polarizer is assumed to be at HLP.
    """
    # Get PSG and PSA waveplate angles
    pol_angles = gen_pol_state_sequence(pset_name)[1]
    psg_hwp = pol_angles[0]
    psg_qwp = pol_angles[1]
    psa_hwp = pol_angles[2]
    psa_qwp = pol_angles[3]

    num_psg_states = len(psg_hwp)
    num_psa_states = len(psa_hwp)

    det_s0 = np.ndarray([num_psa_states, num_psg_states])

    # Nonlinear Mueller matrix of the sample
    nmmat = get_nsm_matrix(**kwargs)

    # Laser input state
    svec_laser = get_stokes_vec('hlp')

    for ind_psg in range(num_psg_states):
        # PSG Mueller matrix
        mmat_hwp = get_mueller_mat('hwp', psg_hwp[ind_psg])
        mmat_qwp = get_mueller_mat('qwp', psg_qwp[ind_psg])
        mmat_psg = np.dot(mmat_qwp, mmat_hwp)

        # Input Stokes vector
        # The input Stokes vector can be simply taken according to the PSG
        # state list of the state set. Explicit use of Mueller matrices serves
        # an addition check and can be used for additional simulation options,
        # e.g. diattenuation.
        svec_in = np.dot(mmat_psg, svec_laser)
        nsvec_in = get_nsvec(svec_in, nlord=2)

        # SHG Stokes vector after sample
        svec_sample = np.dot(nmmat, nsvec_in)

        for ind_psa in range(num_psa_states):
            # PSA Mueller matrix
            mmat_hwp = get_mueller_mat('hwp', psa_hwp[ind_psa])
            mmat_qwp = get_mueller_mat('qwp', psa_qwp[ind_psa])
            mmat_plz = get_mueller_mat('plz', 0)
            mmat_psa = np.dot(mmat_plz, np.dot(mmat_qwp, mmat_hwp))

            # Output Stokes vector
            svec_out = np.dot(mmat_psa, svec_sample)

            # Intensity at detector
            det_s0[ind_psa, ind_psg] = svec_out[0]

    # Scale to max 1
    det_s0 = det_s0/np.max(det_s0)

    # Round values
    det_s0 = round_to(det_s0, trunc_thr)

    return det_s0


def make_pipo_animation_delta(
    sample='zcq', num_steps=9, fps=3, pset_name='pipo_100x100'):
    """Make PIPO map GIF of a sample by varying delta.

    Args:
        sample - Sample name
        pset_name - Polarization set name
        num_steps - Number of delta steps in aniation
        fps - Display FPS of the GIF
    """
    zzz = None
    if sample == 'zcq':
        title_str = 'Z-cut quartz'
        symmetry_str = 'd3'
    elif sample == 'collagen':
        zzz = 1.5
        title_str = 'Collagen R={:.2f}'.format(zzz)
        symmetry_str = 'c6v'

    print("Making PIPO map animation for " + title_str + " with varying delta")

    # Generate delta array. The +1 and [:-1] syntax makes a linear space that
    # ends one step short of 180 so that the next step is 0 thus making the
    # animation smooth
    delta_arr = np.linspace(0, 180, num_steps+1)[:-1]/180*np.pi

    file_names = []
    for ind, delta in enumerate(delta_arr):
        plt.clf()

        pipo_data = simulate_pipo(
            symmetry_str=symmetry_str, zzz=zzz, delta=delta,
            pset_name=pset_name)

        plot_pipo(pipo_data, show_fig=False, pset_name=pset_name)
        plt.title(title_str + " PIPO map, delta={:.1f} deg".format(delta*180/np.pi))
        print("Exporting frame {:d}".format(ind))
        file_name = 'frame_{:d}.png'.format(ind)
        file_names.append(file_name)
        export_figure(file_name, resize=False)

    print("Exporting GIF...")
    make_gif(output_name=sample + '_delta_pipo.gif', file_names=file_names, fps=fps)

    print("Removing frame files...")
    for file_name in file_names:
        os.remove(file_name)

    print("All done")


def make_pipo_animation_zzz(
        sample='collagen', num_steps=20, fps=3, pset_name='pipo_100x100', **kwargs):
    """Make PIPO map GIF of a sample by varying zzz.

    Args:
        sample - Sample name
        num_steps - Number of zzz steps in aniation
        fps - Display FPS of the GIF
    """
    zzz_arr = np.linspace(0.7, 5, num_steps)
    delta = kwargs.get('delta', 0)
    symmetry_str = 'c6v'

    print("Making PIPO map animation for collagen with delta={:.1f}".format(delta) + " and varying zzz")

    file_names = []
    for ind, zzz in enumerate(zzz_arr):
        plt.clf()

        pipo_data = simulate_pipo(
            symmetry_str=symmetry_str, zzz=zzz, delta=delta,
            pset_name=pset_name)

        plot_pipo(pipo_data, show_fig=False, pset_name=pset_name)

        plt.title("Collagen R={:.2f}".format(zzz) + " PIPO map, delta={:.1f} deg".format(delta*180/np.pi))
        print("Exporting frame {:d}".format(ind))
        file_name = 'frame_{:d}.png'.format(ind)
        file_names.append(file_name)
        export_figure(file_name, resize=False)

    print("Exporting GIF...")
    make_gif(output_name=sample + '_zzz_pipo.gif', file_names=file_names, fps=fps)

    print("Removing frame files...")
    for file_name in file_names:
        os.remove(file_name)

    print("All done")
