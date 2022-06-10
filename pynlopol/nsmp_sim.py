
"""Nonlinear Stokes-Mueller polarimetry (NSMP) simulation.

This module contains NSMP simulation routines to make static and animated
SHG PIPO maps.

This script is part of pynlopol, a Python library for nonlinear polarimetry.w

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import imageio

from lkcom.util import round_to
from lkcom.plot import export_figure
from lkcom.image import make_gif

from pynlopol.gen_pol_state_sequence import gen_pol_state_sequence
from pynlopol.nsmp import get_nsm_matrix
from pynlopol.nsmp_common import get_nsvec
from pynlopol.polarimetry import get_stokes_vec, get_mueller_mat
from pynlopol.plot import plot_pipo


def simulate_pipo_1point(trunc_thr=None, pset_name='pipo_8x8', **kwargs):
    """Simulate SHG PIPO response of a sample.

    The code currenlty works for a pipo_8x8 state set only. Input and output
    states are assumed to be HLP. PSA polarizer is assumed to be at HLP.
    """
    # Get PSG and PSA waveplate angles
    pol_angles = gen_pol_state_sequence(pset_name, vlvl=0)[1]
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


def simulate_pipo_img(
        img_sz=[128, 128], dtype='float', max_ampl=100, img_type='ramp',
        ref_img_name=None, with_poisson_noise=True, **kwargs):
    """Simulate a PIPO dataset.

    Simulate a PIPO dataset for a given tensor and sample parameters.
    Currently, SHG c6v (for collagen, muscle and starch) and D3 (for z-cut
    quartz) tensors are supported with R=zzz/zxx ratio and in-plane
    angle (delta) parameters.

    The symmetry_str, zzz and delta arguments are handled by
    simulate_pipo_1point().

    Args:
        symetry_str – 'c6v' for collagen and 'd3' for z-cut quartz
        zzz – R-ratio (zzz/zxx) in the collagen case
        delta – sample in-plane orientation angle in degrees
        img_sz - image size [rows, cols] in pixels
        dtype - output data type, 'float' or 'uint16'
        max_ampl - maximum amplitude
        img_type - 'flat', 'ramp', 'ref_img'
        ref_img_name - reference image name
        with_poisson noise - sample pixel count values from a Poisson
            distribution
    """
    pipo_arr1 = simulate_pipo_1point(**kwargs)
    num_psa, num_psg = np.shape(pipo_arr1)
    num_row = img_sz[0]
    num_col = img_sz[1]

    pipo_arr = np.ndarray([num_row, num_col, num_psa, num_psg])

    for ind_row in range(num_row):
        for ind_col in range(num_col):
            pipo_arr[ind_row, ind_col, :, :] = pipo_arr1

    mask = np.ndarray([num_row, num_col])

    if img_type == 'ramp':
        mask_vec = np.linspace(1, max_ampl, num_col)
        for indrow in range(num_row):
            mask[indrow, :] = mask_vec

        mask *= max_ampl

    elif img_type == 'ref_img':
        mask = imageio.imread(ref_img_name)

    else:
        mask.fill(max_ampl)

    for indrow in range(num_row):
        for indcol in range(num_col):
            pipo_arr[indrow, indcol, :, :] = mask[indrow, indcol]*pipo_arr1

    if with_poisson_noise:
        pipo_arr = np.random.poisson(pipo_arr)

    return pipo_arr.astype(dtype)


def simulate_pipo(output_type='1point', **kwargs):
    """Simulate a PIPO dataset.

    Simulate a single-point PIPO dataset or a PIPO image. See the documentation
    for simulate_pipo_1point or simuate_pipo_img for more details.
    """
    if output_type == '1point':
        return simulate_pipo_1point(**kwargs)
    elif output_type == 'img':
        return simulate_pipo_img(**kwargs)
    else:
        print("Unsupported output type '{:s}'".format(output_type))
        return None


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

        plt.title(
            title_str + " PIPO map, delta={:.1f} deg".format(delta*180/np.pi))

        print("Exporting frame {:d}".format(ind))
        file_name = 'frame_{:d}.png'.format(ind)
        file_names.append(file_name)
        export_figure(file_name, resize=False)

    print("Exporting GIF...")
    make_gif(
        output_name=sample + '_delta_pipo.gif', file_names=file_names, fps=fps)

    print("Removing frame files...")
    for file_name in file_names:
        os.remove(file_name)

    print("All done")


def make_pipo_animation_zzz(
        sample='collagen', num_steps=20, fps=3, pset_name='pipo_100x100',
        **kwargs):
    """Make PIPO map GIF of a sample by varying zzz.

    Args:
        sample - Sample name
        num_steps - Number of zzz steps in aniation
        fps - Display FPS of the GIF
    """
    zzz_arr = np.linspace(0.7, 5, num_steps)
    delta = kwargs.get('delta', 0)
    symmetry_str = 'c6v'

    print("Making PIPO map animation for collagen with delta={:.1f} and "
          "varying zzz".format(delta))

    file_names = []
    for ind, zzz in enumerate(zzz_arr):
        plt.clf()

        pipo_data = simulate_pipo(
            symmetry_str=symmetry_str, zzz=zzz, delta=delta,
            pset_name=pset_name)

        plot_pipo(pipo_data, show_fig=False, pset_name=pset_name)

        plt.title("Collagen R={:.2f} PIPO map, delta={:.1f} deg".format(
            zzz, delta*180/np.pi))
        print("Exporting frame {:d}".format(ind))
        file_name = 'frame_{:d}.png'.format(ind)
        file_names.append(file_name)
        export_figure(file_name, resize=False)

    print("Exporting GIF...")
    make_gif(
        output_name=sample + '_zzz_pipo.gif', file_names=file_names, fps=fps)

    print("Removing frame files...")
    for file_name in file_names:
        os.remove(file_name)

    print("All done")
