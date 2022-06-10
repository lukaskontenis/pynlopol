
"""Nonlinear Stokes-Mueller polarimetry (NSMP).

This module contains advanced NSMP routines, more common functions are
in nsmp_common.py.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import time
import numpy as np

from pynlopol.class_chi import ClassChi
from pynlopol.nsmp_common import get_pauli_matrix_arr, \
    get_gell_mann_matrix_arr

from lkcom.util import isnone, round_to
from lkcom.string import get_human_val_str


def get_lab_chi(trunc_thr=None, **kwargs):
    """Get the laboratory-frame components of a tensor."""
    # Create tensor
    chi = ClassChi(**kwargs)
    chi.rotate(**kwargs)
    chi.zero_small_values()
    return round_to(chi.get_lab_chi(), trunc_thr)


def get_nsm_matrix(
        delta=0, alpha=0, theta=0, nlorder=None, sample_par=None,
        cmplx_r=0, algorithm=None, eff_chi=0, trunc_thr=None, **kwargs):
    """Get the nonlinear Stokes-Mueller matrix.

    Get the nonlinear Stokes Mueller matrix for a given order, symmetry type,
    and sample orienation angles delta (in-plane) and alpha (out-of-plane).

    If algorithm is not specified, 'formulae_rk' is used if the tensor is real
    and 'general' is used if the tensor is complex.

    This code was translated from GetNSMMatrix.m in NLPS suite on 2020.11.21.
    At the time the code had a last modification timestamp of 2017.07.19, the
    modification year in code was 2015.
    """
    # TODO: Translate sample_par parsing
    # Parse delta and alpha angles from the sample_par structure
    # if(~isempty(sample_par))
    #     if(isfield(sample_par, 'delta'))
    #         delta = sample_par.delta
    #     end
    #     if(isfield(sample_par, 'alpha'))
    #         alpha = sample_par.alpha
    #     end
    # end

    verbosity = kwargs.get('verbosity', 0)

    # Create and rotate a tensor
    if verbosity >= 2:
        print('Generating tensor...')
        t_start = time.time()

    # Create tensor
    chi = ClassChi(**kwargs)

    chi.zero_small_values()

    # Get the nonlinear order
    if isnone(nlorder):
        nlorder = chi.get_nlorder()

    num_rows = 4
    if nlorder == 2:
        nlproc = 'shg'
        num_col = 9
        num_chi = 3
    elif nlorder == 3:
        nlproc = 'thg'
        num_col = 16
        num_chi = 4
    else:
        print('Unsupported nonlinear order {:d}'.format(nlorder))
        return None

    if not chi.is_valid() and verbosity >= 2:
        print('Invalid tensor')
        return None
    elif verbosity >= 2:
        print('Done in {:.3f} s'.format(time.time() - t_start))

    # TODO: Translate symbolic mode
    # is_symbolic = Chi.IsSymbolic()
    is_symbolic = False

    if verbosity >= 2:
        print('Rotating tensor...')
        t_start = time.time()

    if theta:
        print("Theta rotation not implemented yet")
        # chi.rotate_tda_via_top(theta, delta, alpha, **kwargs)
    else:
        chi.rotate(delta=delta, alpha=alpha, **kwargs)

    if verbosity >= 2:
        print('Done in {:.3f} s'.format(time.time() - t_start))

    if isnone(algorithm):
        if not isnone(cmplx_r):
            algorithm = 'general'
        else:
            algorithm = 'formulae_rk'

    if verbosity >= 2:
        print('Calculating Mueller matrix using ' + algorithm + \
              ' algorithm...')
        t_start = time.time()

    if algorithm == 'general':
        if not chi.is_symbolic():
            mmat = np.ndarray([num_rows, num_col])

        pmat_arr = get_pauli_matrix_arr(nlproc)
        gmmat_arr = get_gell_mann_matrix_arr(nlproc)

        # It is critcal that the correct (X or Z) axis is chosen as the
        # primary. Normally the Z axis is primary one, but in the DSMP paper
        # the x axis was chosen as the primary. An incorrect choice will
        # introduce a 90° delta offset and may have other consequences.
        #
        # Since the nonlinear Mueller matrix calculation is performed using
        # the generalized NSMP framework, the primary axis should be Z. For
        # legacy uses it can be changed to X, but this may or may not work.
        chimat = chi.get_ms_contraction(primary_axis='x')

        for ind_t in range(num_rows):
            for ind_N in range(num_col):
                if verbosity >= 2:
                    print('Element {:d} of {:d}'.format(ind_t*num_col + ind_N, num_rows*num_col))

                val = 0
                for ind_a in range(2):
                    for ind_b in range(2):
                        for ind_A in range(num_chi):
                            for ind_B in range(num_chi):
                                val += np.conj(chimat[ind_a, ind_A])*chimat[ind_b, ind_B]*pmat_arr[ind_a, ind_b, ind_t]*gmmat_arr[ind_B, ind_A, ind_N]

                if not is_symbolic:
                    if np.abs(np.imag(val)) > 1E-10:
                        print('Unexpected imaginary values in the Muller matrix!')
                    val = np.real(val)
                mmat[ind_t, ind_N] = val/2

        if not is_symbolic:
            mask = np.abs(np.imag(mmat)) > 100*np.finfo(mmat.dtype).eps

            if np.any(mask):
                print('Nonlinear Mueller matrix contains imaginary elements!')

            mmat[mask] = np.real(mmat[mask])

    elif algorithm == 'formulae_rk':
        chi = chi.get_tensor_array()
        if nlorder == 2:
            xxx = xmat[0, 0, 0]
            zxx = xmat[2, 0, 0]
            zzz = xmat[2, 2, 2]
            xzz = xmat[0, 2, 2]

            mmat = np.zeros([4, 9])

            # if(strcmp(class(X), 'sym'))
            #     M = sym(M)

            mmat[0, 0] = 1/np.sqrt(6) * (xxx**2 + zzz**2 + 2*xzz**2 + 2*zxx**2)
            mmat[0, 1] = 1/np.sqrt(12) * (xxx**2 + zzz**2 - xzz**2 - zxx**2)
            mmat[0, 2] = 1/2 *(xxx**2 - zzz**2 - xzz**2 + zxx**2)
            mmat[0, 3] = zzz*zxx + xxx*xzz
            mmat[0, 4] = zzz*xzz + xzz*zxx
            mmat[0, 5] = xzz*zxx + xxx*zxx

            mmat[1, 0] = -1/np.sqrt(6) * (xxx**2 - zzz**2)
            mmat[1, 1] = -1/np.sqrt(12) * (xxx**2 - zzz**2 + 3*xzz**2 - 3*zxx**2)
            mmat[1, 2] = -1/2 * (xxx**2 + zzz**2 - xzz**2 - zxx**2)
            mmat[1, 3] = zzz*zxx - xxx*xzz
            mmat[1, 4] = zzz*xzz - xzz*zxx
            mmat[1, 5] = xzz*zxx - xxx*zxx

            mmat[2, 0] = np.sqrt(2)/np.sqrt(3) * (xxx*zxx + xzz*zxx + zzz*xzz)
            mmat[2, 1] = 1/np.sqrt(3) * (xxx*zxx - 2*xzz*zxx + zzz*xzz)
            mmat[2, 2] = xxx*zxx - zzz*xzz
            mmat[2, 3] = xzz*zxx + zzz*xxx
            mmat[2, 4] = zzz*zxx + xzz**2
            mmat[2, 5] = zxx**2 + xxx * xzz

            mmat[3, 6] = xzz*zxx - zzz*xxx
            mmat[3, 7] = zzz*zxx - xzz**2
            mmat[3, 8] = zxx**2 - xxx*xzz

        elif nlorder == 3:
            zzzz = xmat[2, 2, 2, 2]
            zzxx = xmat[2, 2, 0, 0]
            xxxx = xmat[0, 0, 0, 0]
            zzzx = xmat[2, 2, 2, 0]
            zxxx = xmat[2, 0, 0, 0]

            mmat = np.zeros([4, 16])

            # if(strcmp(class(X), 'sym'))
            #     M = sym(M)

            mmat[0, 0] = np.sqrt(2)/4  * (zzzz**2 + xxxx**2 + 2*zzzx**2 + 2*zxxx**2 + 2*zzxx**2)
            mmat[0, 1] = np.sqrt(6)/12 * (zzzz**2 + xxxx**2 + 2*zzzx**2 - 2*zxxx**2 - 2*zzxx**2)
            mmat[0, 2] = np.sqrt(3)/6  * (zzzz**2 + xxxx**2 -   zzzx**2 +   zxxx**2 - 2*zzxx**2)
            mmat[0, 3] =          1/2  * (zzzz**2 - xxxx**2 +   zzzx**2 -   zxxx** 2           )

            mmat[0, 4] = zzzz*zxxx + xxxx*zzzx
            mmat[0, 5] = xxxx*zzxx + zzzx*zxxx
            mmat[0, 6] = zzzx*zzxx + zxxx*zzxx
            mmat[0, 7] = xxxx*zxxx + zxxx*zzxx
            mmat[0, 8] = zzzz*zzzx + zzzx*zzxx
            mmat[0, 9]= zzzz*zzxx + zzzx*zxxx

            mmat[1, 0] = np.sqrt(2)/4 *(zzzz**2 - xxxx**2                                 )
            mmat[1, 1] = np.sqrt(6)/12*(zzzz**2 - xxxx**2            + 4*zxxx**2 - 4*zzxx**2)
            mmat[1, 2] = np.sqrt(3)/6 *(zzzz**2 - xxxx**2 - 3*zzzx**2 +   zxxx**2 + 2*zzxx**2)
            mmat[1, 3] =       1/2 *(zzzz**2 + xxxx**2 -   zzzx**2 -   zxxx**2           )

            mmat[1, 4] = zzzz*zxxx - xxxx*zzzx
            mmat[1, 5] = zzzx*zxxx - xxxx*zzxx
            mmat[1, 6] = zzzx*zzxx - zxxx*zzxx
            mmat[1, 7] = zxxx*zzxx - xxxx*zxxx
            mmat[1, 8] = zzzz*zzzx - zzzx*zzxx
            mmat[1, 9] = zzzz*zzxx - zzzx*zxxx

            mmat[2, 0] = np.sqrt(2)/2 *(zzzx*zzxx +   zxxx*zzxx + zzzz*zzzx + xxxx*zxxx)
            mmat[2, 1] = np.sqrt(6)/6 *(zzzx*zzxx - 3*zxxx*zzxx + zzzz*zzzx + xxxx*zxxx)
            mmat[2, 2] = np.sqrt(3)/3 *(zzzz*zzzx - 2*zzzx*zzxx + xxxx*zxxx)

            mmat[2, 3] = zzzz*zzzx - xxxx*zxxx
            mmat[2, 4] = zzzz*xxxx + zzzx*zxxx
            mmat[2, 5] = zxxx*zzxx + xxxx*zzzx
            mmat[2, 6] = zzzx*zxxx + zzxx**2
            mmat[2, 7] = zxxx**2 + xxxx*zzxx
            mmat[2, 8] = zzzx**2 + zzzz*zzxx
            mmat[2, 9] = zzzx*zzxx + zzzz*zxxx

            mmat[3, 10] = zzzz*xxxx - zzzx*zxxx
            mmat[3, 11] = zxxx*zzxx - xxxx*zzzx
            mmat[3, 12] = zzzx*zxxx - zzxx**2
            mmat[3, 13] = zxxx**2 - xxxx*zzxx
            mmat[3, 14] = zzzz*zzxx - zzzx**2
            mmat[3, 15] = zzzz*zxxx - zzzx*zzxx

        else:
            raise(Exception('Cannot get nonlinear Mueller matrix for {:d} order'.format(nlorder)))

    # if(is_symbolic && correct_rotation_frame)

    #     % Use a persistent variable to print the DSMP/NSMP warning only once
    #     % every 30 seconds. This is not a very good solution since the warning
    #     % can now be not printed for the first 30 seconds of a fitting run,
    #     % also it will not be printed a the fitting run only calls GetNSMMatrix
    #     % once. A better solution would be to print warnings like these at the
    #     % end of the execution.
    #     persistent last_warn
    #     curr_time = clock
    #     curr_sec = curr_time(end)
    #     if(isempty(last_warn) || abs(curr_sec - last_warn) > 30)
    #         last_warn = curr_sec
    #         PrintWarning([ 'Switching between legacy DSMP and general NSMP theory ', ...
    #         'frames of reference by rotating the in-plane angle by 90 deg. This ', ...
    #         'operation has not been verified and may not work in all cases. Please ', ...
    #         'update core NLPS functions once conflicting frames of reference ', ...
    #         'in NSMP have been resolved. You have been warned.' ])
    #     end

    #     d1 = sym('delta', 'real')
    #     d2 = d1 + sym('PI/2')
    #     M = subs(M, d1, d2)

    if verbosity >= 2:
        print('Done in {:.3f} s'.format(time.time() - t_start))

    return round_to(mmat, trunc_thr)


def print_mueller_mat(mmat):
    """Print a nonlinear Mueller matrix."""
    for ind_row in range(mmat.shape[0]):
        for ind_col in range(mmat.shape[1]):
            val_str = get_human_val_str(
                mmat[ind_row, ind_col],
                str_order=0,
                fixed_width_sign=True,
                show_suffix=False,
                num_sig_fig=3,
                fixed_str_len=4)
            print(val_str + ', ', end='')
        print('\n')


def print_chid(chid):
    """Print the contracted chi tensor."""
    for ind_row in range(chid.shape[0]):
        for ind_col in range(chid.shape[1]):
            val_str = get_human_val_str(
                chid[ind_row, ind_col],
                str_order=0,
                fixed_width_sign=True,
                show_suffix=False,
                fixed_str_len=4)
            print(val_str + ', ', end='')
        print('\n')
