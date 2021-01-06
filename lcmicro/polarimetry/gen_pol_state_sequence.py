
"""Nonlinear Stokes-Mueller polarimetry (NSMP).

This file contains the gen_pol_state_sequence function to create polarization
state sequences for nonlinear PIPO and NSMP measurements.

Note: a similar MATLAB version of this code is availble as part of NLPS in
GetPIPOStateSequence.m and GetSPIPOStateSequence.m.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import numpy as np

from lcmicro.polarimetry.polarimetry import get_stokes_vec, get_mueller_mat, \
    tensor_eq
from lcmicro.polarimetry.nsmp_common import get_nsmp_state_order, \
    is_pset_pipo, get_num_states, validate_pset_name


def gen_pol_state_sequence(
        pset_name='shg_nsmp', scan_order='psa_first', input_state='hlp',
        output_state='hlp', with_ref_states=True, ref_state='hlp',
        write_files=False, verbosity=0, **kwargs):
    """Create a polarization state sequence for PIPO or NSMP.

    Args:
        pset_name - Polarization state sequence name
        scan_order - PSG or PSA states should be switched first
        input_state - Stokes state at the input before the PSG
        output_state - Stokes state at the output after the PSA
        with_ref_stats - Include reference states after each PSG or PSA cycle
        ref_state - State to use as reference
        write_files - Write the state sequence to PolStates.dat
    """
    # Verbosity level
    vlvl = kwargs.get('vlvl', 2)

    if vlvl >= 2:
        print("Generating polarization state sequence...")
        print("Seqeunce name: {:s}".format(pset_name))
        print("Input state: {:s}".format(input_state))
        print("Output state: {:s}".format(output_state))
        print("Reference states: {:b}".format(with_ref_states))

    validate_pset_name(pset_name)
    valid_scan_oders = {'psa_first', 'psg_first'}

    if scan_order not in valid_scan_oders:
        print("Scan order '{:s}' is not valid".format(scan_order))

    # Get PSG states
    # The HWP angle to produce a desired linear polarization angle for a VLP
    # input is:
    #   HWP = LP/2 + 45
    # For HLP input it is:
    #   HWP = LP/2
    #
    # For RCP and LCP states the HWP is set to 90, so that the fast axis is
    # aligned with VLP input polarization.
    #
    # QWP fast axis is aligned with the HWP FA for all LP states.

    if is_pset_pipo(pset_name):
        psg_states, psa_states = get_nsmp_state_order(pset_name)
        num_psg_states, num_psa_states = get_num_states(pset_name)

        if output_state == 'hlp':
            psa_hwp_ofs = 0
            psa_pol = 0 / 180 * np.pi
        elif output_state == 'vlp':
            psa_hwp_ofs = 45
            psa_pol = 90 / 180 * np.pi

        psg_hwp = np.linspace(0, 180, num_psg_states+1)[:-1]/2 / 180*np.pi
        psg_qwp = np.linspace(0, 180, num_psg_states+1)[:-1] / 180 *np.pi

        psa_hwp = (np.linspace(0, 180, num_psa_states+1)[:-1]/2 + psa_hwp_ofs)/ 180*np.pi

        # For the PSA in PIPO mode the incident state is transformed to
        # the output state after the HWP. The QWP then does nothing.
        psa_qwp = np.zeros_like(psa_hwp)

    elif pset_name == 'shg_nsmp':
        # Nonlinear Stokes-Mueller polarimetry state set for the SHG case
        # 54 states, 9 psg x 6 psa
        psg_states = get_nsmp_state_order(pset_name)[0]
        if psg_states != ['hlp', 'vlp', '+45', '-45', 'rcp', 'lcp', '-22.5', 'rep', 'lep']:
            raise(Exception("Invalid SHG NSMP PSG state order"))

        if input_state == 'vlp':
            #                   HLP  VLP  +45   -45  RCP  LCP  -22.5    REP    LEP
            psg_hwp = np.array([ 45, 90, 67.5, 22.5,  90, 90,  33.75, 11.25, 56.25]) / 180*np.pi
            psg_qwp = np.array([  0, 90,   45,  -45, 135, 45,  -22.5, 0,      -45]) / 180*np.pi

        elif input_state == 'hlp':
            #                   HLP  VLP   +45    -45  RCP  LCP   -22.5    REP    LEP
            psg_hwp = np.array([  0,  45, 22.5, -22.5,   0,   0, -11.25, 33.75, 33.75]) / 180*np.pi
            psg_qwp = np.array([  0,  90,   45,  -45,   45,  -45,  -22.5,    90,    45]) / 180*np.pi

    # == Generate PSA states ==
    if pset_name[0:4] == 'pipo':
        # PIPO PSG and PSA states are generated earlier together
        pass

    elif pset_name == 'shg_nsmp':
        # Nonlinear Stokes-Mueller polarimetry state set for the SHG case
        # 54 states, 9 psg x 6 psa
        psa_states = get_nsmp_state_order('shg_nsmp')[1]
        if psa_states != ['hlp', 'vlp', '+45', '-45', 'rcp', 'lcp']:
            raise(Exception("Invalid SHG NSMP PSA state order"))

        if output_state == 'vlp':
            #          HLP  VLP   +45   -45  RCP  LCP
            psa_hwp = np.array([ 45, 90, 67.5, 22.5,  0,  0 ]) / 180*np.pi
            psa_qwp = np.array([ 90, 90,  90,  90, -45, 45 ]) / 180*np.pi
            psa_pol = 90 / 180 *np.pi
        elif output_state == 'hlp':
            #          HLP  VLP   +45    -45  RCP  LCP
            psa_hwp = np.array([  0, 45, 22.5, -22.5,  0,  0 ]) / 180*np.pi
            psa_qwp = np.array([  0,  0,   0,    0, 45, -45 ]) / 180*np.pi
            psa_pol = 0 / 180 *np.pi

    num_psg = len(psg_states)
    num_psa = len(psa_states)

    # === Verify states ===

    # For the PSG to work correctly it has to transform the input state to the
    # given PSG state at the PSG output
    svec_in = get_stokes_vec(input_state)

    for ind_psg in range(num_psg):
        mmat_hwp = get_mueller_mat('hwp', psg_hwp[ind_psg])
        mmat_qwp = get_mueller_mat('qwp', psg_qwp[ind_psg])
        svec_psg = mmat_qwp.dot(mmat_hwp.dot(svec_in))

        if not tensor_eq(svec_psg, get_stokes_vec(psg_states[ind_psg])):
            print("PSG state {:d} is not {:s}".format(ind_psg, psg_states[ind_psg]))

    # For the PSA to work correctly it has to transform all the given states
    # arriving at the PSA into the detector output states after the PSA
    for ind_psa in range(num_psa):

        svec_in = get_stokes_vec(psa_states[ind_psa])
        mmat_hwp = get_mueller_mat('hwp', psa_hwp[ind_psa])
        mmat_qwp = get_mueller_mat('qwp', psa_qwp[ind_psa])

        if psa_pol is not None:
            mmat_pol = get_mueller_mat('pol', psa_pol)
        else:
            mmat_pol = get_mueller_mat('nop')

        svec_det = mmat_pol.dot(mmat_qwp.dot(mmat_hwp.dot(svec_in)))

        if not tensor_eq(svec_det, get_stokes_vec(output_state)):
            print("PSA state {:d} is not {:s}", ind_psg, psg_states[ind_psg])

    # === Generate state sequence ===

    if scan_order == 'psg_first':
        if with_ref_states:
            num_states = num_psg * num_psa + num_psa
            ref_state_psg_ind = 0
            ref_state_psa_ind = 0
        else:
            num_states = num_psg * num_psa

        seq = np.ndarray(num_states, 6)

        ind_seq = 0
        for ind_psa in range(num_psa):
            for ind_psg in range(num_psg):
                seq[ind_seq, 0] = ind_psg
                seq[ind_seq, 1] = ind_psa
                seq[ind_seq, 2] = psg_hwp[ind_psg]/np.pi*180
                seq[ind_seq, 3] = psg_qwp[ind_psg]/np.pi*180
                seq[ind_seq, 4] = psa_hwp[ind_psa]/np.pi*180
                seq[ind_seq, 5] = psa_qwp[ind_psa]/np.pi*180

                ind_seq += 1

                if with_ref_states and ind_psg == num_psg-1:
                    # Add reference state before switching to next PSA state
                    seq[ind_seq, 0] = 0
                    seq[ind_seq, 1] = 0
                    seq[ind_seq, 2] = psg_hwp[ref_state_psg_ind]/np.pi*180
                    seq[ind_seq, 3] = psg_qwp[ref_state_psg_ind]/np.pi*180
                    seq[ind_seq, 4] = psa_hwp[ref_state_psa_ind]/np.pi*180
                    seq[ind_seq, 5] = psa_qwp[ref_state_psa_ind]/np.pi*180
                    ind_seq += 1

    elif scan_order == 'psa_first':
        if with_ref_states:
            num_states = num_psg * num_psa + num_psg
            ref_state_psg_ind = 0
            ref_state_psa_ind = 0
        else:
            num_states = num_psg * num_psa

        seq = np.ndarray([num_states, 6])

        ind_seq = 0
        for ind_psg in range(num_psg):
            for ind_psa in range(num_psa):
                seq[ind_seq, 0] = ind_psg
                seq[ind_seq, 1] = ind_psa
                seq[ind_seq, 2] = psg_hwp[ind_psg]/np.pi*180
                seq[ind_seq, 3] = psg_qwp[ind_psg]/np.pi*180
                seq[ind_seq, 4] = psa_hwp[ind_psa]/np.pi*180
                seq[ind_seq, 5] = psa_qwp[ind_psa]/np.pi*180

                ind_seq += 1

                if with_ref_states and ind_psa == num_psa-1:
                    # Add reference state before switching to next PSA state
                    seq[ind_seq, 0] = 0
                    seq[ind_seq, 1] = 0
                    seq[ind_seq, 2] = psg_hwp[ref_state_psg_ind]/np.pi*180
                    seq[ind_seq, 3] = psg_qwp[ref_state_psg_ind]/np.pi*180
                    seq[ind_seq, 4] = psa_hwp[ref_state_psa_ind]/np.pi*180
                    seq[ind_seq, 5] = psa_qwp[ref_state_psa_ind]/np.pi*180
                    ind_seq += 1

    # === Write files ===
    if write_files:
        print("Writing 'PolStates.dat' file...")
        np.savetxt('PolStates.dat', seq, delimiter=',\t',
                   fmt=['%d', '%d', '%.2f', '%.2f', '%.2f', '%.2f'])

    if verbosity >= 1:
        print("All done")

    pol_angles = [psg_hwp, psg_qwp, psa_hwp, psa_qwp]

    return seq, pol_angles
