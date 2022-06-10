
"""Nonlinear Stokes-Mueller polarimetry (NSMP) common routines.

This module contains common NSMP routines, more advanced functions are
in nsmp.py.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import numpy as np


def get_pauli_matrix_arr(nlproc):
    """Return all NSMP Pauli matrices in an array."""
    pmat_arr = np.zeros([2, 2, 4], dtype=complex)
    for ind in range(4):
        pmat_arr[:, :, ind] = get_pauli_matrix(nlproc, ind)

    return pmat_arr


def get_pauli_matrix(nlproc, ind):
    """Get NSMP Pauli matrix.

    This function defines a set of Pauli matrices that are used to express
    the outgoing Stokes vector in terms of the coherency matrix of the electric
    field.

    Get the lowercase tau (Pauli) matrix for the two- or three-photon case.
    Due to the different coordinate system definitions in the DSMP vs TSMP/NSMP
    theories, the matrix definition is different – the element signs in the
    second (ind=1) and fourth (ind=3) matrix are interchanged.

    The matrices for the two-photon case are defined by Eq. (6) on p. 452 in:
        Samim et al., J. Opt. Soc. Am. B 32, 451 (2015)

    The matrices for the three-photon case are defined by Eq. (A1) on p. 8 in:
        Samim et al., Phys. Rev. A 93, 033839 (2016)

    The general NSMP theory has the same matrix form as TSMP.

    An equivalent function in NLPS v2.25 is called GetTauMatrix.m.
    """
    if nlproc == 'shg':
        if ind == 0:
            return np.array([
                [ 1,  0],
                [ 0,  1]])
        elif ind == 1:
            return np.array([
                [-1,  0],
                [ 0,  1]])
        elif ind == 2:
            return np.array([
                [ 0,  1],
                [ 1,  0]])
        elif ind == 3:
            return np.array([
                [ 0,  1j],
                [-1j,  0]])
        else:
            raise(Exception("Undefined tau matrix for t = {:d}".format(ind)))

    elif nlproc == 'thg':
        if ind == 0:
            return np.array([
                [ 1,  0],
                [ 0,  1]])
        elif ind == 1:
            return np.array([
                [ 1,  0],
                [ 0, -1]])
        elif ind == 2:
            return np.array([
                [ 0,  1],
                [ 1,  0]])
        elif ind == 3:
            return np.array([
                [ 0, -1j],
                [ 1j,  0]])
        else:
            raise(Exception("Undefined tau matrix for t = {:d}".format(ind)))

    else:
        raise(Exception("No tau matrix defined for {:s}".format(nlproc)))


def get_gell_mann_matrix_arr(nlproc):
    """Return all NSMP Gell-Mann matrices in an array."""
    if nlproc == 'shg':
        gmmat_arr = np.zeros([3, 3, 9], dtype=complex)
    elif nlproc == 'thg':
        gmmat_arr = np.zeros([3, 3, 16], dtype=complex)
    else:
        raise(Exception("No Gell-Mann matrices defined for {:s}".format(nlproc)))

    for ind in range(gmmat_arr.shape[2]):
        gmmat_arr[:, :, ind] = get_gell_mann_matrix(nlproc, ind+1)

    return gmmat_arr


def get_gell_mann_matrix(nlproc, ind):
    """Get NSMP Gell-Mann matrix.

    This function defines a set of Gell-Mann matrices that are used to express
    the nonlinear Mueller matrix in terms of the susceptibility tensor.
    The 9 matrices of the two-photon case are defined by Eq. (14) on p. 453 of:
        Samim et al., J. Opt. Soc. Am. B 32, 451 (2015)

    The 16 matrices of the three-photon case defined by Eq. (A1) on p. 8 in:
        Samim et al., Phys. Rev. A 93, 033839 (2016)

    An equivalent function in NLPS v2.25 is called GetEtaMatrix.m.
    """
    if nlproc == 'shg':
        if ind == 1:
            return np.array([
                [ 1,  0,  0],
                [ 0,  1,  0],
                [ 0,  0,  1]]) * np.sqrt(2/3)
        elif ind == 2:
            return np.array([
                [ 1,  0,  0],
                [ 0,  1,  0],
                [ 0,  0, -2]]) * np.sqrt(1/3)
        elif ind == 3:
            return np.array([
                [ 1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0,  0]])
        elif ind == 4:
            return np.array([
                [ 0,  1,  0],
                [ 1,  0,  0],
                [ 0,  0,  0]])
        elif ind == 5:
            return np.array([
                [ 0,  0,  0],
                [ 0,  0,  1],
                [ 0,  1,  0]])
        elif ind == 6:
            return np.array([
                [ 0,  0,  1],
                [ 0,  0,  0],
                [ 1,  0,  0]])
        elif ind == 7:
            return np.array([
                [ 0,-1j,  0],
                [1j,  0,  0],
                [ 0,  0,  0]])
        elif ind == 8:
            return np.array([
                [ 0,  0,  0],
                [ 0,  0,-1j],
                [ 0, 1j,  0]])
        elif ind == 9:
            return np.array([
                [ 0,  0,-1j],
                [ 0,  0,  0],
                [1j,  0,  0]])
        else:
            raise(Exception("Undefined eta matrix for N = {:d}".format(ind)))

    elif nlproc == 'thg':
        if ind == 1:
            return np.array([
                [  1,  0,  0,  0],
                [  0,  1,  0,  0],
                [  0,  0,  1,  0],
                [  0,  0,  0,  1]]) * np.sqrt(2)/2
        if ind == 2:
            return np.array([
                [  1,  0,  0,  0],
                [  0,  1,  0,  0],
                [  0,  0,  1,  0],
                [  0,  0,  0, -3]]) * np.sqrt(6)/6
        if ind == 3:
            return np.array([
                [  1,  0,  0,  0],
                [  0,  1,  0,  0],
                [  0,  0, -2,  0],
                [  0,  0,  0,  0]]) * np.sqrt(3)/3
        if ind == 4:
            return np.array([
                [  1,  0,  0,  0],
                [  0, -1,  0,  0],
                [  0,  0,  0,  0],
                [  0,  0,  0,  0]])
        if ind == 5:
            return np.array([
                [  0,  1,  0,  0],
                [  1,  0,  0,  0],
                [  0,  0,  0,  0],
                [  0,  0,  0,  0]])
        if ind == 6:
            return np.array([
                [  0,  0,  0,  0],
                [  0,  0,  1,  0],
                [  0,  1,  0,  0],
                [  0,  0,  0,  0]])
        if ind == 7:
            return np.array([
                [  0,  0,  0,  0],
                [  0,  0,  0,  0],
                [  0,  0,  0,  1],
                [  0,  0,  1,  0]])
        if ind == 8:
            return np.array([
                [  0,  0,  0,  0],
                [  0,  0,  0,  1],
                [  0,  0,  0,  0],
                [  0,  1,  0,  0]])
        if ind == 9:
            return np.array([
                [  0,  0,  1,  0],
                [  0,  0,  0,  0],
                [  1,  0,  0,  0],
                [  0,  0,  0,  0]])
        if ind == 10:
            return np.array([
                [  0,  0,  0,  1],
                [  0,  0,  0,  0],
                [  0,  0,  0,  0],
                [  1,  0,  0,  0]])
        if ind == 11:
            return np.array([
                [  0,-1j,  0,  0],
                [ 1j,  0,  0,  0],
                [  0,  0,  0,  0],
                [  0,  0,  0,  0]])
        if ind == 12:
            return np.array([
                [  0,  0,  0,  0],
                [  0,  0,-1j,  0],
                [  0, 1j,  0,  0],
                [  0,  0,  0,  0]])
        if ind == 13:
            return np.array([
                [  0,  0,  0,  0],
                [  0,  0,  0,  0],
                [  0,  0,  0,-1j],
                [  0,  0, 1j,  0]])
        if ind == 14:
            return np.array([
                [  0,  0,  0,  0],
                [  0,  0,  0,-1j],
                [  0,  0,  0,  0],
                [  0, 1j,  0,  0]])
        if ind == 15:
            return np.array([
                [  0,  0,-1j,  0],
                [  0,  0,  0,  0],
                [ 1j,  0,  0,  0],
                [  0,  0,  0,  0]])
        if ind == 16:
            return np.array([
                [  0,  0,  0,-1j],
                [  0,  0,  0,  0],
                [  0,  0,  0,  0],
                [ 1j,  0,  0,  0]])
        else:
            raise(Exception("Undefined eta matrix for N = {:d}".format(ind)))
    else:
        raise(Exception("No eta matrix defined for {:s}".format(nlproc)))


def get_nsvec(svec, nlord=2):
    """Convert a linear Stokes vector to a nonlinear Stokes vector."""

    s0 = svec[0]
    s1 = svec[1]
    s2 = svec[2]
    s3 = svec[3]

    if nlord == 2:
        nsvec = np.ndarray([9, 1])

        nsvec[0] = np.sqrt(1/6) * (3*s0**2 - s1**2)
        nsvec[1] = np.sqrt(1/12) * (5*s1**2 - 3*s0**2)
        nsvec[2] = -s0*s1
        nsvec[3] = 1/2 *(s2**2 - s3**2)
        nsvec[4] = s2 * (s1 + s0)
        nsvec[5] = -s2 * (s1 - s0)
        nsvec[6] = -s2 * s3
        nsvec[7] = s3 * (s1 + s0)
        nsvec[8] = s3 * (s1 - s0)
    else:
        print("Only 2-nd order NSMP is implemented")
        nsvec = None

    return nsvec


def get_num_states(pset_name):
    """Get the number of PSG and PSA states in the polarization state set."""
    if is_pset_pipo(pset_name):
        ind1 = pset_name.find('_')
        ind2 = pset_name.find('x')
        num_psg_states = int(pset_name[ind1+1:ind2])
        num_psa_states = int(pset_name[ind2+1:])
        num_states = num_psg_states * num_psa_states
        if num_states < 1 or num_states > 1E6:
            print("Only 1 to 1M PIPO states are supported")

    elif pset_name == 'shg_nsmp':
        num_psg_states = 9
        num_psa_states = 6

    elif pset_name == 'thg_nsmp':
        num_psg_states = 16
        num_psa_states = 6

    return num_psg_states, num_psa_states


def get_nsmp_state_order(pset_name):
    """ Get the state order for an NSMP measurement.

    Args:
        pset_name - Polarimetric state set name
    """
    validate_pset_name(pset_name)

    if is_pset_pipo(pset_name):
        num_psg_states, num_psa_states = get_num_states(pset_name)
        psg_states = [str(x) for x in np.linspace(0, 180, num_psg_states+1)[:-1]]
        psa_states = [str(x) for x in np.linspace(0, 180, num_psa_states+1)[:-1]]

    elif pset_name == 'shg_nsmp':
        psg_states = ['hlp', 'vlp', '+45', '-45', 'rcp', 'lcp', '-22.5', 'rep',
                      'lep']
        psa_states = ['hlp', 'vlp', '+45', '-45', 'rcp', 'lcp']

    elif pset_name == 'thg_nsmp':
        psg_states = ['hlp', 'vlp', '+45', '-45', 'rcp', 'lcp', '-22.5', 'rep1',
                      'lep1', '+22.5', '+67.5', 'rep2', 'lep2', 'rep3', 'rep4',
                      'rep5']
        psa_states = ['hlp', 'vlp', '+45', '-45', 'rcp', 'lcp']

    return psg_states, psa_states


def validate_pset_name(pset_name):
    """Make sure the polarization state set name is a valid string."""
    valid_pset_names = {'cars', 'thg_nsmp', 'shg_nsmp'}

    if is_pset_pipo(pset_name):
        num_psg_states, num_psa_states = get_num_states(pset_name)
        if num_psg_states != num_psa_states:
            raise(Exception("Only rectangular PIPO sequences are supported"))
    elif pset_name not in valid_pset_names:
        raise(Exception("Sequence '{:s}' is not valid".format(pset_name)))


def is_pset_pipo(pset_name):
    """Return true if the polarization state set is a PIPO set."""
    return pset_name[:5] == 'pipo_'

