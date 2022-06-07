"""Nonlinear Stokes-Mueller polarimetry (NSMP) reference expressions.

This module contains reference NSMP expressions for testing that are derived by
hand and verified externaly.

This script is part of pynolpol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np


def get_ref_c6v_nmmat(zzz=1.5, zxx=1):
    """Get the C6v nonlinear Mueller matrix.

    Formula is from LK Thesis, p. 47, Eq. (3.17). Assuming delta=0.

    This formula has not been verified numerically or symbolically.
    """
    num_col = 9
    num_row = 4
    nmmat = np.ndarray([num_row, num_col])
    nmmat.fill(0)

    nmmat[0, 0] = 1/np.sqrt(6) * (zzz**2 + 2*zxx**2)
    nmmat[1, 0] = 1/np.sqrt(6) * zzz**2

    nmmat[0, 1] = 1/np.sqrt(12) * (zzz**2 + 2*zxx**2)
    nmmat[1, 1] = 1/np.sqrt(12) * (zzz**2 + 3*zxx**2)

    nmmat[0, 2] = -1/2 * (zzz**2 - zxx**2)
    nmmat[1, 2] = -1/2 * (zzz**2 - zxx**2)

    nmmat[0, 3] = zzz*zxx
    nmmat[1, 3] = zzz*zxx

    nmmat[2, 4] = zzz*zxx
    nmmat[2, 5] = zzz*zxx
    nmmat[3, 7] = zzz*zxx
    nmmat[3, 8] = zzz*zxx

    return nmmat
