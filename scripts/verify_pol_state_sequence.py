"""Verify a PIPO or NSMP polarization state sequence.

Verify a polarization state sequence for PIPO or NSMP polarimetry.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

from lcmicro.polarimetry import verify_pol_state_sequence

par = {
    'file_name': 'PolStates.dat',
    'pset_name': 'shg_nsmp',
    'input_state': 'hlp',
    'output_state': 'hlp',
    'with_ref_states': True
}

verify_pol_state_sequence(**par)

input("Press any key to close this window")
