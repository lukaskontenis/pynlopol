"""Generate a PIPO or NSMP polarization state sequence.

This script will produce a 'PolStates.dat' file with a polarization state
sequence table for a given polarimetry measurement. A polarization measurement
consists of a series of input and output states, which are prepared by a
polarization state generator and analyzer (PSG and PSA), respectively. The
sequence of the measurement is the serial order in which the input and output
states are measured.

Input and output state combinations are unique, but the measurement can also
include reference states, which are identical input-output state combinations
that are periodically visited during the measurement to make sure the signal is
constant. Set 'with_ref_states' to True to enable reference states.

The PSG and PSA each consists of a half- followed by a quarter-wave plate (HWP
and QWP). The PSG HWP is typically preceded by a polarizer to clean up the
input polarization state, and the PSA QWP is followed by polarizer which is
where the polarization states are transformed into intensity modulation that is
measured by the signal detector.

The sequence file contains 6 columns:
    [PSG ID], [PSA ID], [PSG HWP], [PSG QWP], [PSA HWP], [PSA QWP]
    
where PSG and PSA IDs are indices of the unique input and output states and
PSG/PSA HWP/QWP are the orientation angles of the half- and quarter-wave plates
for the PSG and PSA.

Currently, linear PIPO and SHG NSMP state sets are supported by setting
'pset_name' to either 'pipo_NxN' or 'shg_nsmp'. The PIPO sequence can contain
any number of states N. The recommended set is 'pipo_8x8', anything less than
'pipo_6x6' will likely fail to fit.


This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2021 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

from lcmicro.polarimetry import gen_pol_state_sequence, verify_pol_state_sequence

par = {
    'file_name': 'PolStates.dat',
    'pset_name': 'shg_nsmp',  # E.g. 'shg_nsmp', 'pipo_8x8'
    'input_state': 'hlp',
    'output_state': 'hlp',
    'with_ref_states': False
}

gen_pol_state_sequence(**par, write_files=True)
verify_pol_state_sequence(**par)

input("Press any key to close this window")
