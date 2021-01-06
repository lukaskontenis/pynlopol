"""Fit a model to a PIPO map.

Fit a 'zcq' nonlinear SHG tensor model to a PIPO dataset.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os

from lklib.fileread import list_files_with_extension
from lklib.util import handle_general_exception

from lcmicro.dataio import get_microscopy_data_file_name
from lcmicro.polarimetry.nsmp_fit import fit_pipo


print("=== PIPO fitter ===")

file_name = None
num_args = len(sys.argv)
if num_args < 2:
    file_name = get_microscopy_data_file_name()
else:
    file_name = sys.argv[1]

if file_name is None:
    print("No input provided. Specify a file name using:")
    print("\t" + os.path.basename(__file__) + " scan.dat")
    print("\nOr drag a dat file on the script icon.\n")
else:
    try:
        fit_model='zcq'
        print("Fitting '{:s}' model to dataset '{:s}'".format(fit_model, file_name))
        fit_pipo(file_name=file_name, fit_model=fit_model, plot_progress=False)

    except Exception:
        handle_general_exception("Figure generation failed")

input("Pess any key to close this window...")
