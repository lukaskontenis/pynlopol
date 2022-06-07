"""Fit a model to a PIPO map.

Fit a nonlinear tensor model to a PIPO dataset.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os

from lkcom.util import handle_general_exception

from lcmicro.dataio import get_microscopy_data_file_name
from lcmicro.polarimetry.nsmp_fit import fit_pipo

# Image area to crop before fitting:
#   [from_row, to_row, from_col, to_col], in pixels
cropsz = [46, 87, 135, 152]

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
        fit_model='c6v'
        print("Fitting '{:s}' model to dataset '{:s}'".format(fit_model, file_name))

        fit_pipo(
            file_name=file_name, fit_model=fit_model, use_fit_accel=True,
            binsz=None, cropsz=cropsz, max_fit_pts=None,
            show_input=True, show_fig=True, plot_progress=False)

    except Exception:
        handle_general_exception("Fitting failed")

input("Pess any key to close this window...")
