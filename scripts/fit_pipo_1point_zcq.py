"""Fit a zcq model to a PIPO map.

Fit a 'zcq' nonlinear SHG tensor model to a PIPO dataset.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os

from lkcom.util import handle_general_exception
from pynlomic.dataio import get_microscopy_data_file_name

from pynlopol.nsmp_fit import fit_pipo

# Image area to crop before fitting:
#   [from_row, to_row, from_col, to_col], in pixels
cropsz = None  # [46, 87, 135, 152], for example

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

        fit_pipo(
            file_name=file_name, fit_model=fit_model, use_fit_accel=False,
            binsz='all', cropsz=cropsz,
            show_input=False, show_fig=True, export_fig=True,
            plot_progress=False)

    except Exception:
        handle_general_exception("Figure generation failed")

input("Pess any key to close this window...")
