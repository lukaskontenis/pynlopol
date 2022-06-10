"""Plot PIPO fit results.

Plot PIPO fit results from a fitdata.npy file.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file"""

import sys
import os
import numpy as np

from lkcom.util import handle_general_exception
from pynlomic.proc import load_pipo
from pynlomic.dataio import get_microscopy_data_file_name

from pynlopol.nsmp_fit import plot_pipo_fit_img


print("=== PIPO fit plotter ===")

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
        fitdata = np.load('fitdata.npy', allow_pickle=True).item()
        pipo_arr = load_pipo(file_name, binsz=None)

        plot_pipo_fit_img(fitdata, pipo_arr=pipo_arr)

    except Exception:
        handle_general_exception("Fitting failed")

input("Pess any key to close this window...")
