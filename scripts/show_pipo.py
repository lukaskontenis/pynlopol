"""Show a PIPO map.

Display a polarization-in, polarization-out (PIPO) dataset as a 2D map where
rows and columns correspond to the incoming and outgoing polarization,
respectively.

The map shows one pixel per each input-output state combination, dataset images
are summed into a single value.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import sys
import os

from lkcom.dataio import list_files_with_extension
from lkcom.util import handle_general_exception

from lcmicro.polarimetry.report import make_pipo_fig

print("=== lcmicro ===")
print("Generating PIPO map...")

file_name = None
num_args = len(sys.argv)
if num_args < 2:
    file_names = list_files_with_extension(ext='dat')
    if len(file_names) == 1:
        file_name = os.path.basename(file_names[0])
        print("Found a single dat file '{:s}s', loading it".format(file_name))
    else:
        print("More than one dat file found, specify which to use:")
else:
    file_name = sys.argv[1]

if file_name is None:
    print("No input provided. Specify a file name using:")
    print("\t" + os.path.basename(__file__) + " scan.dat")
    print("\nOr drag a dat file on the script icon.\n")
else:
    try:
        make_pipo_fig(file_name)

    except Exception:
        handle_general_exception("Figure generation failed")

input("Pess any key to close this window...")
