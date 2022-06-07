"""Plot PIPO fit results.

Plot PIPO fit results from a fitdata.npy file.

This script is part of pynolpol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file"""

import numpy as np

from pynolmic.proc import load_pipo
from pynolmic.dataio import get_microscopy_data_file_name

from pynolpol.nsmp_fit import plot_pipo_fit_img


fitdata = np.load('fitdata.npy', allow_pickle=True).item()

file_name = get_microscopy_data_file_name()
pipo_arr = load_pipo(file_name, binsz=None)

plot_pipo_fit_img(fitdata, pipo_arr=pipo_arr)
