"""Test C6v PIPO map fitting.

Generate a PIPO map for the C6v SHG case and fit a tensor model to it.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2021 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import numpy as np

from lcmicro.polarimetry.nsmp_fit import fit_pipo
from lcmicro.polarimetry import simulate_pipo

symmetry_str = 'c6v'
delta = 12.3/180*np.pi
zzz = 1.56

pipo_arr = simulate_pipo(symmetry_str=symmetry_str,
                         output_type='img', img_sz=[16, 16],
                         delta=delta, zzz=zzz, with_poisson_noise=True)

fit_model = 'c6v'

fit_result = fit_pipo(
    pipo_arr=pipo_arr, fit_model=fit_model,
    plot_progress=False, show_fig=True, use_fit_accel=True)
