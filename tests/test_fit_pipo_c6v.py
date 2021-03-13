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
pset_name = 'pipo_8x8'
pipo_arr = simulate_pipo(symmetry_str=symmetry_str, delta=delta, zzz=zzz,
                         pset_name=pset_name)

fit_model = 'c6v'

fit_result = fit_pipo(
    pipo_arr=pipo_arr, fit_model=fit_model, fitfun_name='c6v_ag',
    print_results=False, plot_progress=False, show_fig=True)

fit_result.set_ref_pars({'zzz': zzz, 'delta': delta})
fit_result.print(style='table')

fit_result = fit_pipo(
    pipo_arr=pipo_arr, fit_model=fit_model, fitfun_name='nsmpsim',
    print_results=False, plot_progress=False, show_fig=True)

fit_result.set_ref_pars({'zzz': zzz, 'delta': delta})
fit_result.print(style='table')
