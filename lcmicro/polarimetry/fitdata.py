"""fitdata structure definition and routines.

This module contains functions related to fitdata, which is the output data
format for NSMP fitting functions found in nsmp_fit.py

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np


def get_parmap(fitdata, par_name='x', parind=0):
    """Make a 2D parameter map from fitdata."""
    fit_result = fitdata['result']
    fit_mask = fitdata['mask']
    num_row, num_col = np.shape(fit_mask)

    parmap = np.ndarray([num_row, num_col])
    parmap.fill(np.nan)
    indpx = 0
    for ind_row in range(num_row):
        for ind_col in range(num_col):
            if indpx == fitdata['num_pts_to_fit']:
                break
            if fit_mask[ind_row, ind_col] and fit_result[indpx] is not None:
                fit_result1 = fit_result[indpx]
                if par_name == 'x':
                    parmap[ind_row, ind_col] = fit_result1.fit_result.x[parind]
                elif par_name == 'err':
                    parmap[ind_row, ind_col] = fit_result1.fit_result.fun[parind]
                indpx += 1

    return parmap
