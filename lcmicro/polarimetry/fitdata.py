"""fitdata structure definition and routines.

This module contains functions related to fitdata, which is the output data
format for NSMP fitting functions found in nsmp_fit.py

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np
from tabulate import tabulate

from lkcom.util import unwrap_angle
from lkcom.string import get_human_val_str

from lcmicro.polarimetry.fitconfig import FitConfig
from lcmicro.polarimetry.fitdata_base import _FitData_Base


class FitData(_FitData_Base):
    """Fit data class."""

    def __init__(self, result=None, cfg=None, duration=None):
        """Initialize FitData instance."""
        self.result = result
        self.duration = duration
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = FitConfig()

    def get_par(self):
        """Get fit parameters."""
        if self.par is not None:
            return self.par

        fit_model = self.get_fit_model()
        if fit_model == 'zcq':
            ampl = self.result.x[0]
            delta = self.result.x[1]
            delta_period = 60/180*np.pi
            delta = unwrap_angle(delta, period=delta_period)
            self.par = {'ampl': ampl, 'delta': delta}
        elif fit_model == 'c6v':
            ampl = self.result.x[0]
            delta = self.result.x[1]
            zzz = self.result.x[2]
            delta_period = 180/180*np.pi
            delta = unwrap_angle(delta, period=delta_period)
            self.par = {'ampl': ampl, 'zzz': zzz, 'delta': delta}
        else:
            print("Cannot handle '{:s}' model".format(fit_model))

        return self.par

    def get_fit_err(self):
        """Get fit error."""
        return self.result.fun[0]

    def is_fit_success(self):
        """Return fit success/fail result."""
        return self.result.success

    def print(self):
        """Print PIPO fit results for a single point."""
        par = self.get_par()
        print("=== PIPO fit results ===")

        err = self.get_fit_err()

        par_names = list(par.keys())
        par_str = []
        for ind, par_val in enumerate(par.values()):
            par_name = par_names[ind]
            if self.get_par_type(par_name) == 'angle':
                par_val = par_val/np.pi*180
            par_str.append(get_human_val_str(par_val, **self.get_par_fmt(par_name)))

        err_str = get_human_val_str(err)

        par_rsd = []

        print('Fit model: {:s}'.format(self.get_fit_model()))
        print('Fit function: {:s}'.format(self.get_fitfun_name()))
        print('Duration: ' + get_human_val_str(self.duration, is_time=True))

        headers = ['Parameter', 'Fit', 'Ref', 'Err', 'P/F']
        table_rows = []
        for ind, par_name in enumerate(par_names):
            table_row = []
            table_row.append(self.get_par_varstr(par_name))
            table_row.append(self.get_par_human_val_str(par_name, par[par_name]))
            ref_par = self.get_ref_par()
            if ref_par is not None:
                ref_par1 = ref_par.get(par_name)
                if ref_par1 is not None:
                    par_err = par[par_name] - ref_par1
                    # par_rsd.append(np.abs(par_err)/typical_err[par_name])
                    table_row.append(self.get_par_human_val_str(par_name, ref_par1))
                    table_row.append(self.get_par_human_val_str(par_name, par_err))
                    if np.abs(par_err) < self.get_par_typ_err(par_name):
                        table_row.append('PASS')
                    else:
                        table_row.append('FAIL')

            table_rows.append(table_row)

        print(tabulate(table_rows, headers=headers))
        print("\n")

        # Calculate average parameter RSD
        # TODO: This needs work. Neither is it RSD, nor is it that useful to
        # compare fit error to the expected error, which results in e.g. a
        # relative error of 20%.
        if len(par_rsd) > 0:
            avg_par_rsd = np.mean(par_rsd)
            print('Average parameter RSD: {:.1f}%'.format(avg_par_rsd*100))
            if avg_par_rsd > 0.1:
                print('Fit is not very good, off by 10%')
            elif avg_par_rsd < 0.001:
                print('Fit is perfect, off by less than 0.1%')

    def test_against_ref(self):
        """Test fit parameters agains reference parameters."""
        ref_par = self.get_ref_par()
        if ref_par is None:
            print("Cannot test without ref parameters")
            return False

        par = self.get_par()

        par_names = ref_par.keys()
        for par_name in par_names:
            par_val = par.get(par_name)
            ref_val = par.get(par_name)
            if abs(par_val - ref_val) > self.get_par_typ_err(par_name):
                return False

        return True
