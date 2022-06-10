"""FitData base class.

This module contains the base class for FitData and ImgFitData.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np

from lkcom.string import get_human_val_str


class _FitData_Base:
    """Fit data base class."""

    result = None
    cfg = None
    par = None
    err = None
    err_type = None
    ref_par = None
    duration = None

    def set_par(self, par):
        """Set fit parameter arrays."""
        self.par = par

    def get_ref_par(self):
        """Get reference parameters."""
        return self.ref_par

    def set_ref_par(self, ref_par):
        """Set reference parameters."""
        self.ref_par = ref_par

    def get_mask(self):
        """Get fit mask."""
        return self.cfg.get_mask()

    def set_mask(self, *args, **kwargs):
        """Set fit mask."""
        self.cfg.set_mask(*args, *kwargs)

    def get_fit_model(self):
        """Get fit model string."""
        return self.cfg.get_fit_model()

    def set_fit_model(self, *args, **kwargs):
        """Return fit model."""
        self.cfg.set_fit_model(*args, **kwargs)

    def get_fitfun_name(self):
        """Get fit function name string."""
        return self.cfg.get_fitfun_name()

    def set_fitfun_name(self, *args, **kwargs):
        """Get fit function name string."""
        self.cfg.set_fitfun_name(*args, **kwargs)

    def get_fit_duration(self):
        """Get fit duration in seconds."""
        return self.duration

    def set_fit_duration(self, duration):
        """Set fit duration in seconds."""
        self.duration = duration

    def get_par_names(self):
        """Get fit model parameter names."""
        fit_model = self.get_fit_model()
        if fit_model == 'zcq':
            return ['ampl', 'delta']
        if fit_model == 'c6v':
            return ['ampl', 'zzz', 'delta']

    def get_par_human_val_str(self, par_name, par_val):
        """Convert a parameter to a human-readable string."""

        if self.get_par_type(par_name) == 'angle':
            par_val = par_val/np.pi*180

        return get_human_val_str(
            par_val, **self.get_par_fmt(par_name)) + self.get_par_unit(par_name)

    def get_par_fmt(self, par_name):
        """Get parameter string format.

        Formats are indexed by canonical parameter names. Format is returned as
        a dictionary of kwargs for get_human_str.
        """
        par_fmt = {'ampl': {'suppress_suffix': 'm'},
                   'zzz': {'num_decimal_places': 3, 'suppress_suffix': 'm'},
                   'delta': {'num_decimal_places': 2, 'suppress_suffix': 'm'}}
        return par_fmt.get(par_name)

    def get_par_type(self, par_name):
        """Get parameter type.

        Types are indexed by canonical parameter names. Type is returned as
        canonical type string.
        """
        par_type = {'ampl': 'gen',
                    'zzz': 'ratio',
                    'delta': 'angle'}
        return par_type.get(par_name)

    def get_par_varstr(self, par_name):
        """Get parameter variable string.

        Strings are indexed by canonincal parameter names.
        """
        par_varstr = {'ampl': 'A',
                      'zzz': 'R',
                      'delta': 'δ'}
        return par_varstr.get(par_name, par_name)

    def get_par_typ_err(self, par_name):
        """Get typical parameter error.

        Errors are indexed by canonical parameter names.
        """
        typical_err = {'zzz': 0.01,
                       'delta': 0.1/180*np.pi}
        return typical_err.get(par_name)

    def get_par_unit(self, par_name):
        """Get parameter unit string.

        Units are indexed by canonical parameter names. The unit string contains
        as a space if necessary.
        """
        unit_str = {'gen': '',
                    'ratio': '',
                    'angle': '°'}
        return unit_str.get(self.get_par_type(par_name))
