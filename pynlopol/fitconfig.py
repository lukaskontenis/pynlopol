"""Fit configuration class.

A class to contain fit configuration options. These aptios are set once for a
given fit, and the purpose of this class is to store these options for easier
handling.

At this point the options are stored in a dictinary and the class has no added
functionality except some convenience set/get functions for the more common
options. In the future, it maybe useful to have the fit options stored in a
class rather than a dictionary.

Dictionary storage allows standard as well as custom options to be stored and
retrieved without having to write handlers. Some of the more common options
are:
    fit_model - name of the fit model, e.g. 'zcq', 'c6'
    fitfun_name - name of the fit model function

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""


class FitConfig:
    """Fit config class."""

    def __init__(self, fit_model=None, fitfun_name=None):
        """Initialize FitConfig instance."""
        self.opts = {}
        self.opts['fit_model'] = fit_model
        self.opts['fitfun_name'] = fitfun_name

        self.opts['with_hist_prog_update'] = True
        self.opts['hist_prog_update_period'] = 10
        self.opts['fit_smooth_kernel_sz'] = 2

    def get_opt(self, key):
        """Get a fit configuration option."""
        return self.opts.get(key)

    def set_opt(self, key, val):
        """Set a fit configuration option."""
        self.opts[key] = val

    def get_fit_model(self):
        """Get fit model."""
        return self.get_opt('fit_model')

    def set_fit_model(self, model):
        """Set fit model string."""
        self.set_opt('fit_model', model)

    def get_fitfun_name(self):
        """Get fit function name string."""
        return self.get_opt('fitfun_name')

    def set_fitfun_name(self, name):
        """Get fit function name string."""
        self.set_opt('fitfun_name', name)
