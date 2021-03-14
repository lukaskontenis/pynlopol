"""Fit configuration class.

This module contains the fit configuration class.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2021 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""


class FitConfig:
    """Fit config class."""

    fit_model = None    # Fit model name string
    fitfun_name = None  # Fit function name string

    def __init__(self, fit_model=None, fitfun_name=None):
        """Initialize FitConfig instance."""
        self.fit_model = fit_model
        self.fitfun_name = fitfun_name

    def get_fit_model(self):
        """Get fit model string."""
        return self.fit_model

    def set_fit_model(self, model):
        """Set fit model string."""
        self.fit_model = model

    def get_fitfun_name(self):
        """Get fit function name string."""
        return self.fitfun_name

    def set_fitfun_name(self, name):
        """Get fit function name string."""
        self.fitfun_name = name
