"""Fit configuration class.

This module contains the fit configuration class.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2021 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
from lcmicro.polarimetry.fitconfig import FitConfig


class ImgFitConfig(FitConfig):
    """Fit config class."""

    num_row = None
    num_col = None
    mask = None
    mark_thr = None
    max_fit_pts = None

    def get_sz(self):
        """Get image size."""
        return [self.num_row, self.num_col]

    def set_sz(self, sz):
        """Set image size."""
        self.num_row = sz[0]
        self.num_col = sz[1]

    def get_mask(self):
        """Get fit mask."""
        return self.mask

    def set_mask(self, mask):
        """Set mask."""
        self.mask = mask

    def get_mask_thr(self):
        """Get fit mask theshold."""
        return self.mask_thr

    def set_mask_thr(self, thr):
        """Set fit mask theshold."""
        self.mask_thr = thr

    def get_max_fit_pts(self):
        """Get the maximum number of points to fit."""
        return self.max_fit_pts

    def set_max_fit_pts(self, pts):
        """Set the maximum number of points to fit."""
        self.max_fit_pts = pts
