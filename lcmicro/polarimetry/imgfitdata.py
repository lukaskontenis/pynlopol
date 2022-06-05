"""Image fit data class.

This module contains the image fit data class.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2021 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np

from lklib.string import get_human_val_str, arr_summary_str

from lcmicro.polarimetry.imgfitconfig import ImgFitConfig
from lcmicro.polarimetry.fitdata_base import _FitData_Base


class ImgFitData(_FitData_Base):
    """Image fit data class."""

    def __init__(self, result=None, cfg=None):
        """Initialize image fit data."""
        self.result = result
        if cfg is not None:
            self.cfg = cfg
        else:
            self.cfg = ImgFitConfig()

    def get_par(self):
        """Get fit parameter arrays.

        If the arrays don't exist, construc them from the fit data.
        """
        if self.par is not None:
            return self.par

        self.par = {}

        par_names = self.get_par_names()
        [num_row, num_col] = self.get_sz()
        max_pts_to_fig = self.get_max_fit_pts()
        mask = self.cfg.get_mask()
        for par_name in par_names:
            self.par[par_name] = np.ndarray([num_row, num_col])
            self.par[par_name].fill(np.nan)
            indpx = 0
            for ind_row in range(num_row):
                for ind_col in range(num_col):
                    if indpx == max_pts_to_fig:
                        break
                    if mask[ind_row, ind_col] and indpx < len(self.result) and self.result[indpx] is not None:
                        result1 = self.result[indpx]
                        self.par[par_name][ind_row, ind_col] = result1.get_par()[par_name]
                        indpx += 1

        return self.par

    def get_fit_err(self):
        """Get fit error."""
        if self.err is not None:
            return self.err

        self.err = self.result.fun[0]

        return self.err

    def set_fit_err(self, err, type=None):
        """Set fit error."""
        self.err = err
        self.err_type = type

    def set_mask(self, *args, **kwargs):
        """Set fit mask."""
        self.cfg.set_mask(*args, **kwargs)

        if self.par is not None:
            mask = self.get_mask()
            par_names = self.par.keys()
            for par_name in par_names:
                par_arr = self.par[par_name]
                par_arr[mask] = np.nan
                self.par[par_name] = par_arr

    def get_max_fit_pts(self):
        """Get the maximum number of points to fit."""
        return self.cfg.get_max_fit_pts()

    def get_num_pts_to_fit(self):
        """Get the number of points to fit."""
        total_pts = np.prod(self.cfg.get_sz())

        max_fit_pts = self.get_max_fit_pts()
        if max_fit_pts is None:
            max_fit_pts = np.Inf

        mask = self.cfg.get_mask()
        if mask is None:
            mask_pts = np.Inf
        else:
            mask_pts = np.sum(mask)

        return np.min([total_pts, max_fit_pts, mask_pts])

    def get_sz(self):
        """Get image size."""
        return self.cfg.get_sz()

    def print(self):
        """Print a PIPO image fit result summary for an image."""
        print("=== Fit results ===")

        print('Fit model: {:s}'.format(self.cfg.get_fit_model()))
        fitfun_name = self.cfg.get_fitfun_name()
        if fitfun_name is not None:
            print('Fit function: ' + fitfun_name)
        print('Duration: ' + get_human_val_str(self.duration, is_time=True))

        mask = self.cfg.mask
        num_row, num_col = self.cfg.get_sz()
        num_px = num_row*num_col
        num_pts_to_fit = self.get_num_pts_to_fit()
        print("Image size: {:d}x{:d}".format(num_row, num_col))
        print("Fit threshold: {:d} c.".format(self.cfg.get_mask_thr()))
        num_thr = np.sum(mask)
        print("Number of px. above threshold: {:d}, {:.0f}%".format(
            num_thr, num_thr/num_px*100))

        if num_pts_to_fit < num_thr:
            print("Number of px. fitted: {:d}, {:.0f}%".format(
                num_pts_to_fit, num_pts_to_fit/num_px*100))

        par = self.get_par()
        ampl = par.get('ampl')
        delta = par.get('delta')
        zzz = par.get('zzz')

        err = self.err

        print('Parameters:')
        print('\tA: ' + arr_summary_str(ampl, suppress_suffix='m'))
        print('\tδ (°): ' + arr_summary_str(delta/np.pi*180, num_sig_fig=3))
        if zzz is not None:
            print('\tzzz: ' + arr_summary_str(
                zzz, num_sig_fig=3, suppress_suffix='m'))

        if err is not None:
            print('RMS residual error: ' + arr_summary_str(err))
        else:
            print('No fit error data')
        print('\n')

    def test_against_ref(self):
        """Test fit parameters against reference parameters."""
        ref_par = self.get_ref_par()
        if ref_par is None:
            print("Cannot test without ref parameters")
            return False

        par = self.get_par()

        par_names = ref_par.keys()
        for par_name in par_names:
            par_val = par.get(par_name)
            ref_val = ref_par.get(par_name)
            if np.any(abs(par_val - ref_val) > self.get_par_typ_err(par_name)):
                return False

        return True
