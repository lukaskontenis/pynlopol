
"""pynlopol - a Python library for nonlinear polarimetry.

This file contains polarimetry fitting tests.

Copyright 2015-2023 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import unittest
import pathlib

import numpy as np

import pynlopol as pol
from pynlopol.polarimetry import tensor_eq
from pynlopol.nsmp_sim import simulate_pipo
from pynlopol.nsmp_fit import fit_pipo


class TestPolarimetryFit(unittest.TestCase):
    """Test polarimetry fit routines."""

    # pylint: disable=C0111,C0326
    # flake8: noqa

    def test_c6v_pipo_fit(self):
        """Test single-point C6v PIPO fitting."""
        print("Testing C6v single-point PIPO fitting...")

        symmetry_str = 'c6v'
        delta = 12.3/180*np.pi
        zzz = 1.56
        pipo_arr = simulate_pipo(symmetry_str=symmetry_str,
                                 delta=delta, zzz=zzz)

        fit_model = 'c6v'
        fitfun_names = ['c6v_ag', 'nsmpsim']

        for fitfun_name in fitfun_names:
            fit_result = fit_pipo(ask_before_overwrite=False,
                pipo_arr=pipo_arr, fit_model=fit_model, fitfun_name=fitfun_name,
                print_results=False, plot_progress=False, show_fig=False)

            fit_result.set_ref_par({'zzz': zzz, 'delta': delta})
            fit_result.print()
            if fit_result.test_against_ref():
                print(fit_model + ' fitting using ' + fitfun_name +
                      ' fit function verified')
            else:
                print(fit_model + ' fitting using ' + fitfun_name +
                      ' fit function FAILED')
                self.assertTrue(False)

    def test_c6v_pipo_img_fit(self):
        """Test C6v image PIPO fitting."""
        print("Testing C6v image PIPO fitting...")
        symmetry_str = 'c6v'
        delta = 12.3/180*np.pi
        zzz = 1.56

        pipo_arr = simulate_pipo(symmetry_str=symmetry_str,
                                output_type='img', img_sz=[4, 4],
                                delta=delta, zzz=zzz, with_poisson_noise=False)

        fit_model = 'c6v'

        fit_result = fit_pipo(
            pipo_arr=pipo_arr, fit_model=fit_model, ask_before_overwrite=False,
            plot_progress=False, show_fig=False)

        fit_result.set_ref_par({'zzz': zzz, 'delta': delta})
        if fit_result.test_against_ref():
            print(fit_model + ' image PIPO fitting verified')
        else:
            print(fit_model + ' image PIPO fitting FAILED')
            self.assertTrue(False)

    def test_zcq_pipo_fit(self):
        """Test ZCQ nonlinear polarimetry."""
        print("Testing z-cut quartz fitting...")

        symmetry_str = 'zcq'
        delta = 12.3/180*np.pi
        pipo_arr = simulate_pipo(symmetry_str=symmetry_str, delta=delta)

        fit_model = 'zcq'
        fit_result = fit_pipo(ask_before_overwrite=False,
            pipo_arr=pipo_arr, fit_model=fit_model,
            print_results=False, plot_progress=False, show_fig=False)

        fit_result.set_ref_par({'delta': delta})
        fit_result.print()
        fit_result.test_against_ref()

        if fit_result.test_against_ref():
            print(fit_model + ' fitting verified')
        else:
            print(fit_model + ' fitting FAILED')
            self.assertTrue(False)

    def test_thg_c6v_pipo_fit(self):
        """Test single-point THG C6v PIPO fitting."""
        print("Testing THG C6v single-point PIPO fitting...")

        # THG C6v
        sim_par = {
            'sample_name': 'c6v',
            'symmetry_str': 'c6v',
            'nlorder': 3,
            'delta': 10/180*3.14,
            'zzzz': 10,
            'xxxx': 13,
            'zzxx': 2.5,
            'pset_name': 'pipo_8x8',
            'output_type': '1point'
        }

        pipo_data = simulate_pipo(**sim_par)

        fit_model = 'c6v'
        fitfun_names = ['c6v_ag', 'nsmpsim']

        for fitfun_name in fitfun_names:
            fit_result = fit_pipo(ask_before_overwrite=False,
                pipo_arr=pipo_arr, fit_model=fit_model, fitfun_name=fitfun_name,
                print_results=False, plot_progress=False, show_fig=False)

            fit_result.set_ref_par({'zzz': zzz, 'delta': delta})
            fit_result.print()
            if fit_result.test_against_ref():
                print(fit_model + ' fitting using ' + fitfun_name +
                      ' fit function verified')
            else:
                print(fit_model + ' fitting using ' + fitfun_name +
                      ' fit function FAILED')
                self.assertTrue(False)

    

if __name__ == '__main__':
    unittest.main(exit=False)
    input("Press any key to close this window.")
