
"""Nonlinear Stokes-Mueller polarimetry (NSMP) figure generation tests.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import unittest
import numpy as np

import lcmicro.polarimetry as pol


SHOW_FIG = False

class TestPolarimetry(unittest.TestCase):
    """Test polarimetry routines."""

    ## pylint: disable=C0111,C0326
    # flake8: noqa

    def test_pipo_map_zcq(self):
        """Test polarimetry figure generation."""
        print("Z-cut quartz PIPO map generation...")
        par = {
            'symmetry_str': 'd3',
            'delta': 0/180*np.pi,
            'trunc_thr': 1E-3}
        pipo_data = pol.simulate_pipo(**par)

        title_str = "Z-cut quartz PIPO map, δ={:.0f}°".format(par.get('delta')/np.pi*180)
        pol.plot_pipo(pipo_data, title_str=title_str, show_fig=SHOW_FIG, export_fig=True, fig_file_name='zcq_pipo')


    def test_pipo_map_collagen(self):
        """Test polarimetry figure generation."""
        print("Collagen C6v PIPO map generation...")
        par = {
            'symmetry_str': 'c6v',
            'delta': 0/180*np.pi,
            'zzz': 1.5,
            'trunc_thr': 1E-3}
        pipo_data = pol.simulate_pipo(**par)

        title_str = "Collagen R={:.2f}".format(par.get('zzz')) + " PIPO map, δ={:.0f}°".format(par.get('delta')/np.pi*180)
        pol.plot_pipo(pipo_data, title_str=title_str, show_fig=SHOW_FIG, export_fig=True, fig_file_name='collagen_pipo')


    # def test_pipo_map_anim_zcq(self):
    #     """Test polarimetry figure generation."""
    #     print("Generating z-cut quartz (D3) PIPO map GIFs...")
    #     pol.make_pipo_animation_delta(sample='zcq')


    # def test_pipo_map_anim_collagen(self):
    #     """Test polarimetry figure generation."""
    #     print("Generating collagen (C6v) PIPO map GIFs...")
    #     pol.make_pipo_animation_delta(sample='collagen')
    #     pol.make_pipo_animation_zzz(sample='collagen')


if __name__ == '__main__':
    unittest.main(exit=False)
    input("Press any key to close this window.")
