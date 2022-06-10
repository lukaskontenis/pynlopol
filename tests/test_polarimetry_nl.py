
"""pynlopol - a Python library for nonlinear polarimetry.

This file contains nonlinear polarimetry unit tests.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import unittest
import pathlib

import numpy as np

import pynlopol as pol
from pynlopol.polarimetry import tensor_eq
from pynlopol.nsmp_sim import simulate_pipo


class TestPolarimetry(unittest.TestCase):
    """Test polarimetry routines."""

    # pylint: disable=C0111,C0326
    # flake8: noqa

    def test_nsmp_d3(self):
        """Test D3 nonlinear polarimetry."""
        print("Testing D3 nonlinear polarimetry...")

        par = {'symmetry_str': 'd3',
               'delta': 0/180*np.pi,
               'trunc_thr': 1E-4}

        nmmat = pol.get_nsm_matrix(**par)

        ref_nmmat = [
            [1.2247,      0,  0, -1,  0, 0, 0,  0, 0],
            [0.4082, 1.1547,  0, -1,  0, 0, 0,  0, 0],
            [     0,      0,  0,  0, -1, 1, 0,  0, 0],
            [     0,      0,  0,  0,  0, 0, 0, -1, 1]]

        self.assertTrue(tensor_eq(nmmat, ref_nmmat, thr=1E-4))

        pipo_data = simulate_pipo(**par)
        ref_pipo = np.array([
            [1.   , 0.5  , 0.   , 0.5  , 1.   , 0.5  , 0.   , 0.5  ],
            [0.854, 0.146, 0.146, 0.854, 0.854, 0.146, 0.146, 0.854],
            [0.5  , 0.   , 0.5  , 1.   , 0.5  , 0.   , 0.5  , 1.   ],
            [0.146, 0.146, 0.854, 0.854, 0.146, 0.146, 0.854, 0.854],
            [0.   , 0.5  , 1.   , 0.5  , 0.   , 0.5  , 1.   , 0.5  ],
            [0.146, 0.854, 0.854, 0.146, 0.146, 0.854, 0.854, 0.146],
            [0.5  , 1.   , 0.5  , 0.   , 0.5  , 1.   , 0.5  , 0.   ],
            [0.854, 0.854, 0.146, 0.146, 0.854, 0.854, 0.146, 0.146]])

        self.assertTrue(tensor_eq(pipo_data, ref_pipo, thr=1E-3))

    def test_nsmp_c6v(self):
        """Test C6v nonlinear polarimetry."""
        print("Testing C6v nonlinear polarimetry...")

        par = {
            'symmetry_str': 'c6v',
            'delta': 0/180*np.pi,
            'trunc_thr': 1E-4}

        labchi = pol.get_lab_chi(**par)
        ref_labchi = [1, 1.5, 0, 0, 0, 1]
        self.assertTrue(tensor_eq(labchi, ref_labchi, thr=1E-4))

        nmmat = pol.get_nsm_matrix(**par)
        ref_nmmat = [
            [1.7351, 0.3608, -0.625, 1.5,   0, 0, 0,   0, 0],
            [0.9186, 1.5155, -0.625, 1.5,   0, 0, 0,   0, 0],
            [     0,      0,      0,   0, 1.5, 1, 0,   0, 0],
            [     0,      0,      0,   0,   0, 0, 0, 1.5, 1]]

        self.assertTrue(tensor_eq(nmmat, ref_nmmat, thr=1E-4))

        pipo_data = simulate_pipo(**par)

        # This array was validated against AG C6 PIPO formula and NLPS v2.28 on
        # 2021.03.11
        ref_pipo = np.array([
            [2.25 , 2.036, 1.562, 1.152, 1.   , 1.152, 1.562, 2.036],
            [1.92 , 2.524, 2.364, 1.593, 0.854, 0.52 , 0.596, 1.097],
            [1.125, 2.277, 2.531, 1.585, 0.5  , 0.067, 0.031, 0.259],
            [0.33 , 1.438, 1.966, 1.132, 0.146, 0.059, 0.198, 0.012],
            [0.   , 0.5  , 1.   , 0.5  , 0.   , 0.5  , 1.   , 0.5  ],
            [0.33 , 0.012, 0.198, 0.059, 0.146, 1.132, 1.966, 1.438],
            [1.125, 0.259, 0.031, 0.067, 0.5  , 1.585, 2.531, 2.277],
            [1.92 , 1.097, 0.596, 0.52 , 0.854, 1.593, 2.364, 2.524]])

        # This array was validated against AG C6 PIPO formula and NLPS v2.28 on
        # 2021.03.11
        ref_pipo_90ofs = np.array([
            [0.   , 0.5  , 1.   , 0.5  , 0.   , 0.5  , 1.   , 0.5  ],
            [0.146, 1.132, 1.966, 1.438, 0.33 , 0.012, 0.198, 0.059],
            [0.5  , 1.585, 2.531, 2.277, 1.125, 0.259, 0.031, 0.067],
            [0.854, 1.593, 2.364, 2.524, 1.92 , 1.097, 0.596, 0.52 ],
            [1.   , 1.152, 1.563, 2.036, 2.25 , 2.036, 1.562, 1.152],
            [0.854, 0.52 , 0.596, 1.097, 1.92 , 2.524, 2.364, 1.593],
            [0.5  , 0.067, 0.031, 0.259, 1.125, 2.277, 2.531, 1.585],
            [0.146, 0.059, 0.198, 0.012, 0.33 , 1.438, 1.966, 1.132]])

        # Simulated PIPO instensity scales with ratio and can also be influenced
        # by amplitude normalization. It is likely safe to compare normalized
        # PIPO arrays.
        allow_free_pipo_amplitude = True
        if allow_free_pipo_amplitude:
            test = tensor_eq(pipo_data/np.max(pipo_data),
                             ref_pipo/np.max(ref_pipo), thr=1E-3)
            test_90ofs = tensor_eq(pipo_data/np.max(pipo_data),
                                   ref_pipo_90ofs/np.max(ref_pipo_90ofs),
                                   thr=1E-3)
        else:
            test = tensor_eq(pipo_data, ref_pipo, thr=1E-3)
            test_90ofs = tensor_eq(pipo_data, ref_pipo_90ofs, thr=1E-3)

        if test_90ofs:
            print("WARNING: PIPO map test shows a 90° delta offset. This may "
                  "indicate a primary X/Z axis inconsistency due to a mixing "
                  "of DSMP/TSMP/NSMP frameworks.")

        self.assertTrue(test or test_90ofs)

        # Stokes measurement array
        # ref_nsmat = np.array([
            # [0.8780    0.3902    1.0000    1.0000    0.4146    0.4146    0.9895    0.3519    0.7073
            # [0.8780    0.3902    0.2195    0.2195   -0.3659   -0.3659    0.5993   -0.0383   -0.0732
            # [0         0    0.9756   -0.9756         0         0   -0.7874         0    0.6899
            # [0         0         0         0    0.1951   -0.1951         0   -0.3498   -0.1380]

        # self.assertTrue(tensor_eq(pipo_data, ref_pipo, thr=1E-3))

    def test_gen_pol_state_sequence(self):
        """Test polarization state sequence generation."""
        print("Tesing PIPO 8x8 polarization state sequence...""")

        seq = pol.gen_pol_state_sequence(pset_name='pipo_8x8', write_files=False, verbosity=0)[0]
        ref_seq = np.loadtxt(str(pathlib.Path(__file__).parent.absolute()) + '\\pipo_8x8_pol_states.dat', delimiter=',')

        self.assertTrue((seq == ref_seq).all())


if __name__ == '__main__':
    unittest.main(exit=False)
    input("Press any key to close this window.")
