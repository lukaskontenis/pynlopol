
"""pynlopol - a Python library for nonlinear polarimetry.

This file contains nonlinear polarimetry unit tests.

The 'dsmp_nsmp_chaos_warning' flag is to keep track whether the tests require
explicit coordinate system changes and other trickery to pass consistency. A
serious overhaul of the DSMP/NSMP theory is required for these problems to go
away. It has been an ongoing issue since 2016. Good luck.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import unittest
import pathlib

import numpy as np

from pynlopol.polarimetry import tensor_eq
from pynlopol.nsmp_sim import simulate_pipo, get_nsm_matrix
from pynlopol.nsmp import get_lab_chi
from pynlopol.gen_pol_state_sequence import gen_pol_state_sequence


class TestPolarimetry(unittest.TestCase):
    """Test polarimetry routines."""

    # pylint: disable=C0111,C0326
    # flake8: noqa

    def test_nsmp_d3(self):
        """Test SHG D3 nonlinear polarimetry."""
        print("Testing SHG D3 nonlinear polarimetry...")

        dsmp_nsmp_chaos_warning = False

        par = {'symmetry_str': 'd3',
               'delta': 0/180*np.pi,
               'trunc_thr': 1E-4}

        # Primary axis has to be set to X. Likely a DSMP legacy thing.
        dsmp_nsmp_chaos_warning = True
        nmmat = get_nsm_matrix(**par, primary_axis='x')

        ref_nmmat = [
            [1.2247,      0,  0, -1,  0, 0, 0,  0, 0],
            [0.4082, 1.1547,  0, -1,  0, 0, 0,  0, 0],
            [     0,      0,  0,  0, -1, 1, 0,  0, 0],
            [     0,      0,  0,  0,  0, 0, 0, -1, 1]]

        self.assertTrue(tensor_eq(nmmat, ref_nmmat, thr=1E-4))

        # Primary axis has to be set to X. Likely a DSMP legacy thing.
        dsmp_nsmp_chaos_warning = True
        pipo_data = simulate_pipo(**par, primary_axis='x')
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

        if dsmp_nsmp_chaos_warning:
            print("WARNING: DSMP/NSMP coordinate system problems encountered. "
                  "DSMP is left-handed X-primary, NSMP is right-handed "
                  "Z-primary. When the two formalisms are mixed, sometimes a "
                  "rotation of delta by 90 degress is sufficient, other times "
                  "lab-frame X/Z index swaps are required. The full "
                  "implications of the mixing are unknown. Good luck.")

    def test_nsmp_c6v(self):
        """Test SHG C6v nonlinear polarimetry."""
        print("Testing SHG C6v nonlinear polarimetry...")

        dsmp_nsmp_chaos_warning = False

        par = {
            'symmetry_str': 'c6v',
            'delta': 0/180*np.pi,
            'trunc_thr': 1E-4}

        labchi = get_lab_chi(**par)
        ref_labchi = [1, 1.5, 0, 0, 0, 1]
        self.assertTrue(tensor_eq(labchi, ref_labchi, thr=1E-4))

        # Primary axis has to be set to X. Likely a DSMP legacy thing.
        dsmp_nsmp_chaos_warning = True
        nmmat = get_nsm_matrix(**par, primary_axis='x')
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

        if dsmp_nsmp_chaos_warning:
            print("WARNING: DSMP/NSMP coordinate system problems encountered. "
                  "DSMP is left-handed X-primary, NSMP is right-handed "
                  "Z-primary. When the two formalisms are mixed, sometimes a "
                  "rotation of delta by 90 degress is sufficient, other times "
                  "lab-frame X/Z index swaps are required. The full "
                  "implications of the mixing are unknown. Good luck.")

        # Stokes measurement array
        # ref_nsmat = np.array([
            # [0.8780    0.3902    1.0000    1.0000    0.4146    0.4146    0.9895    0.3519    0.7073
            # [0.8780    0.3902    0.2195    0.2195   -0.3659   -0.3659    0.5993   -0.0383   -0.0732
            # [0         0    0.9756   -0.9756         0         0   -0.7874         0    0.6899
            # [0         0         0         0    0.1951   -0.1951         0   -0.3498   -0.1380]

        # self.assertTrue(tensor_eq(pipo_data, ref_pipo, thr=1E-3))


    def test_nsmp_c6v_thg(self):
        """Test THG C6v nonlinear polarimetry."""
        print("Testing THG C6v nonlinear polarimetry...")

        par = {
            'symmetry_str': 'c6v',
            'nlorder': 3,
            'zzzz': 7,
            'xxxx': 5,
            'zzxx': 3,
            'delta': 0/180*np.pi,
            'trunc_thr': 1E-4}

        lab_chi = get_lab_chi(**par)

        # ZZZZ, ZXXX, ZZZX, ZZXX, XZZZ, XXXX, XXZZ, XXXZ
        ref_labchi = [7, 0, 0, 3, 0, 5, 3, 0]
        self.assertTrue(tensor_eq(lab_chi, ref_labchi, thr=1E-4))

        nmmat = get_nsm_matrix(**par)

        # The refence nonlinear Mueller matrix is calculated using NLPS with
        # the following command sequence:
        #   SP = SimulateSPIPOData('SampleName', 'THG_C6v', 'zzzz', 7, 'xxxx', 5, 'zzxx', 3)
        #   squueze(SP.GetMuellerMatrix())
        #
        # Note 1: This assumes as Z-primary coordinate system. An indication
        # that the corrdinate system is incorrect is the 0, 3 term which
        # becomes negative when X/Z are swapped.
        ref_nmmat = np.array([
            [0.576, 0.202, 0.286, 0.212,     0, 0.266, 0,     0,     0, 0.372,     0, 0,     0,     0,     0, 0],
            [0.150,-0.043, 0.215, 0.655,     0,-0.266, 0,     0,     0, 0.372,     0, 0,     0,     0,     0, 0],
            [    0,     0,     0,     0, 0.620, 0, 0.159, 0.266, 0.372,     0, 0,     0,     0,     0,     0, 0],
            [    0,     0,     0,     0,     0,     0, 0,     0,     0,     0, 0.620, 0,-0.159,-0.266, 0.372, 0]])

        self.assertTrue(tensor_eq(nmmat/nmmat[0, 0], ref_nmmat/ref_nmmat[0, 0], thr=1E-3))

        pipo_data = simulate_pipo(**par, pset_name='pipo_9x9', duplicate_pipo_states=True)

        # The reference PIPO array is generated using NLPS with the following
        # command sequence:
        #   SP = SimulateSPIPOData('SampleName', 'THG_C6v', 'zzzz', 7, 'xxxx', 5, 'zzxx', 3, 'PolSeqName', 'PIPO_9x9')
        #   A = cell2mat(SP.Data.I)
        #   A/max(A(:))
        #
        # Note 1: The SP.GetPIPOIntensityMap() cannot be used here because it
        # reorders the values in the PIPO array to match the visual output of
        # PIPONATOR, i.e. the PIPO bubble is centered at PSG/PSA 90/90. This is
        # yet another legacy trap comming from NLPS implicitly dealing with
        # PIPONATOR-specific data when it comes to PIPO datasets.
        #
        # Note 2: The 'PIPO_9x9' polarization sequence has to be used, and the
        # 'duplicate_pipo_states' flag has to be passed to 'simulate_pipo' in
        # order to match the input states used in NLPS. It is not clear why
        # NLPS calculates 0 and 180 deg states twice, likely a bug or PIPONATOR
        # support feature.

        ref_pipo = np.array([
            [0.871, 0.807, 0.569, 0.197,     0, 0.197, 0.569, 0.807, 0.871],
            [0.744, 0.989, 0.901, 0.454, 0.065, 0.022, 0.197, 0.443, 0.744],
            [0.436, 0.881, 1.000, 0.641, 0.222, 0.030, 0.004, 0.110, 0.436],
            [0.128, 0.548, 0.807, 0.649, 0.379, 0.217, 0.103, 0.003, 0.128],
            [    0, 0.184, 0.436, 0.474, 0.444, 0.474, 0.436, 0.184,     0],
            [0.128, 0.003, 0.103, 0.217, 0.379, 0.649, 0.807, 0.548, 0.128],
            [0.436, 0.110, 0.004, 0.030, 0.222, 0.641, 1.000, 0.881, 0.436],
            [0.744, 0.443, 0.197, 0.022, 0.065, 0.454, 0.901, 0.989, 0.744],
            [0.871, 0.807, 0.569, 0.197,     0, 0.197, 0.569, 0.807, 0.871]])

        self.assertTrue(tensor_eq(
            pipo_data/np.max(pipo_data), ref_pipo/np.max(ref_pipo), thr=1E-3))


    def test_gen_pol_state_sequence(self, **kwargs):
        """Test polarization state sequence generation."""
        print("Tesing PIPO 8x8 polarization state sequence...""")

        seq = gen_pol_state_sequence(pset_name='pipo_8x8', write_files=False, verbosity=0, **kwargs)[0]
        ref_seq = np.loadtxt(str(pathlib.Path(__file__).parent.absolute()) + '\\pipo_8x8_pol_states.dat', delimiter=',')

        self.assertTrue((seq == ref_seq).all())


if __name__ == '__main__':
    unittest.main(exit=False)
    input("Press any key to close this window.")
