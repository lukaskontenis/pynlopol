"""
=== lcmicro ===

A Python library for nonlinear microscopy and polarimetry.

This file contains polarimetry module unit tests.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import unittest

import numpy as np

from lcmicro.polarimetry import col_vec, tensor_eq, get_eps, get_mueller_mat, get_stokes_vec


class TestPolarimetry(unittest.TestCase):
    # pylint: disable=C0111,C0326

    def test_helpers(self):
        print("Testing helper functions...")
        vec1 = col_vec([1., 2., 3., 4.])
        self.assertTrue(tensor_eq(vec1, vec1))
        self.assertTrue(tensor_eq(vec1, vec1 + get_eps()))
        self.assertFalse(tensor_eq(vec1, vec1 + 2*get_eps()))

    def test_get_stokes_vec(self):
        print("Testing reference Stokes vector values...")
        self.assertTrue(tensor_eq(get_stokes_vec('hlp'), col_vec([1., +1., 0, 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('vlp'), col_vec([1., -1., 0, 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('+45'), col_vec([1., 0, +1., 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('-45'), col_vec([1., 0, -1., 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('rcp'), col_vec([1., 0, 0, +1.])))
        self.assertTrue(tensor_eq(get_stokes_vec('lcp'), col_vec([1., 0, 0, -1.])))

    def test_get_mueller_mat(self):
        print("Testing reference Mueller matrices...")

        test_uni = np.array([
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1]])
        self.assertTrue(tensor_eq(get_mueller_mat('unity'), test_uni))

        test_hwp = np.array([
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0, -1]])
        self.assertTrue(tensor_eq(get_mueller_mat('hwp'), test_hwp))

        test_qwp = np.array([
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  0,  1],
            [ 0,  0, -1,  0]])
        self.assertTrue(tensor_eq(get_mueller_mat('qwp'), test_qwp))

        test_pol = 0.5*np.array([
            [ 1,  1,  0,  0],
            [ 1,  1,  0,  0],
            [ 0,  0,  0,  0],
            [ 0,  0,  0,  0]])
        self.assertTrue(tensor_eq(get_mueller_mat('pol'), test_pol))

if __name__ == '__main__':
    unittest.main()
