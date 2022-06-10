
"""pynlopol - a Python library for nonlinear polarimetry.

Linear polarimetry tests.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import unittest

from pynlopol.polarimetry import col_vec, tensor_eq, get_eps, get_stokes_vec


class TestPolarimetry(unittest.TestCase):
    """Test polarimetry routines."""

    # pylint: disable=C0111,C0326
    # flake8: noqa

    def test_helpers(self):
        """Test helper functions."""
        print("Testing helper functions...")
        vec1 = col_vec([1., 2., 3., 4.])
        self.assertTrue(tensor_eq(vec1, vec1))
        self.assertTrue(tensor_eq(vec1, vec1 + get_eps()))
        self.assertFalse(tensor_eq(vec1, vec1 + 2*get_eps()))

    def test_get_stokes_vec(self):
        """Test Stokes vector constructors."""
        print("Testing reference Stokes vector values...")
        self.assertTrue(tensor_eq(get_stokes_vec('hlp'), col_vec([1., +1., 0, 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('vlp'), col_vec([1., -1., 0, 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('+45'), col_vec([1., 0, +1., 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('-45'), col_vec([1., 0, -1., 0])))
        self.assertTrue(tensor_eq(get_stokes_vec('rcp'), col_vec([1., 0, 0, +1.])))
        self.assertTrue(tensor_eq(get_stokes_vec('lcp'), col_vec([1., 0, 0, -1.])))

if __name__ == '__main__':
    print("=== Linear polarimetry tests ===")
    unittest.main(exit=False)
    input("Press any key to close this window.")
