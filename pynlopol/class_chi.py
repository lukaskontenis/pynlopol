"""pynlopol - a Python library for nonlinear polarimetry.

Nonlinear susceptibility tensor class.

Note: a MATLAB version of this code is availble as part of NLPS in classChi.m.
This Python version was first translated in 2020.11.21 using NLPS code from
2017.07.12.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# pep8: ignore=E202

import numpy as np


class ClassChi:
    """A class to store the nonlinear susceptibility tensor."""

    # Rank
    rank = None

    # Nonlinear order
    nlorder = None

    # Symemtry string
    symmetry_str = None

    # The full Chi(n) tensor, 3x3x3 for n=2, 3x3x3x3 for n=3
    chi = None

    # Contracted Chi(n) tensor, 3x6 for n=2, 3x10 for n=3
    chid = None

    symbolic = False   # True for a symbolic tensor representation

    primary_axis = None  # Primary axis of the tensor

    def __init__(self, symmetry_str=None, verbosity=0, nlorder=None, **kwargs):
        """Contruct a chi tensor."""
        if symmetry_str is None:
            print("No symmetry information given.")
            symmetry_str = 'nop'

        self.symmetry_str = symmetry_str

        if nlorder is None:
            self.nlorder = 2
            if verbosity >= 1:
                print("Assuming 2-nd order tensor")
        else:
            self.nlorder = nlorder

        self.rank = 2

        if symmetry_str == 'nop':
            # Unity tensor that does nothing
            if self.nlorder == 2:
                self.chi = np.zeros([3, 3, 3])
                self.chi[0, 0, 0] = 1
                self.chi[1, 1, 1] = 1
                self.chi[2, 2, 2] = 1
            elif self.nlorder == 3:
                self.chi = np.zeros([3, 3, 3, 3])
                self.chi[0, 0, 0, 0] = 1
                self.chi[1, 1, 1, 1] = 1
                self.chi[2, 2, 2, 2] = 1
                self.chi[3, 3, 3, 3] = 1

            self.contract()

        elif symmetry_str in ['iso', 'isotropic']:
            # Isotropic tensor
            if self.nlorder == 2:
                # Assuming SHG case, which has no signal for an isotropic
                # sample. For other cases this might not be valid.
                self.chid = np.zeros([4, 6])
            elif self.nlorder == 3:
                self.chid = np.array([
                    [1, 0, 0,   0, 1/3,   0, 1/3,   0,   0,   0],
                    [0, 1, 0, 1/3,   0,   0,   0,   0, 1/3,   0],
                    [0, 0, 1,   0,   0, 1/3,   0, 1/3,   0,   0]])

            self.expand()

        elif symmetry_str in ['d3', 'zcq']:
            # D3 tensor
            # NOTE: In the current implementation, 'd3' and 'zcq' are used
            # interchangeably to denote the formal D3 symmetry class. The z-cut
            # quartz case is actually the D3 tensor, for which the optical axis
            # is along X, rotated so that the optical axis becomes along Z.
            # This should not be an issue, as long as the D3 symmetry is used
            # only in this manner
            if self.nlorder == 2:
                xxx = 1
                xyz = 0

                # The contracted tensor given below is for the case where the
                # beam propagates along the X axis. A notable use of the D3
                # symmtry class if for z-cut quartz where the beam propagates
                # along Z. To obtain the D3 tensor in the z-cut quartz case,
                # the given tensor needs to be rotated by 90 deg about X
                # followed by a 30 deg rotation about Y.
                #
                # Kleinman symmetry is assumed and zxy=xyz
                self.chid = np.array([
                    [xxx, -xxx,   0, xyz,    0,    0],
                    [  0,    0,   0,   0, -xyz, -xxx],
                    [  0,    0,   0,   0,    0,    0]])

                self.expand()

                # Rotate the tensor around X and Y for the zcq case
                self.rotate_x(np.pi/2)
                self.rotate_y(np.pi/6)

        elif symmetry_str == 'c6v':

            if self.nlorder == 2:
                # chi(2) C6v
                # z is along axis of symmetry
                # R = zzz/zxx, Rk = xxz/zxx, Rc = xyz/zxx
                zzz = kwargs.get('zzz', 1.5)
                zxx = kwargs.get('zxx', 1)
                xxz = kwargs.get('xxz', 1)
                xyz = 0

                self.chid = np.array([
                    [  0,   0,   0, xyz,  xxz, 0],
                    [  0,   0,   0, xxz, -xyz, 0],
                    [zxx, zxx, zzz,   0,    0, 0]])

                self.expand()
            elif self.nlorder == 3:
                # chi(3) C6v, THG with Kleinman symmetry
                # Ratio definitions from Tokarz2014
                # z is along axis of symmetry
                # R1 = zzxx/xxxx, R2 = zzzz/xxxx
                zzzz = kwargs.get('zzzz', 2)
                zzxx = kwargs.get('zzxx', 2)
                xxxx = kwargs.get('xxxx', 1)

                self.chid = np.array([
                    #   1,   2,    3,      4,      5,    6,    7,    8,    9, 10
                    [xxxx,   0,    0,      0, xxxx/3,    0, zzxx,    0,    0,  0],
                    [   0,xxxx,    0, xxxx/3,      0,    0,    0,    0, zzxx,  0],
                    [   0,   0, zzzz,      0,      0, zzxx,    0, zzxx,    0,  0]])

                self.expand()
            else:
                raise Exception("No C6v tensor definition for "
                                "order {:}".format(self.nlorder))

        else:
            raise Exception('Unrecognized symmetry string ' + symmetry_str)

    def expand(self):
        """Expand the tensor from contracted form."""
        self.chi = expand_tensor(self.chid)

    def contract(self):
        """Contract the tensor."""
        self.chid = contract_tensor(self.chi)

    def get_ms_contraction(self, primary_axis='z'):
        """Get Masood's contraction of the tensor.

        Get the tensor ZX components in Masood Samim-style contraction. The
        contraction is similar to Boyd's but only for the in-plane tensors.
        The contraction is defined for 1, 2 indices which allows you to pick
        whichever coordinates are your in-plane ones. The default for a beam
        propagating along y is z-primary, but not sure if that is a free choice
        in NSMP.

        It would be great to now have these corrdinate system-level
        discrepancies.

        The z, x components for SHG in x-primary MS notation are:
            xxx, xzz, xxz
            zxx, zzz, zzx

        In z-primary notation:
            zzz, zxx, zzx
            xzz, xxx, xxz

        The z, x components for THG in x-primary MS notation are:
            xxxx, xzzz, xxxz, xxzz
            zxxx, zzzz, zzxx, zzzx

        In z-primary notation:
            zzzz, zxxx, zzzx, zzxx
            xzzz, xxxx, xxzz, xxxz
        """
        if self.nlorder == 2:
            if primary_axis == 'x':
                return np.array([
                    [self.chi[0, 0, 0], self.chi[0, 2, 2], self.chi[0, 0, 2]],
                    [self.chi[2, 0, 0], self.chi[2, 2, 2], self.chi[2, 2, 0]]])
            elif primary_axis == 'z':
                return np.array([
                    [self.chi[2, 2, 2], self.chi[2, 0, 0], self.chi[2, 2, 0]],
                    [self.chi[0, 2, 2], self.chi[0, 0, 0], self.chi[0, 0, 2]]])
        elif self.nlorder == 3:
            if primary_axis == 'x':
                return np.array([
                    [self.chi[0, 0, 0, 0], self.chi[0, 2, 2, 2], self.chi[0, 0, 0, 2], self.chi[0, 0, 2, 2]],
                    [self.chi[2, 0, 0, 0], self.chi[2, 2, 2, 2], self.chi[2, 2, 0, 0], self.chi[2, 2, 2, 0]]])
            elif primary_axis == 'z':
                return np.array([
                    [self.chi[2, 2, 2, 2], self.chi[2, 0, 0, 0], self.chi[2, 2, 2, 0], self.chi[2, 2, 0, 0]],
                    [self.chi[0, 2, 2, 2], self.chi[0, 0, 0, 0], self.chi[0, 0, 2, 2], self.chi[0, 0, 0, 2]]])
        else:
            raise RuntimeError("Unedfined nonlinear order {:d}".format(
                self.nlorder))

    def get_lab_chi(self):
        """Return laboratory-frame chi(n) components.

        See note in nsmp_common.py regarding component 2-nd and 3-rd order
        cases.
        """
        chi_vec = self.get_ms_contraction(primary_axis='z')
        if self.get_nlorder() == 2:
            # Components are returned in NSMP order:
            #   ZXX, ZZZ, ZXZ, XXX, XZZ, XXZ
            return np.array([
                chi_vec[0, 1],
                chi_vec[0, 0],
                chi_vec[0, 2],
                chi_vec[1, 1],
                chi_vec[1, 0],
                chi_vec[1, 2]])
        elif self.get_nlorder() == 3:
            # Components are returned in NSMP order:
            #   ZZZZ, ZXXX, ZZZX, ZZXX, XZZZ, XXXX, XXZZ, XXXZ
            return np.array([
                chi_vec[0, 0],
                chi_vec[0, 1],
                chi_vec[0, 2],
                chi_vec[0, 3],
                chi_vec[1, 0],
                chi_vec[1, 1],
                chi_vec[1, 2],
                chi_vec[1, 3]])

    def get_nlorder(self):
        """Return the nonlinear order for the tensor."""
        return self.nlorder

    def is_valid(self):
        """Check if the tensor is valid."""
        if self.chi is None or self.chid is None:
            return False

        if np.any(np.isnan(self.chi)) or np.any(np.isnan(self.chid)):
            return False

        return True

    def is_symbolic(self):
        """Check if the tensor is symbolic."""
        return False

    def zero_small_values(self):
        mask = np.abs(self.chid) < 1e-10
        self.chid[mask] = 0
        self.expand()

    def rotate_x(self, theta=None):
        """Rotate the tensor around the lab x axis."""
        self._rotate(get_rotation_matrix_labx(theta=theta))

    def rotate_y(self, theta=None):
        """Rotate the tensor around the lab y axis."""
        self._rotate(get_rotation_matrix_laby(theta=theta))

    def rotate_z(self, theta=None):
        """Rotate the tensor around the lab z axis."""
        self._rotate(get_rotation_matrix_labz(theta=theta))

    def _rotate(self, rmat=None):
        """Rotate the tensor using the given rotation matrix."""
        mask_small_rmat_values = True
        if mask_small_rmat_values:
            mask = np.abs(rmat) < 1e-16
            rmat[mask] = 0

        chi_rot = np.zeros_like(self.chi)

        if self.rank == 2:
            for I in range(3):
                for J in range(3):
                    for K in range(3):
                        for i in range(3):
                            for j in range(3):
                                for k in range(3):
                                    chi_rot[I, J, K] += \
                                        rmat[I, i] * rmat[J, j] * \
                                        rmat[K, k] * \
                                        self.chi[i, j, k]

        elif self.rank == 3:
            for I in range(3):
                for J in range(3):
                    for K in range(3):
                        for L in range(3):
                            # For perfomance reasons perform summation over
                            # ijkl separate chi_rot IJKL indexing indexing.
                            # This is trivial, but for some reason e.g.
                            # MATLAB does not guess this optimization and
                            # explicit separate indexing is much faster. Not
                            # sure if this is the case in Python but taking
                            # no risks here.
                            val = chi_rot[I, J, K, L]

                            for i in range(3):
                                # If any of the rotation matrix elements are
                                # zero, skip to next iteration because the
                                # whole contribution to the sum will be zero.
                                if rmat[I, i] == 0:
                                    continue

                                for j in range(3):
                                    if rmat[J, j] == 0:
                                        continue

                                    for k in range(3):
                                        if rmat[K, k] == 0:
                                            continue

                                        for l in range(3):
                                            if rmat[L, l] == 0:
                                                continue
                                            val += \
                                                rmat[I, i] * rmat[J, j] * \
                                                rmat[K, k] * rmat[L, l] * \
                                                self.chi[i, j, k, l]

                            chi_rot[I, J, K, L] = val
        else:
            raise(Exception("Unsupported tensor rank {:d}".format(self.rank)))

        self.chi = chi_rot
        self.contract()

    def rotate(self, delta=None, alpha=None, **kwargs):
        """Rotate the tensor by delta and alpha.

        Perform the rotation by the given in-plane delta and out-of-plane alpha
        angles using proper spherical omega and psi rotation angles.

        Note that when dealing with rotationally-symmetric tesnors the roll
        angle can be ignored which allows an intuitively simpler in-plane and
        out-of-plane rotation to be performed. The cylindrical symmetry (Cinf)
        case is an example of this roll-invariance, and since it applies to
        collagen many sources just assume that the in-plane, out-of-plane
        notation is a general tensor property. It is not. Rotations in the
        general case must be performed using all three proper Euler rotation
        angles.

        To maintain compatibility with the widespread delta-alpha notation this
        function converts the delta and alpha anlges to omega and psi.
        """
        omega, psi = delta_alpha_2_omega_psi(delta, alpha)

        rmat = get_rotation_matrix(omega=omega, psi=psi)
        self._rotate(rmat)

# === Helper functions ===

def delta_alpha_2_omega_psi(delta=None, alpha=None):
    """Convert delta, angle rotation angles to omega, psi.

    The delta, angle corresponds to in-plane, out-of-plane rotation notation,
    whereas omega, psi are the proper Euler rotation angles.
    """
    if alpha is None or alpha == 0:
        omega = delta
        psi = 0
    else:
        omega = np.arccos(np.cos(delta)*np.cos(alpha))
        if np.cos(delta)*np.cos(alpha) == 1:
            psi = 0
        else:
            cos_psi = np.sin(delta)*np.cos(alpha) / \
                        np.sqrt(1-np.cos(delta)**2*np.cos(alpha)**2)
            if cos_psi <= 1:
                psi = np.arccos(cos_psi)
            elif cos_psi - 1 < 1E-6:
                psi = 0
            else:
                raise(Exception("Invalid psi angle"))

    return omega, psi


def get_rotation_matrix(omega=None, psi=None):
    """Get a rotation matrix.

    Get a rotation matrix for Euler rotations. Currently only rotation around the
    proper omega and psi angles is supported. For other cases see the
    GetRotationMatrix.m file in NLPS.

    The solid-body rotation around the omega, psi angles introduces an
    additional psi roll about the Z axis. For cylindrical tensors this is
    irrelevant, but the matrix form with be different.

    The rotation matrices correspond to Serguei Krouglovs 2017.03.03 (V2)
    derivation.
    """
    co = np.cos(omega)
    so = np.sin(omega)
    cp = np.cos(psi)
    sp = np.sin(psi)
    return np.array([
        [co*cp*cp + sp*sp,    -(1-co)*cp*sp, so*cp],
        [   -(1-co)*cp*sp, co*sp*sp + cp*cp, so*sp],
        [          -so*cp,           -so*sp,    co]])

def get_rotation_matrix_labx(theta=0):
    """Get a rotation matrix for rotation around the X axis."""

    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [1, 0, 0],
        [0, ct, -st],
        [0, st,  ct]])

def get_rotation_matrix_laby(theta=0):
    """Get a rotation matrix for rotation around the Y axis."""

    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [ ct,  0, st],
        [  0,  1,  0],
        [-st,  0, ct]])


def get_rotation_matrix_labz(theta=0):
    """Get a rotation matrix for rotation around the Z axis."""

    ct = np.cos(theta)
    st = np.sin(theta)

    return np.array([
        [ ct, -st, 0],
        [ st,  ct, 0],
        [  0,  0,  0]])


def expand_tensor(chid):
    """Expand a tensor from contracted to full form.

    Expand a rank-2 or rank-3 tensor defined in Boyd's 2D contracted notation into
    a full 3D or 4D representation.

    The elements of the nonlinear suspectibility tensor are indexed ijk and ijkl
    for the second-order (rank-2) and third-order (rank-3) cases, respectively.
    Each of the indices goes over the cartesian coordinates xyz for the incoming
    and outgoing polarization. The outgoing polarization is denoted by the first
    index i and the remaining indices are for the incoming polarization. For
    example, the zxx component (i=z, j,k=x) represents the nonlinear interaction
    where two x-polarized laser photons produce one z-polarized SHG photon.

    The conracted notation makes use of the fact that in harmonic-generation the
    two incoming photons are indistinguishable, therefore the jk and jkl indices
    can be freely interchanged significantly reducing the number of unique elements
    required to describe the tensor.

    Using the contracted notation a 3D or a 4D tensor can be written down in a
    convenient 2D matrix which has three rows for the three outgoing polarizations
    and 6 or 10 columns for all possible incoming polarization combinations.

    The SHG contraction prescription in Boyd's 1-based notation is:
        jk    11  22  33  23,32   31,13   12,21
        A     1   2   3   4       5       6

    The THG contraction prescription is:
        jkl   111  222  333  112  122  113  133  223  233  123
        A     1    2    3    4    5    6    7    8    9    10
    """
    # Verify contracted tensor dimensions
    numr, numc = chid.shape
    if numr != 3 or numc not in [6, 10]:
        raise(Exception("Tensor has and invalid contracted dimesions ({:d}, {:d})".format()))

    # Loop over the full tensor ijk indices and assign values from the
    # contracted from.
    #
    # It may be faster to asign values explicitly without loops that only take
    # a few itertions each, but this is likely clearer.

    if numc == 6:
        # SHG case
        chi = np.ndarray([3, 3, 3])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if j == 0 and k == 0:
                        val = chid[i, 0]
                    elif j == 1 and k == 1:
                        val = chid[i, 1]
                    elif j == 2 and k == 2:
                        val = chid[i, 2]
                    elif (j == 1 and k == 2) or (j == 2 and k == 1):
                        val = chid[i, 3]
                    elif (j == 0 and k == 2) or (j == 2 and k == 0):
                        val = chid[i, 4]
                    elif (j == 0 and k == 1) or (j == 1 and k == 0):
                        val = chid[i, 5]

                    chi[i, j, k] = val

    elif numc == 10:
        # THG case
        chi = np.ndarray([3, 3, 3, 3])

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        if j == 0 and k == 0 and l == 0:
                            val = chid[i, 0]
                        elif j == 1 and k == 1 and l == 1:
                            val = chid[i, 1]
                        elif j == 2 and k == 2 and l == 2:
                            val = chid[i, 2]
                        elif (j == 0 and k == 0 and l == 1) or \
                                (j == 0 and k == 1 and l == 0) or \
                                (j == 1 and k == 0 and l == 0):
                            val = chid[i, 3]
                        elif (j == 0 and k == 1 and l == 1) or \
                                (j == 1 and k == 0 and l == 1) or \
                                (j == 1 and k == 1 and l == 0):
                            val = chid[i, 4]
                        elif (j == 0 and k == 0 and l == 2) or \
                                (j == 0 and k == 2 and l == 0) or \
                                (j == 2 and k == 0 and l == 0):
                            val = chid[i, 5]
                        elif (j == 0 and k == 2 and l == 2) or \
                                (j == 2 and k == 0 and l == 2) or \
                                (j == 2 and k == 2 and l == 0):
                            val = chid[i, 6]
                        elif (j == 1 and k == 1 and l == 2) or \
                                (j == 1 and k == 2 and l == 1) or \
                                (j == 2 and k == 1 and l == 1):
                            val = chid[i, 7]
                        elif (j == 1 and k == 2 and l == 2) or \
                                (j == 2 and k == 1 and l == 2) or \
                                (j == 2 and k == 2 and l == 1):
                            val = chid[i, 8]
                        elif (j == 0 and k == 1 and l == 2) or \
                                (j == 0 and k == 2 and l == 1) or \
                                (j == 1 and k == 0 and l == 2) or \
                                (j == 1 and k == 2 and l == 0) or \
                                (j == 2 and k == 0 and l == 1) or \
                                (j == 2 and k == 1 and l == 0):
                            val = chid[i, 9]

                        chi[i, j, k, l] = val

    return chi


def contract_tensor(chi):
    """Contract a tensor from its full form.

    Calculate Boyd's contracted 2D tensor form a full 3D or 4D representation
    of a rank-2 or rank-3 tensor.

    The elements of the nonlinear suspectibility tensor are indexed ijk and ijkl
    for the second-order (rank-2) and third-order (rank-3) cases, respectively.
    Each of the indices goes over the cartesian coordinates xyz for the incoming
    and outgoing polarization. The outgoing polarization is denoted by the first
    index i and the remaining indices are for the incoming polarization. For
    example, the zxx component (i=z, j,k=x) represents the nonlinear interaction
    where two x-polarized laser photons produce one z-polarized SHG photon.

    The conracted notation makes use of the fact that in harmonic-generation the
    two incoming photons are indistinguishable, therefore the jk and jkl indices
    can be freely interchanged significantly reducing the number of unique elements
    required to describe the tensor.

    Using the contracted notation a 3D or a 4D tensor can be written down in a
    convenient 2D matrix which has three rows for the three outgoing polarizations
    and 6 or 10 columns for all possible incoming polarization combinations.

    The SHG contraction prescription in Boyd's 1-based notation is:
        jk    11  22  33  23,32   31,13   12,21
        A     1   2   3   4       5       6

    The THG contraction prescription is:
        jkl   111  222  333  112  122  113  133  223  233  123
        A     1    2    3    4    5    6    7    8    9    10
    """

    check_kleinman_sym = True

    # Verify contracted tensor dimensions
    rank = len(chi.shape) - 1
    if rank < 2 or rank > 3:
        raise(Exception("Rank {:d} tensors are not supported".format(rank)))

    # Loop over the full tensor ijk indices and assign values from the
    # contracted from.
    #
    # It may be faster to asign values explicitly without loops that only take
    # a few itertions each, but this is likely clearer.

    if rank == 2:
        # SHG case
        chid = np.ndarray([3, 6])

        for i in range(3):
            chid[i, 0] = chi[i, 0, 0]
            chid[i, 1] = chi[i, 1, 1]
            chid[i, 2] = chi[i, 2, 2]
            chid[i, 3] = chi[i, 1, 2]
            chid[i, 4] = chi[i, 0, 2]
            chid[i, 5] = chi[i, 0, 1]

            if check_kleinman_sym:
                eps = np.finfo(chi.dtype).eps
                if chi[i, 1, 2] - chi[i, 2, 1] > eps:
                    print("Kleinman symmetry violation between elements 2,3 and 3,2")

                if chi[i, 0, 2] - chi[i, 2, 0] > eps:
                    print("Kleinman symmetry violation between elements 1,3 and 3,1")

                if chi[i, 0, 1] - chi[i, 1, 0] > eps:
                    print("Kleinman symmetry violation between elements 1,2 and 2,1")

    elif rank == 3:
        # THG case
        chid = np.ndarray([3, 10])

        for i in range(3):
            chid[i, 0] = chi[i, 0, 0, 0]
            chid[i, 1] = chi[i, 1, 1, 1]
            chid[i, 2] = chi[i, 2, 2, 2]
            chid[i, 3] = chi[i, 0, 0, 1]
            chid[i, 4] = chi[i, 0, 1, 1]
            chid[i, 5] = chi[i, 0, 0, 2]
            chid[i, 6] = chi[i, 0, 2, 2]
            chid[i, 7] = chi[i, 1, 1, 2]
            chid[i, 8] = chi[i, 1, 2, 2]
            chid[i, 9] = chi[i, 0, 1, 2]

            # TODO: Implement Kleinman symmetry checks

    return chid
