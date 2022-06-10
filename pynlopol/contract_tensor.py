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
    

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np


def contract_tensor(chi):

    check_kleinman_sym = True

    # Verify contracted tensor dimensions
    rank = len(chi.size) - 1
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
                if chi[i, 1, 2] - chi[i, 2, 1] > np.eps:
                    print("Kleinman symmetry violation between elements 2,3 and 3,2")

                if chi[i, 0, 2] - chi[i, 2, 0] > np.eps:
                    print("Kleinman symmetry violation between elements 1,3 and 3,1")

                if chi[i, 0, 1] - chi[i, 1, 0] > np.eps:
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

    return chi
