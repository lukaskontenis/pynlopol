
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

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import numpy as np

def expand_tensor(chid):
    # Verify contracted tensor dimensions
    numr, numc = chid.size
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
