
"""Nonlinear Stokes-Mueller polarimetry (NSMP).

This file contains the verify_pol_state_sequence function to verify
polarization state sequences for nonlinear PIPO and NSMP measurements.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import numpy as np

from lcmicro.polarimetry import gen_pol_state_sequence, get_stokes_vec, \
    tensor_eq, get_nsmp_state_order, get_psgpsa_vector


def verify_pol_state_sequence(
        file_name='PolStates.dat', test_seq=None,
        pset_name='shg_nsmp', input_state='hlp', output_state='hlp',
        with_ref_states=False, **kwargs):
    """Verify polarization state sequence.

    Check that a polarization state sequence is correct, taking the state
    angles, state order, reference states, and input/output states into
    account.

    The test sequence can be provided as an array (test_seq) or in a
    file (file_name).

    Verbosity levels:
        0 - nothing is printed
        1 - a single verdict message is printed
        2 - state errors and verdict are printed

    Args:
        file_name (str) - test sequence file name
        test_seq (arr) - test sequence
        pset_name (str) - name of sequence
        input_state (str) - input state name (hlp/vlp)
        output_state (str) - output state name (hlp/vlp)
        with_ref_states (bool) - True if sequence includes reference states

    Returns:
        True if sequence is valid
    """
    # Verbosity level
    vlvl = kwargs.get('vlvl', 2)

    if vlvl >= 2:
        print("Veriying polarization state sequence...")
        if file_name is not None:
            print("State file: {:s}".format(file_name))
        print("Seqeunce name: {:s}".format(pset_name))
        print("Input state: {:s}".format(input_state))
        print("Output state: {:s}".format(output_state))
        print("Reference states: {:b}".format(with_ref_states))

    # Read state sequence and get PSG and PSA HWP and QWP angles
    test_seq = np.loadtxt(file_name, delimiter=',')

    test_num_col = np.shape(test_seq)[1]
    if test_num_col == 6:
        # Assuming test sequence has sequence ID in first column
        test_angles = test_seq[:, 2:]
    elif test_num_col == 5:
        # Assuming test sequence has sequence ID in first column
        test_angles = test_seq[:, 1:]
    else:
        print("Test sequence has an unsupported number of columns ({:d}), should be 5 or 6".format(test_num_col))
        return False

    # Generate the true sequence
    true_seq = gen_pol_state_sequence(
        pset_name=pset_name, with_ref_states=with_ref_states,
        input_state=input_state, output_state=output_state, vlvl=0)[0]

    # Assuming the true sequence has PSG and PSA IDs in the first two columns
    # and angles in the remaining four
    true_angles = true_seq[:, 2:]

    # Sequence shape checks
    test_shape = np.shape(test_angles)
    ref_shape = np.shape(true_angles)

    # Check that the sequences have the correct number of columns
    if test_shape[1] != ref_shape[1]:
        if vlvl >= 1:
            print("The sequence has an incorrect number of columns, cannot "
                  "test")
        return False

    # Check that the sequences have the correct number of rows
    if test_shape[0] != ref_shape[0]:
        if vlvl >= 1:
            print("The sequence has an incorrect number of rows ({:d} vs "
                  "{:d}). Either the number of reference states is wrong, or "
                  "this is a different sequence altogether.".format(
                      test_shape[0], ref_shape[0]))
        return False

    # PSA eigenvector testing needs a list of PSA states in the same order as
    # used in the sequence
    psa_states = get_nsmp_state_order(pset_name)[1]

    # Loop over all sequence states to test them
    bad_states = 0
    for ind_seq in range(test_shape[0]):

        # To test PSG, take the input Stokes vector and calculate the outgoing
        # vector after the PSG using the test and true angles
        test_vec = get_psgpsa_vector(
            test_angles[ind_seq, 0:2]/180*np.pi, input_state=input_state)

        true_vec = get_psgpsa_vector(
            true_angles[ind_seq, 0:2]/180*np.pi, input_state=input_state)

        # Check if the test and true vectors are the same
        if not tensor_eq(test_vec, true_vec):
            bad_states += 1
            if vlvl >= 2:
                print("PSG eigenvector for state {:d} is incorrect".format(
                    ind_seq))

        # Get the true PSA state name index
        if with_ref_states:
            # When reference states are used, there is an extra reference state
            # after each PSA cycle. The reference state is assumed to have the
            # 0 th index.
            ind_psa = np.divmod(ind_seq, len(psa_states)+1)[1]
            if ind_psa == 6:
                ind_psa = 0
        else:
            ind_psa = np.divmod(ind_seq, len(psa_states))[1]

        # To test PSA, take the true PSA state names and calculate the outgoing
        # vector after the PSA using the test and true angles. If the
        # outgoing vector corresponds to the output state of the PSA the
        # polarimeter is working correctly.
        #
        # Note, that PSG testing takes a fixed input state and checks whether
        # the polarimeter produces the correct PSG states, while PSA testing
        # takes the true state list and checks whether the polarimeter produces
        # the same fixed output state for all of them. So the conversion is
        # one-to-many for PSG and many-to-one for PSA.

        test_vec = get_psgpsa_vector(
            test_angles[ind_seq, 2:4]/180*np.pi,
            input_state=psa_states[ind_psa])

        true_vec = get_psgpsa_vector(
            true_angles[ind_seq, 2:4]/180*np.pi,
            input_state=psa_states[ind_psa])

        if not tensor_eq(test_vec, get_stokes_vec(output_state)):
            bad_states += 1
            if vlvl >= 2:
                print("PSA eigenvector for state {:d} is incorrect".format(
                    ind_seq))

        # The true angle values should always produce the true PSA states, but
        # check that just in case
        if not tensor_eq(true_vec, get_stokes_vec(output_state)):
            if vlvl >= 2:
                print("WARNING: PSA eigenvector for true state {:d} is "
                      "incorrect".format(ind_seq))

    # Print verdict and return test result
    if bad_states == 0:
        if vlvl >= 1:
            print("All states are correct")
        return True
    elif bad_states < test_shape[0]:
        if vlvl >= 1:
            print("{:d} of {:d} states are incorrect".format(bad_states, test_shape[0]))
        return False
    else:
        if vlvl >= 1:
            print("All states are incorrect")
        return False
