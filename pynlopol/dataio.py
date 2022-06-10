"""Polarimetry data input/output functions.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import os
import numpy as np
import matplotlib.pyplot as plt

import scipy.ndimage as ndimg
import tifffile

from lkcom.string import change_extension
from lkcom.dataio import list_files_with_extension, read_bin_file
from pynlomic.cfgparse import read_cfg, parse_chan_idx

from pynlopol.nsmp_common import get_num_states


def get_microscopy_data_file_name(file_name=None):
    """Automatically get data file name in the current dir."""
    file_names = list_files_with_extension(ext='dat')

    # Remove PolStates.dat files
    file_names2 = []
    for file_name in file_names:
        if os.path.basename(file_name) != 'PolStates.dat':
            file_names2.append(file_name)
    file_names = file_names2

    if len(file_names) == 0:
        print("No data files found")
        return None
    if len(file_names) == 1:
        file_name = file_names[0]
        print("Found a single dat file '{:s}s', loading it".format(file_name))
        return file_name
    else:
        print("More than one dat file found, specify which to load")
        return None


def load_nsmp(file_name=None, chan_ind=None, binsz=None, cropsz=None):
    """Load NSMP dataset.

    If binsz == 'all', the images in the dataset are summed to a single pixel.
    """
    config = read_cfg(file_name)
    chan_ind = parse_chan_idx(config, chan_ind)

    print("Reading '{:s}'...".format(file_name), end='')
    data = read_bin_file(file_name)
    print('OK')

    num_chan = 4
    num_img = int(data.shape[2]/num_chan)

    if num_img == 55:
        print("BUG: 55 polarization states detected, truncating to 54")
        data = data[:, :, :-4]
        num_img = 54

    if num_img == 54:
        pset_name = 'shg_nsmp'
    elif num_img == 55:
        print("Dataset contains 55 states which is one too many for an SHG "
              "NSMP set. Discarding the extra state and assuming the data is "
              "SHG NSMP.")
        data = data[:, :, :-4]
        num_img = 54
    else:
        print("Polarization state set cannot be guessed from the number of "
              "images ({:d})in the dataset.".format(num_img))
        return None

    num_psg_states, num_psa_states = get_num_states(pset_name)
    if binsz == 'all':
        pipo_iarr = np.ndarray([num_psa_states, num_psg_states])
    else:
        num_row, num_col = np.shape(data)[0:2]
        if cropsz:
            num_row = cropsz[1] - cropsz[0]
            num_col = cropsz[3] - cropsz[2]
        pipo_iarr = np.ndarray(
            [num_row, num_col, num_psa_states, num_psg_states])

    if cropsz:
        print("Cropping image to " + str(cropsz) + " px")

    print("Assuming PSA-first order")
    for ind_psg in range(num_psg_states):
        for ind_psa in range(num_psa_states):
            frame_ind = (ind_psa + ind_psg*num_psa_states)*num_chan + chan_ind
            img = data[:, :, frame_ind]
            if cropsz:
                img = img[cropsz[0]:cropsz[1], cropsz[2]:cropsz[3]]
            if binsz == 'all':
                pipo_iarr[ind_psa, ind_psg] = np.sum(img)
            else:
                pipo_iarr[:, :, ind_psa, ind_psg] = img

    return pipo_iarr


def convert_nsmp_to_tiff(
        file_name=None, pimg_arr=None, preset='basic',
        out_sz=None, add_dummy_ref_states=False, sanity_check=False,
        **kwargs):
    """Convert an NSMP dataset to a 16-bit multipage TIFF.

    Multipage TIFF files are useful for moving data to other software. The
    16-bit output ensures raw count numbers are maintained. Note, that not all
    software supports 16-bit TIFF files.

    The dataset can be supplied either directly (pimg_arr) or by specifying a
    file name (file_name) to load the dataset from.

    The input pimg_arr is a 4D array of polarization-resolved images in a 2D
    PSG-PSA grid. The polarization grid is arranged linearly as pages in the
    output TIFF file in PSA-major order with reference states, if present,
    interleaved after each PSA cycle.

    Output TIFF formatting can be adjusted either using presets or by
    configuring format parameters directly.

    Output image size can be specified using out_sz. If the output size is
    different from the input image size, the images are resampled.

    If the input dataset do not contain reference states, and
    add_dummy_ref_states is True, dummy reference states are created by
    duplicating the first input (0, 0 index) state. This is useful when
    the output TIFF must contain the reference states to maintain a particular
    state order.
    """
    if pimg_arr is None:
        pimg_arr = load_nsmp(file_name, **kwargs).astype('uint16')

    num_row, num_col, num_psa, num_psg = np.shape(pimg_arr)
    print("Input dataset size: {:d}x{:d}p px, {:d}x{:d} PSGxPSA".format(
        num_row, num_col, num_psg, num_psa))

    if num_row != num_col:
        print("This function was only tested for square images")

    num_states = num_psg*num_psa

    if add_dummy_ref_states:
        print("Output dataset with reference states requested, but the input "
              "dataset does not contain any. The PSG=0, PSA=0 state will be "
              "duplicated to make a dummy reference set.")
        # Reference states are usually measured after each PSA cycle. There are
        # as many reference states as there are PSG states, for each of which a
        # PSA state cycle is performed. Add this number to the total number of
        # states
        num_states += num_psg

    # Build output TIFF file name
    tiff_file_name = change_extension(file_name, '.tiff')

    # Initialize output
    if out_sz is None:
        out_row = num_row
        out_col = num_col
    else:
        out_row = out_col = out_sz

    pimg_arr_out = np.ndarray([num_states, out_row, out_col])

    if num_row != out_row or num_col != out_col:
        print("Image will be resampled to 128x128")

    # Calculate the factor to resample the input image to output size
    # Even though this takes the minimum over x/y, the function will likely
    # not work for non-square images.
    resample_fac = np.min([out_row/num_row, out_col/num_col])

    ind_state = 0
    img_ref = None

    # Loop over PSG and PSA to build the linear array for TIFF pages
    for ind_psg in range(num_psg):
        for ind_psa in range(num_psa):
            # Take the input image and resample it to output size
            # Even though a 'zoom' function sounds funny, it works on 16-bit
            # data stored as ndarray. PIL, OpenCV, etc. either need conversion
            # back and forth or don't even work on uint16 data
            img_out = ndimg.zoom(
                pimg_arr[:, :, ind_psa, ind_psg], resample_fac)

            if add_dummy_ref_states and img_ref is None:
                # Copy the PSG=0, PSA=0 state to use as a dummy for all
                # reference states
                img_ref = img_out.copy()

            # Assign output images in a linearly incrementing way
            # The linear increment is crucial as it allows easy reference state
            # interleaving
            pimg_arr_out[ind_state, :, :] = img_out
            ind_state += 1

            if add_dummy_ref_states and ind_psa == num_psa - 1:
                # Add a reference state after each PSA cycle
                pimg_arr_out[ind_state, :, :] = img_ref
                ind_state += 1

    print("Output dataset size: {:d}x{:d}p px, {:d} interleaved states".format(
        out_row, out_col, np.shape(pimg_arr_out)[0]))

    if sanity_check:
        # Perform a sanity check by undoing all data formatting steps and
        # checking whether the output data is the same as the input
        check = pimg_arr[0, 0, :, :]
        data = pimg_arr_out[:, 0, 0]

        if add_dummy_ref_states:
            data = np.delete(data, np.arange(-1, num_states, num_psg+1)[1:])

        data = np.reshape(data, [num_psa, num_psg])

        if not np.all(data == check):
            plt.subplot(1, 3, 1)
            plt.imshow(data)
            plt.title('Output data')
            plt.subplot(1, 3, 2)
            plt.imshow(check)
            plt.title('Check')
            plt.subplot(1, 3, 3)
            plt.imshow(data-check, cmap='coolwarm')
            plt.title('Error')
            plt.show()
            raise Exception("TIFF export sanity check failed")

    print("Writing '{:s}'".format(tiff_file_name))
    tifffile.imwrite(tiff_file_name, pimg_arr_out)
    print("All done")
