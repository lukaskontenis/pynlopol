"""Polarimetry ata input/output functions.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
import os
import numpy as np

from lklib.fileread import list_files_with_extension, read_bin_file

from lcmicro.cfgparse import read_cfg, parse_chan_idx

from lcmicro.polarimetry.nsmp_common import get_num_states


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
