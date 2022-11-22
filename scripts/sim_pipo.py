"""Simulate a PIPO dataset.

Simulate a PIPO map for a given nonlinear susceptibility tensor symmetry and
sample parameters. Currently, SHG c6v (for collagen) and D3 (for z-cut quartz)
tensors are supported with zzz/zxx and in-plane angle (delta) parameters.

Args:
    sample_name – 'zcq' for z-cut quartz and 'collagen'
    delta – sample in-plane orientation angle in degrees
    zzz – R-ratio (zzz/zxx) in the collagen case

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

# == Example arguments ==
# Collagen SHG C6v
# sim_par = {
#     'sample_name': 'collagen',
#     'delta': 0/180*3.14,
#     'nlorder': 2,
#     'zzz': 1.5
# }

# THG C6v
sim_par = {
    'sample_name': 'c6v',
    'nlorder': 3,
    'delta': 0/180*3.14,
    'zzzz': 10,
    'xxxx': 5,
    'zzxx': 3
}

# Polarization state set, use, for example:
#   'pipo_8x8' or 'pipo_9x9' for typical microscope scan data
#   'pipo_100x100' for a nicer figure to look at
pset_name = 'pipo_100x100'
output_type = '1point'  # '1point' or 'img'

try:
    print("=== pynlopol ===")
    print("Generating PIPO map...")

    import sys
    import matplotlib.pyplot as plt
    import numpy as np

    from lkcom.util import handle_general_exception
    from lkcom.plot import export_figure
    from pynlomic.proc import convert_pipo_to_tiff

    from pynlopol.nsmp_sim import simulate_pipo, plot_pipo

    sample_name = sim_par.get('sample_name')

    num_args = len(sys.argv)
    if num_args < 2:
        print("Running script with default values.\n")
        print("To specify different values:")
        print("\tsim_pipo.py sample delta zzz num_states")
        print("\nwhere sample is either 'collagen' or 'zcq', delta is\n"
              "in-plane orienation in degrees, and 'num_states' is the\n"
              "number of PSG and PSA states.")
        print("Default values are: zcq, δ=15°, R=1.5, 8 states")
        print("\nFor example:")
        print("\tsim_pipo.py collagen 30 1.5 200")
        print("\tsim_pipo.py collagen 30 1.5")
        print("\tsim_pipo.py zcq 60")
    if num_args >= 2:
        try:
            if sys.argv[1] in ['zcq', 'collagen']:
                sample_name = sys.argv[1]
            else:
                print("Invalid sample name, using default")
                sample_name = None
        except Exception:
            print("Could not determine sample name, using default")

    if num_args >= 3:
        try:
            delta = float(sys.argv[2])
        except Exception:
            print("Could not determine delta, using default")
            delta = None

    if num_args >= 4:
        try:
            zzz = float(sys.argv[3])
            if sample_name == 'zcq' and zzz is not None:
                print("Z-cut quartz has no relative tensor components, ignoring zzz")
                zzz = None
        except Exception:
            print("Could not determine zzz, using default")
            zzz = None

    if num_args >= 5:
        try:
            num_states = int(sys.argv[4])
            pset_name = "pipo_{:d}x{:d}".format(num_states, num_states)
        except Exception:
            handle_general_exception(Exception)
            print("Could not parse number of states argument, using default")

    if sample_name == 'collagen':
        symmetry_str = 'c6v'
    elif sample_name == 'zcq':
        symmetry_str = 'd3'
    elif sample_name == 'c6v':
        symmetry_str = 'c6v'
    else:
        raise(Exception("Unsupported sample name '{:s}'".format(sample_name)))

    delta = sim_par.get('delta')

    print("\nSample name: " + sample_name)
    print("Delta: {:.2f}°".format(delta))
    if sample_name == 'collagen':
        print("zzz: {:.2f}".format(zzz))

    print("Generating map...")

    pipo_data = simulate_pipo(**sim_par, symmetry_str=symmetry_str,
        pset_name=pset_name, output_type=output_type)

    title_str = ''
    if sample_name == 'collagen':
        title_str = "Collagen R={:.2f}".format(zzz) + " PIPO map, δ={:.0f}°".format(delta/3.14*180)
    elif sample_name == 'zcq':
        title_str = "Z-cut quartz PIPO map, δ={:.0f}°".format(delta/3.14*180)

    if len(np.shape(pipo_data)) == 2:
        plot_pipo(pipo_data, title_str=title_str, show_fig=False, pset_name=pset_name)

        print("Exporting 'pipo_map.png'...")
        export_figure('pipo_map.png', resize=False)

        print("Showing figure...")
        plt.show()

    if len(np.shape(pipo_data)) == 4:
        print("Exporting PIPO dataset as a multipage TIFF file...")
        convert_pipo_to_tiff(
            pipo_arr=pipo_data, file_name=sample_name + 'pipo_sim',
            duplicate_first_and_last_state=True,
            preset='piponator',
            add_dummy_ref_states=True)

except Exception:
    handle_general_exception("Could not simulate PIPO")

input("Press any key to close this window")

