"""Generate an SHG PIPO map.

Generate an SHG PIPO map for the given sample. In-plane angle (delta) and zzz
zzz ratio input parameters are supported.

Args:
    sample_name – 'zcq' for z-cut quartz and 'collagen'
    delta – sample in-plane orientation angle in degrees
    zzz – R-ratio (zzz/zxx) in the collagen case

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

sample_name = 'zcq'
delta = 15
zzz = 1.5
pset_name = 'pipo_8x8'

try:
    print("=== lcmicro ===")
    print("Generating PIPO map...")

    import sys
    import matplotlib.pyplot as plt

    from lklib.plot import export_figure
    from lcmicro.polarimetry import simulate_pipo, plot_pipo
    from lklib.util import handle_general_exception

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
    else:
        raise(Exception("Unsupported sample name '{:s}'".format(sample_name)))

    print("\nSample name: " + sample_name)
    print("Delta: {:.2f}°".format(delta))
    if sample_name == 'collagen':
        print("zzz: {:.2f}".format(zzz))

    print("Generating map...")

    pipo_data = simulate_pipo(symmetry_str=symmetry_str, delta=delta/180*3.14, zzz=zzz, pset_name=pset_name)

    if sample_name == 'collagen':
        title_str = "Collagen R={:.2f}".format(zzz) + " PIPO map, δ={:.0f}°".format(delta)
    elif sample_name == 'zcq':
        title_str = "Z-cut quartz PIPO map, δ={:.0f}°".format(delta)

    plot_pipo(pipo_data, title_str=title_str, show_fig=False, pset_name=pset_name)

    print("Exporting 'pipo_map.png'...")
    export_figure('pipo_map.png', resize=False)

    print("Showing figure...")
    plt.show()

except Exception:
    handle_general_exception("Could not simulate PIPO")

input("Press any key to close this window")

