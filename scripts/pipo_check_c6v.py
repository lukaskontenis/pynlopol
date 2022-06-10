"""Check PIPO map behavior for C6v.

Simulate C6v PIPO maps for several cases to showcase its behavior.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

import numpy as np

zzz_arr = [1.4, 1.5, 1.6, 1.7]
delta_arr = np.array([0, 0, 0, 0])/180*3.14
pset_name = 'pipo_100x100'

try:
    print("=== C6v PIPO map properties ===")

    import sys
    import matplotlib.pyplot as plt

    from lkcom.util import handle_general_exception
    from lkcom.plot import export_figure
    from pynlomic.proc import convert_pipo_to_tiff

    from pynlopol import simulate_pipo, plot_pipo
    

    plt.figure(figsize=[11, 11])

    print("Generating PIPO maps...")

    for ind, zzz in enumerate(zzz_arr):
        delta = delta_arr[ind]
        pipo_data = simulate_pipo(symmetry_str='c6v', delta=delta, zzz=zzz, pset_name=pset_name, output_type='1point')

        title_str = "R={:.2f}".format(zzz) + ", δ={:.0f}°".format(delta/3.14*180)
        plt.subplot(2, 2, ind+1)
        plot_pipo(pipo_data, title_str=title_str, show_fig=False, pset_name=pset_name)

    print("Showing figure...")
    plt.show()

except Exception:
    handle_general_exception("Could not simulate PIPO")

input("Press any key to close this window")
