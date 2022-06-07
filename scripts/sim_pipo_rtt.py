"""Simulate a PIPO dataset.

Simulate a rat-tail tendon PIPO dataset using a reference image as a mask and
assuming all pixels have the same R=zzz/zxx ratio and in-plane angle delta.

This script is part of pynolpol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

symmetry_str = 'c6v'
delta = 40/180*3.14
zzz = 1.5
pset_name = 'pipo_8x8'
output_type = 'img'

try:
    print("=== Rat-tail tendon PIPO image simulation ===")

    import numpy as np

    from lkcom.util import handle_general_exception
    from pynolmic.proc import convert_pipo_to_tiff

    from pynolpol import simulate_pipo
    

    print("R = {:.2f}".format(zzz))
    print("δ = {:.2f}°".format(delta/3.14*180))

    print("Generating map...")

    pipo_data = simulate_pipo(
        symmetry_str=symmetry_str, delta=delta, zzz=zzz,
        pset_name=pset_name, output_type=output_type,
        img_type='ref_img',
        ref_img_name='mask_15_28_23_.375_.png')

    print("Exporting PIPO dataset as a pipo_arr.file...")
    np.save('rtt_pipo_sim.npy', pipo_data)

    print("Exporting PIPO dataset as a multipage TIFF file...")
    convert_pipo_to_tiff(
        pipo_arr=pipo_data, file_name='rtt_pipo_sim',
        preset='piponator')

except Exception:
    handle_general_exception("Could not simulate PIPO")

input("Press any key to close this window")
