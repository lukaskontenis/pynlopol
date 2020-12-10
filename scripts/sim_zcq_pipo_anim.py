"""Generate animated SHG PIPO map for z-cut quartz.

Generate an animated GIF of an SHG PIPO map for z-cut quartz at varying delta.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

print("=== lcmicro ===")
print("Generating PIPO map GIF...")

from lklib.util import handle_general_exception
from lcmicro.polarimetry import make_pipo_animation_delta, \
    make_pipo_animation_zzz

try:
    make_pipo_animation_delta(sample='zcq')
except Exception:
    handle_general_exception("Could not generate animations")

input("Press any key to close this window")
