"""Generate animated SHG PIPO map for z-cut quartz.

Generate an animated GIF of an SHG PIPO map for z-cut quartz at varying delta.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

print("=== pynlopol ===")
print("Generating PIPO map GIF...")

from lkcom.util import handle_general_exception
from pynlopol import make_pipo_animation_delta

try:
    make_pipo_animation_delta(sample='zcq')
except Exception:
    handle_general_exception("Could not generate animations")

input("Press any key to close this window")
