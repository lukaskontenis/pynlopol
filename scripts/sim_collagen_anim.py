"""Generate animated SHG PIPO maps for collagen.

Generate animated GIFs of SHG PIPO maps for collagen at varying delta and zzz.

This script is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2022 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

print("=== lcmicro ===")
print("Generating PIPO map GIFs...")

from lkcom.util import handle_general_exception
from lcmicro.polarimetry import make_pipo_animation_delta, \
    make_pipo_animation_zzz

try:
    make_pipo_animation_delta(sample='collagen')
    make_pipo_animation_zzz(sample='collagen')
except Exception:
    handle_general_exception("Could not generate animations")

input("Press any key to close this window")
