"""Generate animated THG PIPO maps for collagen.

Generate animated GIFs of THG PIPO maps for collagen at varying delta.

This script is part of pynlopol, a Python library for nonlinear polarimetry.

Copyright 2015-2023 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""
# flake8: noqa
# pylint: skip-file

print("=== pynlopol ===")
print("Generating PIPO map GIFs...")

from lkcom.util import handle_general_exception
from pynlopol.nsmp_sim import make_pipo_animation_delta, \
    make_pipo_animation_ratio

par = {
    'sample': 'collagen',
    'nlorder': 3,
    'bare_plot_data': False}

try:
    make_pipo_animation_delta(**par)
    make_pipo_animation_ratio(**par, anim_var='zzzz')
    make_pipo_animation_ratio(**par, anim_var='zzxx')
except Exception:
    handle_general_exception("Could not generate animations")

input("Press any key to close this window")
