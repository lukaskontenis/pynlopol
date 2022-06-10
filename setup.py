import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pynlopol",
    version="0.9.2",
    author="Lukas Kontenis",
    author_email="dse.ssd@gmail.com",
    description="A Python library for nonlinear polarimetry.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.19.5', 'matplotlib>=3.3.3', 'scipy>=1.5.4', 'imageio>=2.9.0', 'pynlomic>=0.9.3', 'lkcom>=0.2.0',
    ],
    python_requires='>=3.6',
    data_files=[
        ('scripts', [
        'scripts/fit_pipo_1point.py',
        'scripts/fit_pipo_1point_zcq.py',
        'scripts/fit_pipo_img.py',
        'scripts/gen_pol_state_sequence.py',
        'scripts/make_nsmp_tiff.py',
        'scripts/pipo_check_c6v.py',
        'scripts/plot_pipo_fit.py',
        'scripts/plot_piponator_fit.py',
        'scripts/show_pipo.py',
        'scripts/sim_collagen_anim.py',
        'scripts/sim_pipo.py',
        'scripts/sim_pipo_collagen.bat',
        'scripts/sim_pipo_collagen_hr.bat',
        'scripts/sim_pipo_rtt.py',
        'scripts/sim_pipo_zcq.bat',
        'scripts/sim_pipo_zcq_hr.bat',
        'scripts/sim_zcq_pipo_anim.py',
        'scripts/verify_pol_state_sequence.py']),
        ('tests', [
        'tests/pipo_8x8_pol_states.dat',
        'tests/test_polarimetry_fit.py',
        'tests/test_polarimetry_lin.py',
        'tests/test_polarimetry_nl.py',
        'tests/test_polarimetry_plot.py'])],
)
