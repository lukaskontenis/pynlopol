from pynlopol.nsmp_sim import simulate_pipo
from pynlopol.nsmp_fit import fit_pipo
import numpy as np

from scipy.optimize import minimize

def test_thg_c6v_pipo_fit_1pt():
    """Test single-point THG C6v PIPO fitting."""
    print("Testing THG C6v single-point PIPO fitting...")

    # THG C6v
    sim_par = {
        'sample_name': 'c6v',
        'symmetry_str': 'c6v',
        'nlorder': 3,
        'delta': 15/180*3.14,
        'zzzz': 3,
        'xxxx': 5,
        'zzxx': 1,
        'pset_name': 'pipo_8x8',
        'output_type': '1point'
    }

    pipo_data = simulate_pipo(**sim_par)

    fit_model = 'thg_c6v'
    fitfun_names = ['nsmpsim']
    fit_algorithm = 'scipy_minimize'

    for fitfun_name in fitfun_names:
        fit_result = fit_pipo(
            ask_before_overwrite=False,
            pipo_arr=pipo_data,
            fit_algorithm=fit_algorithm, fit_model=fit_model, fitfun_name=fitfun_name,
            print_results=True, plot_progress=False, show_fig=True)

        if fit_model == 'shg_c6v':
            fit_result.set_ref_par({'zzz': sim_par['zzz'], 'delta': sim_par['delta']})
        elif fit_model == 'thg_c6v':
            fit_result.set_ref_par({'zzzz': sim_par['zzzz'], 'xxxx': sim_par['xxxx'], 'delta': sim_par['delta']})
        fit_result.print()
        if fit_result.test_against_ref():
            print(fit_model + ' fitting using ' + fitfun_name +
                    ' fit function verified')
        else:
            print(fit_model + ' fitting using ' + fitfun_name +
                    ' fit function FAILED')
            self.assertTrue(False)


test_thg_c6v_pipo_fit_1pt()
