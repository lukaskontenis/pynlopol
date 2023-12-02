
import numpy as np
import matplotlib.pyplot as plt
from pynlopol.nsmp_sim import simulate_pipo, plot_pipo

from lkcom.plot import set_ticks_inside, export_figure
from scipy.optimize import minimize

def plot_pipo_fit_err_map():
    """Plot PIPO fit error map.

    Plot PIPO fit error maps showing the total fit error as a function of the
    mismatch between true and fit parameters. The result is a surface on which
    the fiter is optimizing for a minimum. The shape of the surface shows the
    number of local minima that the fit can get stuck at. The curvature gives
    shows the fit difficulty and convergence speed. Noise added to the data
    shows the expected accuracy of the fit and the parameter determination
    sensitivity in the presence of shot noise and systematic experimental
    errors.
    """


def plot_pipo_fit_err_map_shg_c6v():
    """Plot SHG C6v PIPO fit error map."""
    optimize_ampl = False
    # SHG C6v
    sample_str = 'muscle'
    delta_true = 45
    zzz_true = 0.7
    sim_par = {
        'sample_name': 'c6v',
        'symmetry_str': 'c6v',
        'nlorder': 2,
        'ampl': 10,
        'delta': delta_true/180*3.14,
        'zzz': zzz_true,
        'pset_name': 'pipo_8x8',
        'output_type': '1point'
    }

    num_pts = 25
    max_zzz = 3

    delta_arr = np.linspace(0, np.pi, num_pts)
    zzz_arr = np.linspace(0.1, max_zzz, num_pts)

    pipo_data_true = sim_par.get('ampl', 1)*simulate_pipo(**sim_par)

    def err_func(ampl):
        return np.sum(np.abs(ampl*pipo_data_test - pipo_data_true).flatten())

    pipo_err = np.ndarray([num_pts, num_pts])
    for ind_delta, delta in enumerate(delta_arr):
        sim_par['delta'] = delta
        for ind_zzz, zzz in enumerate(zzz_arr):
            print("Simulation point {:} of {:}".format(ind_delta*num_pts + ind_zzz, num_pts**2))
            sim_par['zzz'] = zzz
            pipo_data_test = simulate_pipo(**sim_par)
            if optimize_ampl:
                fit_err = minimize(err_func, 1, tol=0.1)
                fit_err_val = fit_err.fun
            else:
                fit_err_val = err_func(1)

            pipo_err[ind_delta, ind_zzz] = fit_err_val

    plt.contourf(pipo_err, origin='lower', cmap='hot', levels=20, extent=[0.1, max_zzz, 0, 180])
    plt.colorbar()
    plt.contour(pipo_err, origin='lower', levels=20, extent=[0.1, max_zzz, 0, 180], colors='k')
    plt.scatter(zzz_true, delta_true, c='w')
    if max_zzz > 3:
        plt.xticks([0.5, 1, 3, 5, 8, 10])
    else:
        plt.xticks([0.5, 1, 1.5, 3])
    plt.yticks([0, 45, 90, 135, 180])
    plt.xlim([0.1, max_zzz])
    plt.ylim([0, 180])
    plt.grid('on')
    set_ticks_inside()
    plt.xlabel('zzz/zxx')
    plt.ylabel('delta, deg')
    plt.title('Fit error map for {:}, delta = {:}, zzz = {:}'.format(sample_str, delta_true, zzz_true))
    export_figure('fit_error_map_{:}_{:}.png'.format(sample_str, zzz_true))
    plt.show()




    # plot_pipo(pipo_data, title_str=None, show_fig=True, pset_name='pipo_100x100')

def plot_pipo_fit_err_map_thg_c6v():
    """Plot THG C6v PIPO fit error map."""

    optimize_ampl = True
    # THG C6v
    true_par = {
        'delta': 45/180*np.pi,
        'zzzz': 3,
        'xxxx': 3
    }

    sim_par = {
        'sample_name': 'ECM',
        'symmetry_str': 'c6v',
        'nlorder': 3,
        'ampl': 1,
        'pset_name': 'pipo_8x8',
        'output_type': '1point'
    }

    num_pts = 25
    max_zzzz = 10
    max_xxxx = 10

    par_names = ['delta', 'zzzz', 'xxxx']
    par_rng = [[0, np.pi], [0.1, max_zzzz], [0.1, max_xxxx]]
    num_par = len(par_names)
    num_comb = int(num_par*(num_par-1)/2)
    par_arr = np.ndarray([num_pts, num_par])
    for ind_par in range(num_par):
        par_arr[:, ind_par] = np.linspace(*par_rng[ind_par], num_pts)



    def err_func(ampl):
        return np.sum(np.abs(ampl*pipo_data_test - pipo_data_true).flatten())

    comb_arr = []
    for ind1 in range(num_par-1):
        for ind2 in np.arange(ind1+1, num_par):
            comb_arr.append([ind1, ind2])

    pipo_err = np.ndarray([num_pts, num_pts, num_comb])
    for ind_comb in range(num_comb):
        ind_par1, ind_par2 = comb_arr[ind_comb]
        for ind_par, par_name in enumerate(par_names):
            sim_par[par_name] = true_par[par_name]
        pipo_data_true = sim_par.get('ampl', 1)*simulate_pipo(**sim_par)
        for ind1, par1 in enumerate(par_arr[:, ind_par1]):
            sim_par[par_names[ind_par1]] = par1
            for ind2, par2 in enumerate(par_arr[:, ind_par2]):
                sim_par[par_names[ind_par2]] = par2
                print("Simulation point {:} of {:}".format(ind_comb*num_pts**2 + ind1*num_pts + ind2, num_pts**2*num_comb))

                pipo_data_test = simulate_pipo(**sim_par)
                if optimize_ampl:
                    fit_err = minimize(err_func, 1, tol=0.1)
                    fit_err_val = fit_err.fun
                else:
                    fit_err_val = err_func(1)

                pipo_err[ind1, ind2, ind_comb] = fit_err_val

    plt.figure(figsize=[15, 5])
    for ind in range(num_comb):
        plt.subplot(1, 3, ind+1)
        y_par_name = par_names[comb_arr[ind][0]]
        extent = np.array([*par_rng[comb_arr[ind][1]], *par_rng[comb_arr[ind][0]]])
        if y_par_name == 'delta':
            extent[2:] *= 180/np.pi

        plt.contourf(pipo_err[:,:, ind], origin='lower', cmap='hot', levels=10, extent=extent)
        if ind == num_comb - 1:
            plt.colorbar()
        plt.contour(pipo_err[:,:, ind], origin='lower', levels=10, extent=extent, colors='k')

        true_point = [true_par[par_names[comb_arr[ind][1]]], true_par[y_par_name]]
        if y_par_name == 'delta':
            true_point[1] *= 180/np.pi

        plt.scatter(*true_point, c='w')
        if par_names[comb_arr[ind][0]] == 'delta':
            plt.yticks([0, 45, 90, 135, 180])

        plt.grid('on')
        set_ticks_inside()
        plt.xlabel(par_names[comb_arr[ind][1]])
        plt.ylabel(par_names[comb_arr[ind][0]])

    plt.suptitle('Fit error maps for {:}, delta = {:}, zzzz = {:}, xxxx = {:}'.format(sim_par['sample_name'], true_par['delta']/np.pi*180, *[true_par[name] for name in par_names[1:]]))
    export_figure('fit_error_map_{:}_{:}_{:}.png'.format(sim_par['sample_name'], *[true_par[name] for name in par_names[1:]]))
    plt.show()




    # plot_pipo(pipo_data, title_str=None, show_fig=True, pset_name='pipo_100x100')

plot_pipo_fit_err_map_thg_c6v()
