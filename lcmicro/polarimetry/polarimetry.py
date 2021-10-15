
"""Linear polarimetry.

This module is part of lcmicro, a Python library for nonlinear microscopy and
polarimetry.

Copyright 2015-2020 Lukas Kontenis
Contact: dse.ssd@gmail.com
"""

import numpy as np
from numpy import zeros, sin, cos, pi
import matplotlib.pyplot as plt

from lklib.plot import add_y_marker


def col_vec(arr):
    """Convert array to a column vector."""
    vec = np.array(arr)
    vec.shape = (len(vec), 1)
    return vec


def get_eps():
    """Get the floating point math precision."""
    return np.finfo('float64').eps


def tensor_eq(tns1, tns2, thr=get_eps()):
    """Check if two tensors are equal within floating point precision.

    Works for vectors and matrices too.
    """
    if isinstance(tns1, list):
        tns1 = np.array(tns1)
    if isinstance(tns2, list):
        tns2 = np.array(tns2)
    return (np.abs(tns1 - tns2) <= thr).all()


def rot_mueller_mat(mat, theta=0):
    """Get a rotated Mueller matrix."""
    mat_rot = zeros([4, 4])

    mat_rot[0, 0] = 1

    mat_rot[1, 1] = cos(2*theta)
    mat_rot[1, 2] = -sin(2*theta)
    mat_rot[2, 1] = sin(2*theta)
    mat_rot[2, 2] = cos(2*theta)

    mat_rot[3, 3] = 1

    return mat_rot @ mat @ mat_rot.transpose()


def get_mueller_mat(element, theta=0, **kwargs):
    """Get the Mueller matrix of ``element`` with rotation.

    Rotation is given by theta.

    Transmission coefficients for diattenuating polarizer ('PolD') are
    specified using q and r kwargs.
    """
    mat = zeros([4, 4])

    element = element.lower()

    if element == "hwp":
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 2] = -1
        mat[3, 3] = -1

    elif element == "qwp":
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 3] = 1
        mat[3, 2] = -1

    elif element in ("rtd", "retarder"):
        d = kwargs.get('d', 0)  # pylint: disable=C0103
        c = cos(d)  # pylint: disable=C0103
        s = sin(d)  # pylint: disable=C0103

        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 2] = c
        mat[2, 3] = s
        mat[3, 2] = -s
        mat[3, 3] = c

    elif element in ("rotator", "fr"):
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 3] = -1
        mat[3, 2] = 1

    elif element in ("plz", "pol", "polarizer"):
        mat[0, 0] = 1
        mat[0, 1] = 1
        mat[1, 0] = 1
        mat[1, 1] = 1
        mat = mat*0.5

    elif element in ("dattn", "real_pol", "real polarizer"):
        qtrans = kwargs.get('qtrans', 1)
        rtrans = kwargs.get('rtrans', 0)

        mat[0, 0] = qtrans + rtrans
        mat[1, 0] = qtrans - rtrans
        mat[0, 1] = qtrans - rtrans
        mat[1, 1] = qtrans + rtrans
        mat[2, 2] = 2*np.sqrt(qtrans*rtrans)
        mat[3, 3] = 2*np.sqrt(qtrans*rtrans)

        mat = mat*0.5

    elif element in ("unity", "empty", "nop"):
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[2, 2] = 1
        mat[3, 3] = 1

    elif element in ('fresnel_trans'):
        ai = kwargs.get('ai', 0)
        n1 = kwargs.get('n1', 1)
        n2 = kwargs.get('n2', 1.5)
        ar = np.arcsin(n1/n2*sin(ai))

        ts = np.tan(ai)/np.tan(ar) * (2*np.sin(ar)*np.cos(ai)/np.sin(ar + ai))**2
        tp = np.tan(ai)/np.tan(ar) * (2*np.sin(ar)*np.cos(ai)/(np.sin(ar + ai)*np.cos(ai - ar)))**2

        mat[0, 0] = ts + tp
        mat[0, 1] = ts - tp
        mat[1, 0] = ts - tp
        mat[1, 1] = ts + tp

        mat[2, 2] = 2*np.sqrt(ts*tp)
        mat[3, 3] = 2*np.sqrt(ts*tp)

        mat = mat * 0.5

    elif element in ('fresnel_refl'):
        ai = kwargs.get('ai', 0)
        n1 = kwargs.get('n1', 1)
        n2 = kwargs.get('n2', 1.5)
        ar = np.arcsin(n1/n2*sin(ai))

        rs = (np.sin(ai - ar)/np.sin(ar + ai))**2
        rp = (np.tan(ai - ar)/np.tan(ar + ai))**2

        mat[0, 0] = rs + rp
        mat[0, 1] = rs - rp
        mat[1, 0] = rs - rp
        mat[1, 1] = rs + rp

        mat[2, 2] = 2*np.sqrt(rs*rp)
        mat[3, 3] = 2*np.sqrt(rs*rp)

        mat = mat * 0.5

    else:
        print("Element ''{:s}'' not defined".format(element))

    mat = rot_mueller_mat(mat, theta)

    return mat


def get_stokes_vec(state, gamma=0, omega=0):
    """Get the Stokes vector of ``state``."""
    svec = zeros([4, 1])

    try:
        lp_angle = float(state)
    except ValueError:
        lp_angle = None

    if lp_angle is not None:
        # The gamma angle is equal to the polarization angle for purely linear
        # states. The omega angle is zero for linear states.
        gamma = lp_angle / 180 * pi
        omega = 0
    else:
        state = state.lower()
        if state == "hlp":
            gamma = 0
            omega = 0
        elif state == "vlp":
            gamma = pi/2
            omega = 0
        elif state == "rcp":
            gamma = 0
            omega = pi/4
        elif state == "lcp":
            gamma = 0
            omega = -pi/4
        elif state == 'rep':
            gamma = pi/2
            omega = pi/8
        elif state == 'lep':
            gamma = pi/4
            omega = -pi/8
        elif state == 'custom':
            gamma = gamma
            omega = omega
        else:
            print('State ''{:s}'' not defined'.format(state))

    svec[0] = 1
    svec[1] = cos(2*gamma) * cos(2*omega)
    svec[2] = sin(2*gamma) * cos(2*omega)
    svec[3] = sin(2*omega)

    svec.shape = (4, 1)
    return svec


def get_psgpsa_vector(angles, input_state='hlp'):
    """Get Stokes vector after a PSG/PSA.

    Propagate an input Stokes vector through a HWP followed by a QWP rotated by
    the angles given.

    Args:
        angles (arr) - hwp, qwp angles in rad
        input_state (str) - input state before the PSG/PSA

    Returns:
        Output Stokes vector
    """
    svec_in = get_stokes_vec(input_state)
    mmat_hwp = get_mueller_mat('hwp', angles[0])
    mmat_qwp = get_mueller_mat('qwp', angles[1])
    return mmat_qwp.dot(mmat_hwp.dot(svec_in))


def get_waveplate_thickness(
        target_rtrd=None, plate_type='hwp', wavl=None, biref=0.0092, worder=0):
    """Get waveplate thickness given its order and birefringence.

    Default birefringence is 0.0092 for a quartz waveplate in green. Thickess
    is returned in the same units as wavelength.
    """
    if target_rtrd is not None:
        return target_rtrd*wavl/biref
    if plate_type == 'hwp':
        return (2*worder + 1)*wavl / (2*biref)
    elif plate_type == 'qwp':
        return (2*worder + 1)*wavl/2 / (2*biref)
    else:
        print("Unsupported plate type " + plate_type)
        return None


def get_waveplate_retardation(wavl=None, biref=0.0092, thickness=None):
    """Get waveplate retardation given its birefringence and thickness.

    Default birefringence is 0.0092 for a quartz waveplate in green.
    Retartadtion is returned in waves.
    """
    rtrd = biref*thickness/wavl  # in waves
    #rtrd = np.mod(rtrd, 1)
    return rtrd


def plot_waveplate_response(
        plate_type='hwp', rtrd=None, title_str=None, finalize_figure=True):
    """Plot waveplate transmission response.

    Plot the intensity transmited through a rotatating waveplate and a fixed
    polarizer as a function of anle.
    """
    if rtrd is None:
        if plate_type == 'hwp':
            rtrd = 0.5
        elif plate_type == 'qwp':
            rtrd = 0.25
        else:
            print("Unsupported plate type " + plate_type)
            return None

    in_svec = get_stokes_vec('hlp')
    pol_hwp = get_mueller_mat('pol')

    theta_arr = np.linspace(0, 2*np.pi, 500)
    det_ampl = np.empty_like(theta_arr)

    for ind, theta in enumerate(theta_arr):
        hwp_mat = get_mueller_mat('rtd', theta=theta, d=rtrd*2*np.pi)
        out_svec = pol_hwp.dot(hwp_mat.dot(in_svec))
        det_ampl[ind] = out_svec[0]

    plt.plot(theta_arr/np.pi*180, det_ampl)

    if finalize_figure:
        add_y_marker(0)
        add_y_marker(1)
        plt.xlim([0, 360])
        plt.grid('on')
        plt.xticks(np.arange(0, 361, 45))
        if title_str is None:
            title_str = 'Rotating {:s} response'.format(plate_type.upper())
        plt.title(title_str)
        plt.xlabel('{:s} orientation, deg'.format(plate_type.upper()))
        plt.ylabel('Transmitted power, a.u.')


def plot_retarder_response(
        rtrd=None, theta=None, title_str=None, finalize_figure=True):
    """Plot retarder transmission response.

    Plot the intensity transmited through a variable retarder fixed at the given
    orientation. This configuration is applicable to liquid crystal modulators
    and Pockels cells.
    """
    in_svec = get_stokes_vec('hlp')
    pol_mmat = get_mueller_mat('pol')

    det_ampl = np.empty_like(rtrd)

    for ind, rtrd1 in enumerate(rtrd):
        rtd_mmat = get_mueller_mat('rtd', theta, d=rtrd1)
        out_svec = pol_mmat.dot(rtd_mmat.dot(in_svec))
        det_ampl[ind] = out_svec[0]

    rtrd_deg = rtrd/np.pi*180
    plt.plot(rtrd_deg, det_ampl)

    if finalize_figure:
        add_y_marker(0)
        add_y_marker(1)
        plt.ylim([-0.1, 1.1])
        plt.xlim([np.min(rtrd_deg), np.max(rtrd_deg)])
        plt.grid('on')
        #plt.xticks(np.arange(0, 361, 45))
        if title_str is None:
            title_str = 'Variable retarder response'
        plt.title(title_str)
        plt.xlabel('Retardance, deg')
        plt.ylabel('Transmitted power, a.u.')


def plot_pockels_response(
        vctrl=None, theta=None, title_str=None, finalize_figure=True, with_qwp=False):
    """Plot Pockels cell transmission response.

    Plot the intensity transmited through a variable retarder fixed at the given
    orientation. This configuration is applicable to liquid crystal modulators
    and Pockels cells.
    """
    vhwp = 1000
    rtrd = vctrl/vhwp*np.pi

    in_svec = get_stokes_vec(10)
    pol_mmat = get_mueller_mat('pol')
    qwp_mmat = get_mueller_mat('qwp', np.pi/4)

    det_ampl = np.empty_like(rtrd)
    out_svec = np.ndarray([len(det_ampl), 4])

    for ind, rtrd1 in enumerate(rtrd):
        rtd_mmat = get_mueller_mat('rtd', theta, d=rtrd1)
        svec = in_svec
        if with_qwp:
            svec = qwp_mmat.dot(svec)
        svec = rtd_mmat.dot(svec)
        out_svec[ind, :] = np.transpose(svec)
        svec = pol_mmat.dot(svec)
        det_ampl[ind] = svec[0]

    plt.subplot(2, 1, 1)
    plt.plot(vctrl, det_ampl)
    add_y_marker(0)
    add_y_marker(1)
    plt.grid('on')
    if title_str is None:
        title_str = 'Pockels cell response (theta={:.0f} deg)'.format(theta/np.pi*180)
    plt.title(title_str)
    plt.ylabel('Transmitted power, a.u.')
    plt.ylim([-0.1, 1.1])
    plt.xlim([np.min(vctrl), np.max(vctrl)])

    plt.subplot(2, 1, 2)
    docp = np.abs(out_svec[:, 3])/out_svec[:, 0]
    dolp = 1 - docp
    plt.plot(vctrl, dolp)
    plt.plot(vctrl, docp)
    add_y_marker(0)
    add_y_marker(1)
    plt.grid('on')
    plt.ylabel('DOLP/DOCP')
    plt.ylim([-0.1, 1.1])
    plt.xlim([np.min(vctrl), np.max(vctrl)])
    plt.xlabel('Voltage, V')


def test_pol_trans():
    """Test polarizer transmission."""
    svec_in = get_stokes_vec("HLP")

    theta_arr = np.arange(0, pi, pi/100)
    trans = np.zeros_like(theta_arr)

    for (ind, theta) in enumerate(theta_arr):
        mat_pol = get_mueller_mat("POL", theta)
        svec_out = mat_pol * svec_in
        trans[ind] = svec_out[0]

    plt.plot(theta_arr/pi*180, trans)
