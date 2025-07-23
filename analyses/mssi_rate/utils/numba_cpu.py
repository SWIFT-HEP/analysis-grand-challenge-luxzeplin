

import numba
import numpy as np
import uproot as up

coeffs = np.array(
    [
        [
            -1.78880746e-13,
            4.91268301e-10,
            -4.96134607e-07,
            2.26430932e-04,
            -4.71792008e-02,
            7.33811298e01,
        ],
        [
            -1.72264463e-13,
            4.59149636e-10,
            -4.59325165e-07,
            2.14612376e-04,
            -4.85599108e-02,
            7.35290867e01,
        ],
        [
            -3.17099156e-14,
            7.26336129e-11,
            -6.99495385e-08,
            3.85531008e-05,
            -1.33386004e-02,
            7.18002889e01,
        ],
        [
            -6.12280314e-14,
            1.67968911e-10,
            -1.83625538e-07,
            1.00457608e-04,
            -2.86728022e-02,
            7.22754350e01,
        ],
        [
            -1.89897962e-14,
            1.52777215e-11,
            -2.79681508e-09,
            1.25689887e-05,
            -1.33093804e-02,
            7.17662251e01,
        ],
        [
            -2.32118621e-14,
            7.30043322e-11,
            -9.40606298e-08,
            6.29728588e-05,
            -2.28150175e-02,
            7.22661091e01,
        ],
        [
            -8.29749194e-14,
            2.31096069e-10,
            -2.47867121e-07,
            1.27576029e-04,
            -3.24702414e-02,
            7.26357609e01,
        ],
        [
            -2.00718008e-13,
            5.44135757e-10,
            -5.59484466e-07,
            2.73028553e-04,
            -6.46879791e-02,
            7.45264998e01,
        ],
        [
            -7.77420021e-14,
            1.97357045e-10,
            -1.90016273e-07,
            8.99659454e-05,
            -2.30169916e-02,
            7.25038258e01,
        ],
        [
            -5.27296334e-14,
            1.49415580e-10,
            -1.58205132e-07,
            8.00275441e-05,
            -2.13559394e-02,
            7.23995451e01,
        ],
        [
            -6.00198219e-14,
            1.55333004e-10,
            -1.60367908e-07,
            7.97754165e-05,
            -1.94435594e-02,
            7.22714399e01,
        ],
        [
            -8.89919309e-14,
            2.40830027e-10,
            -2.57060475e-07,
            1.33002951e-04,
            -3.32969110e-02,
            7.28696020e01,
        ],
    ]
)

@numba.njit
def fv_wall_contour(dt):
    c = [
        -4.44147071e-14,
        1.43684777e-10,
        -1.82739476e-07,
        1.02160174e-04,
        -2.31617857e-02,
        -2.05932471e00,
    ]
    wall_r2 = (
        c[0] * dt**5 + c[1] * dt**4 + c[2] * dt**3 + c[3] * dt**2 + c[4] * dt + c[5]
    )
    return wall_r2


n_phi_slices = 12
phi_slices = np.linspace(-np.pi, np.pi, n_phi_slices + 1) + np.pi / 4
phi_slices[phi_slices > np.pi] -= 2 * np.pi


@numba.njit
def calc_dR_phi(x, y, dt):

    # Calculate event radii and angles, then mask them according to each slice
    R = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    dR_phi = 0.0
    # Process each phi slice
    for slice in range(n_phi_slices):
        phi_min, phi_max = phi_slices[slice], phi_slices[slice + 1]
        if (phi >= phi_min) & (phi < phi_max):
            dR_phi = R - perform_poly(dt, slice)

    return dR_phi

@numba.njit
def perform_poly(dt, c):
    poly_results = (
        coeffs[c][0] * dt**5
        + coeffs[c][1] * dt**4
        + coeffs[c][2] * dt**3
        + coeffs[c][3] * dt**2
        + coeffs[c][4] * dt
        + coeffs[c][5]
    )
    return poly_results


@numba.njit
def fv_phi_r(x, y, dt):
    dR_phi = calc_dR_phi(x, y, dt)
    contour = fv_wall_contour(dt) - 3  # add cm stand-off
    expandable = (dt > 71) & (dt < 900)

    mask = ((dR_phi < contour) & expandable) | ((dR_phi < contour) & ~expandable)
    return mask & (dR_phi <= 0)


@numba.njit
def fv_resistor(x, y):
    res1X, res1Y, res1R = -69.8, 3.5, 6
    res2X, res2Y, res2R = -67.5, -14.3, 6

    insideRes1 = (x - res1X) * (x - res1X) + (y - res1Y) * (y - res1Y) > res1R**2
    insideRes2 = (x - res2X) * (x - res2X) + (y - res2Y) * (y - res2Y) > res2R**2

    return insideRes1 & insideRes2


@numba.njit
def fv_z(dt):
    return (dt > 71) & (dt < 1030)


@numba.njit
def roi(s1c, s2, s2c):
    return (s1c > 3.0) and (s1c < 80.0) and (s2 > 14.5 * 44.5) and (s2c < 10**4.5)


@numba.jit
def process_events(ss_x, ss_y, ss_dt, ss_s1c, ss_s2, ss_s2c, mc_detS1, mc_detS2):
    n_fv, n_roi, n_fv_roi = 0, 0, 0
    n_mssi, n_fv_mssi, n_roi_mssi, n_fv_roi_mssi = 0, 0, 0, 0

    # Loop over events
    for i in range(len(ss_x)):

        # Evaluate cuts
        cut_bool_resistor_fv = fv_resistor(ss_x[i], ss_y[i])
        cut_bool_z = fv_z(ss_dt[i] / 1000)
        cut_bool_r = fv_phi_r(ss_x[i], ss_y[i], ss_dt[i] / 1000)
        fv_cut = cut_bool_resistor_fv and cut_bool_z and cut_bool_r
        roi_cut = roi(ss_s1c[i], ss_s2[i], ss_s2c[i])

        # Examine if MSSI
        nS1, nS2 = 0, 0
        for j in range(len(mc_detS1[i])):
            if mc_detS1[i][j] > 0.0:
                nS1 += 1
            if mc_detS2[i][j] > 0.0:
                nS2 += 1
        is_mssi = nS1 > nS2

        # Now evaluate cuts and count events
        if is_mssi:
            n_mssi += 1
        if fv_cut:
            n_fv += 1
            if is_mssi:
                n_fv_mssi += 1
            if roi_cut:
                n_fv_roi += 1
                if is_mssi:
                    n_fv_roi_mssi += 1
        if roi_cut:
            n_roi += 1
            if is_mssi:
                n_roi_mssi += 1

    return n_fv, n_roi, n_fv_roi, n_mssi, n_fv_mssi, n_roi_mssi, n_fv_roi_mssi


def process_file(file):

    tfile = up.open(file)
    scatters = tfile["Scatters"]
    truth = tfile["RQMCTruth"]

    # Read arrays
    ss_x = scatters["ss.x_cm"].array()
    ss_y = scatters["ss.y_cm"].array()
    ss_dt = scatters["ss.driftTime_ns"].array()
    ss_s1c = scatters["ss.correctedS1Area_phd"].array()
    ss_s2 = scatters["ss.s2Area_phd"].array()
    ss_s2c = scatters["ss.correctedS2Area_phd"].array()

    mc_detS1 = truth["mcTruthVertices.detectedS1Photons"].array()
    mc_detS2 = truth["mcTruthVertices.detectedS2Photons"].array()

    n_events = len(ss_x)

    result = process_events(
        ss_x, ss_y, ss_dt, ss_s1c, ss_s2, ss_s2c, mc_detS1, mc_detS2
    )
    n_fv, n_roi, n_fv_roi, n_mssi, n_fv_mssi, n_roi_mssi, n_fv_roi_mssi = result

    return (
        file,
        n_events,
        n_fv,
        n_roi,
        n_fv_roi,
        n_mssi,
        n_fv_mssi,
        n_roi_mssi,
        n_fv_roi_mssi,
    )