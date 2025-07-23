"""
GPU functions using numba (cuda)
"""

import numba
from numba import cuda
import math
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

@cuda.jit
def fv_wall_contour_cuda(dt):
    """
    Calculate contour at a given drift time
    """
    return (
        -4.44147071e-14 * math.pow(dt, 5)
        + 1.43684777e-10 * math.pow(dt, 4)
        - 1.82739476e-07 * math.pow(dt, 3)
        + 1.02160174e-04 * math.pow(dt, 2)
        - 2.31617857e-02 * dt
        - 2.05932471e00
    )


@cuda.jit
def calc_dR_phi_cuda(x, y, dt):
    R = math.sqrt(x**2 + y**2)
    phi = math.atan2(y, x)

    n_phi_slices = 12
    # Now define array on device
    phi_slices = cuda.local.array(13, dtype=numba.float32)
    phi_slices[0] = -2.35619449
    phi_slices[1] = -1.83259571
    phi_slices[2] = -1.30899694
    phi_slices[3] = -0.78539816
    phi_slices[4] = -0.26179939
    phi_slices[5] = 0.26179939
    phi_slices[6] = 0.78539816
    phi_slices[7] = 1.30899694
    phi_slices[8] = 1.83259571
    phi_slices[9] = 2.35619449
    phi_slices[10] = 2.87979327
    phi_slices[11] = -2.87979327
    phi_slices[12] = -2.35619449

    for slice in range(n_phi_slices):
        phi_min, phi_max = phi_slices[slice], phi_slices[slice + 1]
        if (phi >= phi_min) and (phi < phi_max):
            return R - perform_poly_cuda(dt, slice)


# copy coeffs to cuda shared memory
cuda.to_device(coeffs)

@cuda.jit
def perform_poly_cuda(dt, c):

    return (
        coeffs[c][0] * math.pow(dt, 5)
        + coeffs[c][1] * math.pow(dt, 4)
        + coeffs[c][2] * math.pow(dt, 3)
        + coeffs[c][3] * math.pow(dt, 2)
        + coeffs[c][4] * dt
        + coeffs[c][5]
    )


@cuda.jit
def fv_phi_r_cuda(x, y, dt):
    dR_phi = calc_dR_phi_cuda(x, y, dt)
    contour = fv_wall_contour_cuda(dt) - 3  # add cm stand-off
    expandable = dt > 71 and dt < 900

    mask = ((dR_phi < contour) and expandable) or (
        (dR_phi < contour) and not expandable
    )
    return mask and dR_phi <= 0


@cuda.jit
def fv_resistor_cuda(x, y):
    res1X, res1Y, res1R = -69.8, 3.5, 6
    res2X, res2Y, res2R = -67.5, -14.3, 6

    insideRes1 = ((x - res1X) ** 2 + (y - res1Y) ** 2) > res1R**2
    insideRes2 = ((x - res2X) ** 2 + (y - res2Y) ** 2) > res2R**2

    return insideRes1 and insideRes2


@cuda.jit
def fv_z_cuda(dt):
    return (dt > 71) and (dt < 1030)


@cuda.jit
def roi_cuda(s1c, s2, s2c):
    return (s1c > 3.0) and (s1c < 80.0) and (s2 > 14.5 * 44.5) and (s2c < 10**4.5)



@cuda.jit
def process_events_cuda(
    ss_x,
    ss_y,
    ss_dt,
    ss_s1c,
    ss_s2,
    ss_s2c,
    mc_detS1,
    mc_detS2,
    n_fv,
    n_roi,
    n_fv_roi,
    n_mssi,
    n_fv_mssi,
    n_roi_mssi,
    n_fv_roi_mssi,
):

    # Set thread
    i = numba.cuda.grid(1)
    # Don't process out of bounds
    if i >= ss_x.shape[0]:
        return

    # Evaluate cuts
    cut_bool_resistor_fv = fv_resistor_cuda(ss_x[i], ss_y[i])
    cut_bool_z = fv_z_cuda(ss_dt[i] / 1000)
    cut_bool_r = fv_phi_r_cuda(ss_x[i], ss_y[i], ss_dt[i] / 1000)
    fv_cut = cut_bool_resistor_fv and cut_bool_z and cut_bool_r
    roi_cut = roi_cuda(ss_s1c[i], ss_s2[i], ss_s2c[i])

    # Examine if MSSI
    nS1, nS2 = 0, 0
    for j in range(len(mc_detS1[i])):
        if mc_detS1[i][j] > 0.0:
            nS1 += 1
        if mc_detS2[i][j] > 0.0:
            nS2 += 1
    is_mssi = nS1 > nS2

    # Now evaluate cuts and count events
    if roi_cut:
        cuda.atomic.add(n_roi, 0, 1)
        if is_mssi:
            cuda.atomic.add(n_roi_mssi, 0, 1)

    if is_mssi:
        cuda.atomic.add(n_mssi, 0, 1)
    if fv_cut:
        cuda.atomic.add(n_fv, 0, 1)
        if is_mssi:
            cuda.atomic.add(n_fv_mssi, 0, 1)
        if roi_cut:
            cuda.atomic.add(n_fv_roi, 0, 1)
            if is_mssi:
                cuda.atomic.add(n_fv_roi_mssi, 0, 1)


def process_file_cuda(file):

    tfile = up.open(file)
    scatters = tfile["Scatters"]
    truth = tfile["RQMCTruth"]

    # Read arrays and copy to GPU
    ss_x = cuda.to_device(scatters["ss.x_cm"].array())
    ss_y = cuda.to_device(scatters["ss.y_cm"].array())
    ss_dt = cuda.to_device(scatters["ss.driftTime_ns"].array())
    ss_s1c = cuda.to_device(scatters["ss.correctedS1Area_phd"].array())
    ss_s2 = cuda.to_device(scatters["ss.s2Area_phd"].array())
    ss_s2c = cuda.to_device(scatters["ss.correctedS2Area_phd"].array())

    # Need to zero pad jagged arrays
    det_s1 = truth["mcTruthVertices.detectedS1Photons"].array()
    det_s2 = truth["mcTruthVertices.detectedS2Photons"].array()
    pad_size = ak.max(ak.num(det_s1))
    mc_detS1 = cuda.to_device(ak.fill_none(ak.pad_none(det_s1, pad_size), 0))
    mc_detS2 = cuda.to_device(ak.fill_none(ak.pad_none(det_s2, pad_size), 0))

    # Result counters
    n_fv = cuda.to_device(np.zeros(1, dtype=np.int32))
    n_roi = cuda.to_device(np.zeros(1, dtype=np.int32))
    n_fv_roi = cuda.to_device(np.zeros(1, dtype=np.int32))
    n_mssi = cuda.to_device(np.zeros(1, dtype=np.int32))
    n_fv_mssi = cuda.to_device(np.zeros(1, dtype=np.int32))
    n_roi_mssi = cuda.to_device(np.zeros(1, dtype=np.int32))
    n_fv_roi_mssi = cuda.to_device(np.zeros(1, dtype=np.int32))

    # Launch kernel
    n_events = len(scatters["ss.x_cm"].array())
    block_size = 256
    n_blocks = (n_events + block_size - 1) // block_size

    process_events_cuda[n_blocks, block_size](
        ss_x,
        ss_y,
        ss_dt,
        ss_s1c,
        ss_s2,
        ss_s2c,
        mc_detS1,
        mc_detS2,
        n_fv,
        n_roi,
        n_fv_roi,
        n_mssi,
        n_fv_mssi,
        n_roi_mssi,
        n_fv_roi_mssi,
    )

    # Copy result back
    cuda.synchronize()
    n_fv_host = n_fv.copy_to_host()[0]
    n_roi_host = n_roi.copy_to_host()[0]
    n_fv_roi_host = n_fv_roi.copy_to_host()[0]
    n_mssi_host = n_mssi.copy_to_host()[0]
    n_fv_mssi_host = n_fv_mssi.copy_to_host()[0]
    n_roi_mssi_host = n_roi_mssi.copy_to_host()[0]
    n_fv_roi_mssi_host = n_fv_roi_mssi.copy_to_host()[0]

    return (
        file,
        n_events,
        n_fv_host,
        n_roi_host,
        n_fv_roi_host,
        n_mssi_host,
        n_fv_mssi_host,
        n_roi_mssi_host,
        n_fv_roi_mssi_host,
    )