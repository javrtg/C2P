"""Simple tests to check whether the solvers work as expected."""

import numpy as np

from nonmin_pose import (
    C2P,
    C2PFast,
    EssentialGSalguero,
    EssentialZhao,
    sdp_zhao,  # type: ignore
)
from nonmin_pose.utils import compute_data_matrix_C
from tests.testing_utils import SyntheticData

CFG_DATASET = {
    "seed": 0,
    "min_depth": 4.0,
    "max_depth": 8.0,
    "focal": 800.0,
}
CFG_DATA = {
    "transl_magnitude": 1.0,
    "euler_ang_magnitude": 0.5,
    "max_npoints": 100,
    "noise_level": 0.0,
}


def epipolar_error(E, C):
    e = E.reshape(9, 1)
    return e.T @ C @ e


def test_essmat_zhao():
    dataset = SyntheticData(**CFG_DATASET)
    data = dataset.generate_data(**CFG_DATA)

    essmat_solver = EssentialZhao()
    sol = essmat_solver(data["f0"], data["f1"])
    E01 = sol["E01"]

    C = compute_data_matrix_C(data["f0"], data["f1"])
    error = epipolar_error(E01, C)
    assert error < 1e-6


def test_essmat_gsalguero():
    dataset = SyntheticData(**CFG_DATASET)
    data = dataset.generate_data(**CFG_DATA)

    essmat_solver = EssentialGSalguero()
    sol = essmat_solver(data["f0"], data["f1"])
    E01 = sol["E01"]

    C = compute_data_matrix_C(data["f0"], data["f1"])
    error = epipolar_error(E01, C)
    assert error < 1e-6


def test_c2p():
    dataset = SyntheticData(**CFG_DATASET)
    data = dataset.generate_data(**CFG_DATA)

    c2p_solver = C2P()
    sol = c2p_solver(data["f0"], data["f1"])
    E01 = sol["E01"]

    C = compute_data_matrix_C(data["f0"], data["f1"])
    error = epipolar_error(E01, C)
    assert error < 1e-6


def test_c2p_fast():
    dataset = SyntheticData(**CFG_DATASET)
    data = dataset.generate_data(**CFG_DATA)

    c2p_solver = C2PFast()
    sol = c2p_solver(data["f0"], data["f1"])
    E01 = sol["E01"]

    C = compute_data_matrix_C(data["f0"], data["f1"])
    error = epipolar_error(E01, C)
    assert error < 1e-6


def test_sdp_zhao():
    dataset = SyntheticData(**CFG_DATASET)
    data = dataset.generate_data(**CFG_DATA)

    C = compute_data_matrix_C(data["f0"], data["f1"])

    X = sdp_zhao.solve(C)
    _, _, Vt_e = np.linalg.svd(X[:9, :9])
    E01 = Vt_e[0].reshape(3, 3)

    error = epipolar_error(E01, C)
    assert error < 1e-6
