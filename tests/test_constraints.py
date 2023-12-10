import numpy as np

from nonmin_pose.constraints import constraints
from nonmin_pose.constraints.constraints import Parameter
from tests.testing_utils import (
    SyntheticData,
    adjoint_of_3x3_mat,
    sdpa2mat,
    skew,
    so3_orbitope,
)

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


def create_parameters():
    params = [
        Parameter("E", 1, list(range(1, 10))),
        Parameter("t", 1, list(range(10, 13))),
        Parameter("q", 1, list(range(13, 16))),
        Parameter("h", 1, [16]),
        Parameter("R", 1, list(range(17, 26))),
        Parameter("sct", 1, [26]),
        Parameter("scr", 1, [27]),
        Parameter("scr2", 1, [28]),
        Parameter("scm1", 1, [29]),
        Parameter("scm2", 1, [30]),
        Parameter("Zc", 1, list(range(31, 47))),
    ]
    return {p.name: p for p in params}


def sample_data():
    dataset = SyntheticData(**CFG_DATASET)
    data = dataset.generate_data(**CFG_DATA)
    h, sct, scr, scr2, scm1, scm2 = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
    q = data["R01"].T @ data["t01_unit"]
    x = np.concatenate(
        (
            data["E01"].ravel(),
            data["t01_unit"].ravel(),
            q.ravel(),
            [h],
            data["R01"].ravel(),
            [sct, scr, scr2, scm1, scm2],
            so3_orbitope(data["R01"]).ravel(),
        )
    )
    return x[:, None], data


def gather_errors(x, A, constraint, constraint_num, is_inequality):
    values = constraint.values
    if is_inequality:
        cond_sdpa_sdpa = np.allclose(values, np.zeros_like(values))
        cond_data_sdpa = np.allclose((x.T @ A @ x).squeeze(), constraint_num)
    else:
        cond_sdpa_sdpa = np.allclose((x.T @ A @ x).squeeze(), values)
        cond_data_sdpa = np.allclose(constraint_num, values)

    errors = []
    if not cond_sdpa_sdpa:
        if is_inequality:
            errors.append("SDPA coefficients are not zero.")
        else:
            errors.append("SDPA coefficients lead to different SDPA values.")
    if not cond_data_sdpa:
        errors.append(
            "SDPA values are different than those derived from data."
            f"\n{(x.T @ A @ x).squeeze()}\n{constraint_num}"
        )
    success = len(errors) == 0
    err_msg = "Errors:\n{}".format("\n".join(errors))
    return success, err_msg


def obtain_errors(constraint_class, x, constraint_num, f0=None, f1=None):
    params = create_parameters()
    constraint = constraint_class(params, 0, 0, None)

    is_inequality = constraint.__class__.__name__.startswith("Cheirality")
    if is_inequality:
        constraint.compute_coeffs(constraint.coeffs, f0, f1)

    A = sdpa2mat(constraint, block_sizes=[len(x)], ndim=len(x))

    errors = gather_errors(x, A, constraint, constraint_num, is_inequality)
    return errors


def test_manif_def_left():
    x, data = sample_data()

    E01, t01_unit = data["E01"], data["t01_unit"]
    constraint_num = E01 @ E01.T - skew(t01_unit) @ skew(t01_unit).T
    constraint_num = constraint_num.ravel()[[0, 4, 8, 1, 2, 5]]

    success, err_msg = obtain_errors(constraints.ManifDefLeft, x, constraint_num)
    assert success, err_msg


def test_manif_def_right():
    x, data = sample_data()

    E01 = data["E01"]
    q = data["R01"].T @ data["t01_unit"]
    constraint_num = E01.T @ E01 - skew(q).T @ skew(q)
    constraint_num = constraint_num.ravel()[[0, 4, 8, 1, 2, 5]]

    success, err_msg = obtain_errors(constraints.ManifDefRight, x, constraint_num)
    assert success, err_msg


def test_normt():
    x, data = sample_data()

    t01_unit = data["t01_unit"]
    constraint_num = t01_unit.T @ t01_unit

    success, err_msg = obtain_errors(constraints.NormT, x, constraint_num)
    assert success, err_msg


def test_normq():
    x, data = sample_data()

    q = data["R01"].T @ data["t01_unit"]
    constraint_num = q.T @ q

    success, err_msg = obtain_errors(constraints.NormQ, x, constraint_num)
    assert success, err_msg


def test_e_def_left():
    x, data = sample_data()

    E01, t01, R01 = data["E01"], data["t01_unit"], data["R01"]
    constraint_num = (E01 - skew(t01) @ R01).ravel()

    success, err_msg = obtain_errors(constraints.EDefLeft, x, constraint_num)
    assert success, err_msg


def test_e_def_right():
    x, data = sample_data()

    E01, R01 = data["E01"], data["R01"]
    q = R01.T @ data["t01_unit"]
    constraint_num = (E01 - R01 @ skew(q)).ravel()

    success, err_msg = obtain_errors(constraints.EDefRight, x, constraint_num)
    assert success, err_msg


def test_e_def_left_right():
    x, data = sample_data()

    R01, t01 = data["R01"], data["t01_unit"]
    q = R01.T @ t01
    constraint_num = (skew(t01) @ R01 - R01 @ skew(q)).ravel()

    success, err_msg = obtain_errors(constraints.EDefLeftRight, x, constraint_num)
    assert success, err_msg


def test_homogenization():
    x, _ = sample_data()
    constraint_num = 1.0
    success, err_msg = obtain_errors(constraints.Homogenization, x, constraint_num)
    assert success, err_msg


def test_adjoint():
    x, data = sample_data()

    E01, t01 = data["E01"], data["t01_unit"]
    q = data["R01"].T @ t01
    adjoint = adjoint_of_3x3_mat(E01)
    constraint_num = (adjoint - q @ t01.T).T.ravel()

    success, err_msg = obtain_errors(constraints.Adjoint, x, constraint_num)
    assert success, err_msg


def test_norm_e():
    x, data = sample_data()

    E01 = data["E01"]
    constraint_num = E01.ravel().dot(E01.ravel())

    success, err_msg = obtain_errors(constraints.NormE, x, constraint_num)
    assert success, err_msg


def test_right_null_space():
    x, data = sample_data()

    E01 = data["E01"]
    q = data["R01"].T @ data["t01_unit"]
    constraint_num = (E01 @ q).ravel()

    success, err_msg = obtain_errors(constraints.RightNullSpace, x, constraint_num)
    assert success, err_msg


def test_left_null_space():
    x, data = sample_data()

    E01, t01 = data["E01"], data["t01_unit"]
    constraint_num = (t01.T @ E01).ravel()

    success, err_msg = obtain_errors(constraints.LeftNullSpace, x, constraint_num)
    assert success, err_msg


def test_cheirality_translation():
    x, data = sample_data()

    t01, R01 = data["t01_unit"], data["R01"]
    f0, f1 = data["f0"], data["f1"]
    q = R01.T @ t01
    f0_agg, f1_agg = f0.sum(1, keepdims=True), f1.sum(1, keepdims=True)

    constraint_num = (f0_agg.T @ R01 @ q - t01.T @ R01 @ f1_agg).ravel()

    success, err_msg = obtain_errors(
        constraints.CheiralityTranslation, x, constraint_num, f0, f1
    )
    assert success, err_msg


def test_cheirality_translation_v2():
    x, data = sample_data()

    t01 = data["t01_unit"]
    f0, f1 = data["f0"], data["f1"]
    q = data["R01"].T @ t01
    f0_agg, f1_agg = f0.mean(1, keepdims=True), f1.mean(1, keepdims=True)

    constraint_num = (f0_agg.T @ t01 - q.T @ f1_agg).ravel()

    success, err_msg = obtain_errors(
        constraints.CheiralityTranslationV2, x, constraint_num, f0, f1
    )
    assert success, err_msg


def test_cheirality_rotation():
    x, data = sample_data()

    E01, t01 = data["E01"], data["t01_unit"]
    f0, f1 = data["f0"], data["f1"]
    n = f0.shape[1]

    E_skewt = E01.T @ skew(t01)
    constraint_num = (
        sum((f1i[None] @ E_skewt @ f0i[:, None])[0, 0] for f0i, f1i in zip(f0.T, f1.T))
        / n
    )

    success, err_msg = obtain_errors(
        constraints.CheiralityRotation, x, constraint_num, f0, f1
    )
    assert success, err_msg


def test_cheirality_rotation_q():
    x, data = sample_data()

    E01 = data["E01"]
    q = data["R01"].T @ data["t01_unit"]
    f0, f1 = data["f0"], data["f1"]

    E_skewq = E01 @ skew(q)
    constraint_num = sum(
        (f0i[None] @ E_skewq @ f1i[:, None])[0, 0] for f0i, f1i in zip(f0.T, f1.T)
    )

    success, err_msg = obtain_errors(
        constraints.CheiralityRotationQ, x, constraint_num, f0, f1
    )
    assert success and constraint_num <= 0, err_msg


def test_cheirality_midpoint():
    x, data = sample_data()

    t01, R01 = data["t01_unit"], data["R01"]
    f0, f1 = data["f0"], data["f1"]
    q = R01.T @ t01

    # fmt:off
    sum0 = sum(
        (
        -(t01.T @ R01 @ f1i[:, None])
        + (f0i[None] @ R01 @ f1i[:, None]) * (f0i[None] @ t01)
        )[0, 0]
        for f0i, f1i in zip(f0.T, f1.T)
    )
    sum1 = sum(
        (
        (f0i[None] @ R01 @ q)
        - (f1i[None] @ q) * (f0i[None] @ R01 @ f1i[:, None])
        )[0, 0]
        for f0i, f1i in zip(f0.T, f1.T)
    )
    # fmt:on
    constraint_num = np.array([sum0, sum1])

    success, err_msg = obtain_errors(
        constraints.CheiralityMidpoint, x, constraint_num, f0, f1
    )
    assert success, err_msg


def test_orthogonality():
    x, data = sample_data()

    R01 = data["R01"]

    constraint_num = np.concatenate(
        (
            (R01 @ R01.T).ravel()[[0, 4, 8, 1, 2, 5]],
            (R01.T @ R01).ravel()[[4, 8, 1, 2, 5]],
        )
    )
    success, err_msg = obtain_errors(constraints.Orthogonality, x, constraint_num)
    assert success, err_msg


def test_determinant_R():
    x, data = sample_data()

    R01 = data["R01"]
    constraint_num = (R01 - adjoint_of_3x3_mat(R01).T).ravel()

    success, err_msg = obtain_errors(constraints.DeterminantR, x, constraint_num)
    assert success, err_msg


def test_t_q_definition():
    x, data = sample_data()

    t01, R01 = data["t01_unit"], data["R01"]
    q = R01.T @ t01
    constraint_num = np.concatenate((t01 - R01 @ q, q - R01.T @ t01))[:, 0]

    success, err_msg = obtain_errors(constraints.TQDefinition, x, constraint_num)
    assert success, err_msg


def test_skew_t_q_definition():
    x, data = sample_data()

    E01, t01, R01 = data["E01"], data["t01_unit"], data["R01"]
    q = R01.T @ t01
    constraint_num = np.concatenate(
        ((skew(t01) - E01 @ R01.T).ravel(), (skew(q) - R01.T @ E01).ravel())
    )

    success, err_msg = obtain_errors(constraints.SkewTQDefinition, x, constraint_num)
    assert success, err_msg


def test_convex_hull_so3():
    x, data = sample_data()

    R01 = data["R01"]
    orbitope = so3_orbitope(R01)
    min_eig = np.linalg.eigvalsh(orbitope).min()
    constraint_num = np.eye(4).ravel()[[0, 1, 2, 3, 5, 6, 7, 10, 11, 15]]

    success, err_msg = obtain_errors(constraints.ConvexHullSO3, x, constraint_num)
    assert success and min_eig >= -1e-12, err_msg
