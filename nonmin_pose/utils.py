from typing import Optional, Tuple

import numpy as np

_W = np.array(
    [
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    ]
)


def skew(a):
    """Skew-symmetric matrix from a 3D vector (array) of shape (3, 1)."""
    return np.array(
        [
            [0, -a[2, 0], a[1, 0]],
            [a[2, 0], 0, -a[0, 0]],
            [-a[1, 0], a[0, 0], 0],
        ]
    )


def compute_data_matrix_C(
    f0: np.ndarray, f1: np.ndarray, w: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute the data matrix C from the bearing vectors.

    Args:
        f0: (3, n) bearing vectors in camera 0.
        f1: (3, n) bearing vectors in camera 1.
        w: (n,) array of weights for each epipolar residual. (default: None)

    Returns:
        C: (9, 9) data matrix.
    """
    assert w is None or w.ndim == 1, "w must be a 1D array."

    n = f0.shape[1]
    f0_kron_f1 = (n**-0.5) * (f0[:, None] * f1).reshape(9, n)

    if w is None:
        return f0_kron_f1 @ f0_kron_f1.T
    return w * f0_kron_f1 @ f0_kron_f1.T


def sdpa2mat(constraint, block_sizes):
    """Convert SDPA format to matrix form.

    Example usage:
        # instantiate the constraint manager with some parameters and constraints.
        cons = ConstraintManager(params, constraints)
        # function call.
        As = sdpa2mat(cons, list(cons.block_sizes.values()))

    Args:
        constraint: constraint manager instance.
        block_sizes: list of block sizes.

    Returns:
        As: (n_constraints, ndim, ndim) array of constraint matrices.
    """
    con_idx, blocks, values, rows, cols, coeffs = (
        constraint.constraint_idx,
        constraint.blocks,
        constraint.values,
        constraint.rows,
        constraint.cols,
        constraint.coeffs,
    )
    n_constraints = len(values)
    assert (np.unique(con_idx) == np.arange(1, n_constraints + 1)).all()
    assert (np.unique(blocks) == np.arange(1, len(block_sizes) + 1)).all()
    # dimensionality.
    ndim = sum(block_sizes)
    # initialize and fill matrices of constraints.
    As = np.zeros((n_constraints, ndim, ndim))
    for block, constraint, row, col, coef in zip(blocks, con_idx, rows, cols, coeffs):
        # 0-based indexing.
        block, constraint, row, col = block - 1, constraint - 1, row - 1, col - 1
        rc_offset = sum(block_sizes[:block])
        row += rc_offset
        col += rc_offset
        As[constraint, row, col] = coef
    return As


def rot_given_Etq(E, t, q):
    """Recover relative rotation given the essential matrix and the relative translation

    Assuming that we have tight solutions for E ∈ ME and t, q ∈ S2, we can directly
    recover the rotation matrix R ∈ SO(3) without requiring disambiguation. Recall that
    a (normalized) essential matrix satisfies E = [t] R, which depends linearly on R.
    Given that rank(E) = 2, this definition provides six linearly independent
    constraints for solving R (its nine parameters). Thus, three additional independent
    equations are needed. Notably, t lies in the nullspace of [t], allowing us to find
    the remaining linear equations from the definition q = R^Tt. Consequently, the
    elements of R can be computed as the solution of this linear system of equations.
    Since our solution is empirically tight, the estimates E, t and q belong to their
    respective spaces, implying that the solution R belongs to SO(3). While not
    theoretically necessary, for better numerical accuracy, we:
        1)  Consider the linearly dependent equations stemming from the definitions:
                E = [t] R,    E = R [q],    q = R^T t   and   t = R q,
            Thus, in essence, we estimate R through minimizing the (potentially
            negligible) square errors from the four above definitions.
            Fortunately, there is a fast closed-form solution to this problem, given by:
                R = t q^T - 0.5 * ([t] E + E [q])
            This follows from forming the the corresponding normal equations, and
            realizing that their LHS and RHS are given by:
                2 I_9  and   2 t q^T - [t] E - E[q], respectively.
        2)  Project the resulting matrix R to SO(3) by classical means.

    Args:
        E: (3, 3) essential matrix.
        t: (3, 1) relative translation.
        q: (3, 1) *rotated* relative translation i.e. q = R^T t.

    Returns:
        R: (3, 3) relative rotation.
    """
    R = t @ q.T - 0.5 * (skew(t) @ E + E @ skew(q))
    # project to SO(3).
    Ur, _, Vtr = np.linalg.svd(R)
    # TODO: This check ensuring proper rotations (avoiding reflections) should
    # only be needed when the solution is not certified as optimal.
    Vtr[2] = -Vtr[2] if np.linalg.det(Ur) * np.linalg.det(Vtr) < 0 else Vtr[2]
    R = Ur @ Vtr
    return R


def decompose_essmat(
    U: np.ndarray,
    Vt: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    th_pure_rotation: float = 1 - 1e-8,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """Decompose the essential matrix into relative rotation and (normalized)
    translation.

    The extraction of the 4 possible relative pose factorizations given an essential
    matrix, follows the approach explained in [1, Sec. 9.6.2].
    To select the best pose candidate, we check the sign of the factor that
    multiplies each bearing vector. This factor must be positive since a bearing
    vector is equivalent to $f_i := X_i / ||X_i||$, where $X_i$ is the corresponding
    3D point. Thus, to recover the 3D point, we multiply $f$ with an estimated
    scalar factor that *must* be positive. This constraint is independent of the
    camera model used (pinhole, fisheye, omnidirectional etc.), thus the camera
    model is not a limiting factor for this approach.
    To compute the scalar factor (the norm of X_i), we use the classic midpoint
    method (see e.g. [2]). However, instead of explicitly computing (triangulating)
    the 3D points, we just compute the sign of the scalar factors (lambdas). As a
    result, we save some computation. Specifically, we avoid computing:
    1) the term (sin angle(f0, R01@f1))^2 = || f0 x R01@f1 ||^2, for each point, and
    2) the XY coordinates of each 3D point.

    [1]: Multiple View Geometry in Computer Vision, Hartley and Zisserman, 2003.
    [2]: Triangulation: why optimize?, Lee and Civera, 2019.

    Args:
        U: (3, 3) left singular vectors of the essential matrix.
        Vt: (3, 3) right singular vectors of the essential matrix.
        f0: (3, n) bearing vectors in camera 0.
        f1: (3, n) bearing vectors in camera 1.
        th_pure_rotation: threshold for checking if the motion is a pure rotation.

    Returns:
        R: (3, 3) rotation matrix.
        t: (3, 1) translation vector.
        is_pure_rotation: True if a (near-)pure rotation is detected.
    """
    # avoid reflection (ensure rotation) when decomposing the essential matrix.
    Vt[2] = -Vt[2] if np.linalg.det(U) * np.linalg.det(Vt) < 0 else Vt[2]

    Ra, Rb = U @ _W @ Vt  # (2, 3, 3)
    ta, tb = U[:, 2:], -U[:, 2:]

    # check if it is a pure rotation.
    is_pure_rotation, choice = check_pure_rotation(
        f0, np.stack((Ra, Rb)) @ f1, th_pure_rotation
    )

    # (Ra, ta)
    Raf1 = Ra @ f1
    lambda0_rhs = (
        np.cross((Raf1).T, f0.T)[:, None] @ np.cross((Raf1).T, ta.T)[..., None]
    )
    lambda1_rhs = np.cross((Raf1).T, f0.T)[:, None] @ np.cross(f0.T, ta.T)[..., None]
    npos_aa = ((lambda0_rhs > 0) & (lambda1_rhs > 0)).sum()

    # (Rb, ta)
    Rbf1 = Rb @ f1
    lambda0_rhs = (
        np.cross((Rbf1).T, f0.T)[:, None] @ np.cross((Rbf1).T, ta.T)[..., None]
    )
    lambda1_rhs = np.cross((Rbf1).T, f0.T)[:, None] @ np.cross(f0.T, ta.T)[..., None]
    npos_ba = ((lambda0_rhs > 0) & (lambda1_rhs > 0)).sum()

    # (Ra, tb)
    lambda0_rhs = (
        np.cross((Raf1).T, f0.T)[:, None] @ np.cross((Raf1).T, tb.T)[..., None]
    )
    lambda1_rhs = np.cross((Raf1).T, f0.T)[:, None] @ np.cross(f0.T, tb.T)[..., None]
    npos_ab = ((lambda0_rhs > 0) & (lambda1_rhs > 0)).sum()

    # (Rb, tb)
    lambda0_rhs = (
        np.cross((Rbf1).T, f0.T)[:, None] @ np.cross((Rbf1).T, tb.T)[..., None]
    )
    lambda1_rhs = np.cross((Rbf1).T, f0.T)[:, None] @ np.cross(f0.T, tb.T)[..., None]
    npos_bb = ((lambda0_rhs > 0) & (lambda1_rhs > 0)).sum()

    npos_tpos = np.r_[npos_aa, npos_ba]
    npos_tneg = np.r_[npos_ab, npos_bb]

    if is_pure_rotation and (npos_tpos[choice] == npos_tneg[choice] == 0):
        # Pure rotation with perfect bearings alignment by just rotating them.
        R01 = Ra if choice == 0 else Rb
        return R01, ta, is_pure_rotation

    if is_pure_rotation:
        # Pure rotation with imperfect bearings alignment. Choose the translation
        # candidate that satisfies the most the positive-norm bearings' constraint.
        t01 = ta if npos_tpos[choice] >= npos_tneg[choice] else tb
        R01 = Ra if choice == 0 else Rb
        return R01, t01, is_pure_rotation

    # Otherwise, select the candidate that satisfies the most the positive-norm
    # bearings' constraint.
    choice, npos = max(
        enumerate((npos_tpos[0], npos_tpos[1], npos_tneg[0], npos_tneg[1])),
        key=lambda x: x[1],
    )

    t01 = ta if choice < 2 else tb
    R01 = Rb if choice % 2 else Ra

    return R01, t01, is_pure_rotation


def check_pure_rotation(
    f0: np.ndarray, Rf1: np.ndarray, th: float = 1 - 1e-8
) -> Tuple[bool, int]:
    """Rotationally-invariant metric for checking if the motion is a pure rotation

    If the motion is a pure rotation, then f0 is the same as R01 @ f1.
    To test this, we use the metric:
        f0^T (R01 @ f1), with f0 and f1 in R3 and R01 in SO3.
    This metric is rotationally invariant since:
        f0^T (R01 @ f1) = (R @ f0)^T (R @ (R01 @ f1)), for any R in SO3.
    As such, it only depends on the magnitude of the translation vector t01.
    For the threshold "th", we found these empirical **approximate**
    relations:
    | threshold "th" | translation magnitude w.r.t. scene |
    | 1 - 1e-14      | 1e-7                               |
    | 1 - 1e-12      | 1e-6                               |
    | 1 - 1e-10      | 1e-5                               |
    | 1 - 1e-8       | 1e-4                               |
    | 1 - 1e-6       | 1e-3                               |

    Args:
        f0: (3, n) bearings in camera 0.
        Rf1: (2, 3, n) bearings in camera 1, rotated by the candidates R01cand.

    Returns:
        is_pure_rotation: True if the motion is a pure rotation.
        choice: 0 or 1, depending on which rotation candidate is the correct motion.
    """
    metric = np.einsum("dn, cdn -> c", f0, Rf1) / f0.shape[1]
    choice, metric = max(enumerate((metric[0], metric[1])), key=lambda x: x[1])
    return metric > th, choice
