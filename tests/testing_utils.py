import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(v):
    out = np.zeros((3, 3))
    out[0, 1] = -v[2, 0]
    out[0, 2] = v[1, 0]
    out[1, 0] = v[2, 0]
    out[1, 2] = -v[0, 0]
    out[2, 0] = -v[1, 0]
    out[2, 1] = v[0, 0]
    return out


class SyntheticData:
    """Data generation based on [1, Sec. 7.2.1] and [2].

    [1] An Efficient Solution to Non-Minimal Case Essential Matrix Estimation, J.Zhao.
    [2] https://github.com/jizhaox/npt-pose/blob/master/src/create2D2DExperiment.cpp
    """

    def __init__(self, seed=0, min_depth=4.0, max_depth=8.0, focal=800.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.focal = focal  # pixels

    def generate_data(
        self,
        transl_magnitude=2.0,
        euler_ang_magnitude=0.5,
        max_npoints=200,
        noise_level=0.0,
        scale_t=None,
        Rw1=None,
        tw1=None,
    ):
        """Generate synthetic data."""
        # absolute camera poses (w.r.t. world "w" reference).
        Rw0, tw0 = self.cam0_absolute_pose
        Rw1, tw1 = self.set_cam1_absolute_pose(
            transl_magnitude, euler_ang_magnitude, scale_t, Rw1, tw1
        )
        # relative pose such that p0 = R01 * p1 + t01.
        R01, t01, t01_unit = self.compute_relative_pose(Rw0, tw0, Rw1, tw1, scale_t)
        E01 = skew(t01_unit) @ R01
        f0_noisy, f1_noisy = self.generate_bearings(
            Rw0, tw0, Rw1, tw1, max_npoints, noise_level
        )
        return {
            "f0": f0_noisy,
            "f1": f1_noisy,
            "R01": R01,
            "t01": t01,
            "t01_unit": t01_unit,
            "E01": E01,
        }

    def generate_bearings(self, Rw0, tw0, Rw1, tw1, max_npoints, noise_level):
        # generate 3D points sampling from a unit cube.
        pw = self.generate_absolute_3d_points(max_npoints)

        # transform points to each camera reference.
        p0 = Rw0.T @ pw - Rw0.T @ tw0
        p1 = Rw1.T @ pw - Rw1.T @ tw1

        # corresponding bearing vectors.
        f0 = p0 / np.linalg.norm(p0, axis=0)
        f1 = p1 / np.linalg.norm(p1, axis=0)

        # add noise to the bearing vectors.
        f0_noisy = self.add_noise_to_bearings(f0, max_npoints, noise_level)
        f1_noisy = self.add_noise_to_bearings(f1, max_npoints, noise_level)
        return f0_noisy, f1_noisy

    def generate_absolute_3d_points(self, max_npoints):
        """Sample 3D points sampling from a unit cube."""
        unit_cube = self.rng.uniform(-0.5, 0.5, (3, max_npoints))
        directions = unit_cube / np.linalg.norm(unit_cube, axis=0)
        magnitudes = self.rng.uniform(self.min_depth, self.max_depth, (1, max_npoints))
        pw = magnitudes * directions
        return pw

    def add_noise_to_bearings(self, f, n, noise_level):
        """Add noise to each bearing vector assuming spherical cameras.

        The noise, in pixels, is added in the tangent plane of each bearing. The
        distance of each tangent plane is determined by the focal length of the camera.
        """
        cols_idx = np.arange(n)

        max_args, min_args = np.abs(f).argmax(0), np.abs(f).argmin(0)
        max_vals, min_vals = f[max_args, cols_idx], f[min_args, cols_idx]

        # first perpendicular vector.
        ortho_a = np.zeros((3, n))
        ortho_a[min_args, cols_idx] = 1.0
        ortho_a[max_args, cols_idx] = -min_vals / max_vals
        ortho_a = ortho_a / np.linalg.norm(ortho_a, axis=0)

        # second perpendicular vector.
        ortho_b = np.cross(f, ortho_a, axis=0)

        # add gaussian noise to each bearing.
        noise = self.rng.normal(0, noise_level, (2, n))
        f_noisy = self.focal * f + noise[0] * ortho_a + noise[1] * ortho_b
        f_noisy = f_noisy / np.linalg.norm(f_noisy, axis=0)
        return f_noisy

    def set_cam1_absolute_pose(
        self, transl_magnitude, euler_ang_magnitude, scale_t, Rw1, tw1
    ):
        """camera 1 pose (w.r.t. world "w" reference)."""
        if Rw1 is None:
            euler_angles = self.rng.uniform(
                -euler_ang_magnitude, euler_ang_magnitude, (3,)
            )
            Rw1 = R.from_euler("zyx", euler_angles).as_matrix()

        if tw1 is None:
            tw1 = transl_magnitude * self.rng.uniform(-1, 1, (3, 1))

        if scale_t is not None:
            # set translation magnitude, useful e.g. for accuracy vs translation length.
            tw1 = tw1 / np.linalg.norm(tw1) * scale_t
        return Rw1, tw1

    def compute_relative_pose(self, Rw0, tw0, Rw1, tw1, scale_t):
        """Compute relative pose such that p0 = R01 * p1 + t01."""
        R01 = Rw0.T @ Rw1
        t01 = Rw0.T @ (tw1 - tw0)
        if scale_t is None or scale_t > 0:
            t01_unit = t01 / np.linalg.norm(t01)
        else:
            # when there is pure rotation, any unit translation would satisfy the
            # epipolar constraint, e.g. we set it here to the x-axis unit vector.
            t01_unit = np.array([[1.0], [0], [0]])
        return R01, t01, t01_unit

    @property
    def cam0_absolute_pose(self):
        """Camera 0 pose (w.r.t. world "w" reference)."""
        return np.eye(3), np.zeros((3, 1))


def sdpa2mat(constraint, block_sizes=[29], ndim=29):
    """Converts SDPA format to matrix form."""
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
    assert ndim == sum(block_sizes)

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


def adjoint_of_3x3_mat(E):
    """Adjoint of a 3x3 matrix (valid for an essential matrix)."""
    assert E.shape == (3, 3)
    det = np.linalg.det
    det_minor00 = det(E[1:, 1:])
    det_minor01 = -det(E[1:, ::2])
    det_minor02 = det(E[1:, :2])
    det_minor10 = -det(E[::2, 1:])
    det_minor11 = det(E[::2, ::2])
    det_minor12 = -det(E[::2, :2])
    det_minor20 = det(E[:2, 1:])
    det_minor21 = -det(E[:2, ::2])
    det_minor22 = det(E[:2, :2])

    # adjugate/adjoint is the *transpose* of the matrix of cofactors.
    adj = np.array(
        [
            [det_minor00, det_minor10, det_minor20],
            [det_minor01, det_minor11, det_minor21],
            [det_minor02, det_minor12, det_minor22],
        ]
    )
    return adj


def so3_orbitope(R):
    """
    [1 + r00 + r11 + r22, r21 - r12,           r02 - r20,           r10 - r01          ]
    [r21 - r12,           1 + r00 - r11 - r22, r10 + r01,           r02 + r20          ]
    [r02 - r20,           r10 + r01,           1 - r00 + r11 - r22, r21 + r12          ]
    [r10 - r01,           r02 + r20,           r21 + r12,           1 - r00 - r11 + r22]
    """
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = R.ravel()
    return np.array(
        [
            [1 + r00 + r11 + r22, r21 - r12, r02 - r20, r10 - r01],
            [r21 - r12, 1 + r00 - r11 - r22, r10 + r01, r02 + r20],
            [r02 - r20, r10 + r01, 1 - r00 + r11 - r22, r21 + r12],
            [r10 - r01, r02 + r20, r21 + r12, 1 - r00 - r11 + r22],
        ]
    )
