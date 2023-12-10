from math import pi
from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
from scipy.spatial.transform import Rotation as R

RAD2DEG = 180 / pi


def skew(t):
    return np.array(
        [
            [0, -t[2, 0], t[1, 0]],
            [t[2, 0], 0, -t[0, 0]],
            [-t[1, 0], t[0, 0], 0],
        ]
    )


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
        noise_level=2.5,
        scale_t=None,
    ):
        """Generate synthetic data."""
        # absolute camera poses (w.r.t. world "w" reference).
        Rw0, tw0 = self.cam0_absolute_pose
        Rw1, tw1 = self.set_cam1_absolute_pose(
            transl_magnitude, euler_ang_magnitude, scale_t
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

    def set_cam1_absolute_pose(self, transl_magnitude, euler_ang_magnitude, scale_t):
        """camera 1 pose (w.r.t. world "w" reference)."""
        euler_angles = self.rng.uniform(-euler_ang_magnitude, euler_ang_magnitude, (3,))
        Rw1 = R.from_euler("zyx", euler_angles).as_matrix()
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


def compute_error_metrics(
    data_gt: Dict[str, np.ndarray], data_est: Dict[str, np.ndarray]
) -> Tuple[Union[np.float64, float], Union[np.float64, float]]:
    """Compute the rotation and translation errors."""
    if data_est.get("fail", False):
        return 180.0, 180.0
    R01_true, t01_true = data_gt["R01"], data_gt["t01"]
    R01_est, t01_est = data_est["R01"], data_est["t01"]
    return _rotation_error(R01_true, R01_est), _translation_error(t01_true, t01_est)


def _rotation_error(R01_true, R01_est) -> np.float64:
    """Rotation error in degrees."""
    # NOTE:
    # R01_true.ravel().dot(R01_est.ravel()) is the same as trace(R01_true.T @ R01_est)
    return RAD2DEG * np.arccos(
        (0.5 * (R01_true.ravel().dot(R01_est.ravel()) - 1)).clip(-1, 1)
    )


def _translation_error(t01_true, t01_est) -> Union[np.float64, float]:
    """Translation error in degrees."""
    t_true_norm = np.linalg.norm(t01_true)
    if t_true_norm == 0:
        return 90.0
    den = t_true_norm * np.linalg.norm(t01_est)
    return RAD2DEG * np.arccos((t01_true[:, 0].dot(t01_est[:, 0]) / den).clip(-1, 1))


def plot_boxplots(
    ax, data: np.ndarray, x_labels: Sequence[str], sample_points=True, **kwargs
):
    nd, nl = data.shape
    assert nl == len(x_labels), "number of data's columns and labels must be the same"

    ax.boxplot(data, **kwargs)

    if sample_points:
        samples_idx = np.random.choice(nd, min(50, nd), replace=False)
        x_jitter = np.random.uniform(-0.4 * 0.5, 0.4 * 0.5, samples_idx.shape)
        for i in range(1, nl + 1):
            xvals = i + x_jitter
            yvals = data[samples_idx, i - 1]
            ax.scatter(xvals, yvals, alpha=0.5, color="0.8", s=5)

    ax.set(xticklabels=x_labels)


def plot_matches(
    fig: plt.Figure,  # type:ignore
    ax0: plt.Axes,
    ax1: plt.Axes,
    kps0: np.ndarray,
    kps1: np.ndarray,
    kwargs_points: Optional[Dict] = None,
    kwargs_lines: Optional[Dict] = None,
):
    """Plot pairwise matches between two images."""
    kw_points = {"s": 10.0, "c": [(0.0, 1.0, 0.0)]}
    kw_points.update(kwargs_points or {})
    kw_lines = {"lw": 2.0, "color": (0.0, 1.0, 0.0), "alpha": 0.5}
    kw_lines.update(kwargs_lines or {})

    # 2d points.
    ax0.scatter(kps0[0], kps0[1], **kw_points)
    ax1.scatter(kps1[0], kps1[1], **kw_points)
    # matches.
    for i in range(kps0.shape[1]):
        fig.add_artist(
            ConnectionPatch(
                xyA=kps0[:, i],  # type:ignore
                xyB=kps1[:, i],  # type:ignore
                axesA=ax0,
                axesB=ax1,
                coordsA=ax0.transData,
                coordsB=ax1.transData,
                **kw_lines,
            )
        )
    return fig
