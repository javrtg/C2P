import os
import shutil
import urllib.request
from argparse import ArgumentParser
from math import atan, cos
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyopengv import pyopengv as gv
from tqdm import tqdm

from exp_utils import compute_error_metrics, plot_boxplots, plot_matches, skew
from nonmin_pose import C2P, C2PFast, EssentialGSalguero, EssentialZhao

SEQUENCE = "castle-P19"

SEQ_MAPS = {
    "fountain-P11": "fountain_dense",
    "Herz-Jesus-P8": "herzjesu_dense",
    "entry-P10": "castle_entry_dense",
    "castle-P19": "castle_dense",
    "Herz-Jesus-P25": "herzjesu_dense_large",
    "castle-P30": "castle_dense_large",
}


def do_request(url, file_name, do_uncompress=False, do_clean=False):
    """Download the file from `url` and save it locally under `file_name`.
    ref: https://stackoverflow.com/a/7244263/14559854
    """
    with urllib.request.urlopen(url) as response, open(file_name, "wb") as out_file:
        shutil.copyfileobj(response, out_file)
    if do_uncompress:
        shutil.unpack_archive(file_name, Path(file_name).parent)
        if do_clean:
            os.remove(file_name)


def download_data(data_dir: Path, sequence: str, remove_tar: bool = True):
    data_dir.mkdir(parents=True, exist_ok=True)
    seq_name = SEQ_MAPS[sequence]
    base_url = (
        "https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/data/"
    )
    seq_url = base_url + f"{seq_name}/urd/"
    print("Downloading and extracting data...")
    # .png images.
    file = f"{seq_name}_images.tar.gz"
    do_request(seq_url + file, str(data_dir / file), True, do_clean=remove_tar)
    # camera files (containing K, R, t).
    file = f"{seq_name}_cameras.tar.gz"
    do_request(seq_url + file, str(data_dir / file), True, do_clean=remove_tar)
    # Projection matrices: K * (R | t).
    file = f"{seq_name}_p.tar.gz"
    do_request(seq_url + file, str(data_dir / file), True, do_clean=remove_tar)
    print("Files downloaded and extracted!")


def read_data(data_dir: Path, check_with_pmats: bool = False):
    """Return images, calibration matrices, rotation matrices and translation vectors.

    Table with Strecha's file .png.camera format:
    ----------------------------------------------------------------------------------
    | Lines | Content                                                                |
    |-------|------------------------------------------------------------------------|
    | 1-3   | K: 3X3 calib matrix                                                    |
    | 4     | row with 3 zeros                                                       |
    | 5-7   | Rwc: 3X3 rotation rotating points from cam. to world reference systems |
    |       |      i.e. pw = Rwc pc                                                  |
    | 8     | twc: translation vector, again from cam. to world reference system     |
    | 9     | width and height (in pixels) of the image                              |
    ----------------------------------------------------------------------------------
    As such, the projection matrix, P, projecting points from world to homogeneous
    image coordinates is given by:
        P = K * [Rwc^T | -Rwc^T t]

    Args:
        data_dir (Path): Path to the directory containing the data.

    Returns:
        ims: ndarray (n, h, w, 3) with the n images.
        Ks: ndarray (n, 3, 3) with the n calibration matrices.
        Rwcs: ndarray (n, 3, 3) with the n rotation matrices (cam. to world coords.).
        twcs: ndarray (n, 3, 1) with the n translation vectors (cam. to world coords.).
    """
    # Strecha's naming convention for the sequences is "{seq}-P{number_of_images}".
    n = int(data_dir.name.rpartition("P")[2])
    data_dir = data_dir / SEQ_MAPS[data_dir.name] / "urd"
    assert data_dir.is_dir(), f"Data directory {data_dir} does not exist."

    # images.
    ims = np.stack(
        [
            np.array(Image.open(p), dtype=np.uint8)
            for p in sorted(data_dir.glob("*.png"))
        ]
    )
    assert n == len(ims), f"Expected {n} images, got {ims.shape[0]}."

    # calibration matrices and extrinsic params.
    Rwcs = np.empty((n, 3, 3))
    twcs = np.empty((n, 3, 1))
    Ks = np.zeros((n, 3, 3))
    Ks[:, 2, 2] = 1.0

    for i, p in enumerate(sorted(data_dir.glob("*.png.camera"))):
        with open(p, "r") as f:
            lines = [line.split() for line in f.readlines()]
        # calibration matrix.
        Ks[i, 0, 0] = lines[0][0]  # fx
        Ks[i, 0, 2] = lines[0][2]  # cx
        Ks[i, 1, 1] = lines[1][1]  # fy
        Ks[i, 1, 2] = lines[1][2]  # cy
        # rotation matrices.
        Rwcs[i] = lines[4:7]
        # translation vectors.
        twcs[i, :, 0] = lines[7]

    if check_with_pmats:
        # read projection matrices.
        Ps = np.empty((n, 3, 4))
        for i, p in enumerate(sorted(data_dir.glob("*.png.P"))):
            with open(p, "r") as f:
                Ps[i] = [line.split() for line in f.readlines()]
        assert i == n - 1, f"Expected {n} projection matrices, got {i + 1}."  # type: ignore
        # ensure that they are consistent with the read Ks, Rwcs and twcs.
        Ps_to_test = Ks @ np.concatenate(
            (Rwcs.transpose(0, 2, 1), -Rwcs.transpose(0, 2, 1) @ twcs), axis=2
        )
        assert (
            np.max(np.abs(Ps - Ps_to_test))
            < 0.2  # we use 0.2 since elements in .png.P files have 1 decimal place.
        ), "Inconsistent projection matrices."

    return ims, Ks, Rwcs, twcs


def extract_local_features(ims: np.ndarray):
    """Extract local features (2D points + desc.) from the images using DoG + SIFT."""

    def rootsift(descs):
        descs = descs / (descs.sum(1, keepdims=True))  # L1 normalization
        descs = np.sqrt(descs)  # now is L2 normalized: ||sqrt(x_l1)|| = sum(x_l1) = 1
        # but for greater precision, we ensure it is L2 normalized.
        descs /= np.linalg.norm(descs, axis=1, keepdims=True)
        return descs

    assert ims.ndim == 4 and ims.shape[3] == 3, "Expected color images."
    gray_ims = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in ims]
    local_feat_extractor = cv2.SIFT_create()  # pyright: ignore[reportGeneralTypeIssues]

    print("Extracting local features...")
    kps_per_im, descs_per_im = [], []
    for im in tqdm(gray_ims, desc="Extracting local features", leave=False):
        kps, descs = local_feat_extractor.detectAndCompute(im, None)
        descs_per_im.append(rootsift(descs))
        # get 2d coordinates.
        kps_per_im.append(np.array([kp.pt for kp in kps]))
    print("Done!")
    return kps_per_im, descs_per_im


def bearings_from_2d_points(kps_per_im, Ks):
    """Return the unit bearing vectors corresponding to the 2D points in each image."""
    assert len(kps_per_im) == len(Ks), "Expected same number of images and Ks."
    bearings_per_im = []
    for kps, K in zip(kps_per_im, Ks):
        # 2d points in normalized plane.
        kps = cv2.undistortPoints(kps, K, None)[:, 0].T  # type:ignore (2, n)
        # corresponding bearings.
        kps = np.concatenate((kps, np.ones_like(kps[:1])), axis=0)
        bearings = kps / np.linalg.norm(kps, axis=0, keepdims=True)
        bearings_per_im.append(bearings)
    return bearings_per_im


class StrechaDataset:
    def __init__(
        self,
        data_dir: Path,
        sequence: str = "castle-P19",
        step: int = 1,
        download: bool = False,
        remove_tar: bool = True,
    ):
        self.step = step
        if download:
            download_data(data_dir, sequence, remove_tar)

        self.ims, self.Ks, Rwcs, twcs = read_data(data_dir, check_with_pmats=True)
        self._len = len(self.ims)

        self.kps_per_im, self.descs_per_im = extract_local_features(self.ims)
        self.bearings_per_im = bearings_from_2d_points(self.kps_per_im, self.Ks)

        # get relative poses.
        Rw0, Rw1 = Rwcs[:-step], Rwcs[step:]
        tw0, tw1 = twcs[:-step], twcs[step:]
        self.R01s = Rw0.transpose(0, 2, 1) @ Rw1
        self.t01s = Rw0.transpose(0, 2, 1) @ (tw1 - tw0)

        self.last_idx = self._len - self.step

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        """Return data for the idx-th image pair: relative pose, calib mats and matches"""
        assert (
            idx < self._len - self.step
        ), f"Index {idx} out of bounds with step {self.step}."

        d0, d1 = self.descs_per_im[idx], self.descs_per_im[idx + self.step]
        matches = pairwise_matching(d0, d1)
        idx0, idx1 = matches[:, 0], matches[:, 1]

        data = {
            "f0": self.bearings_per_im[idx][:, idx0],  # (3,n)
            "f1": self.bearings_per_im[idx + self.step][:, idx1],  # (3,n)
            "kps0": self.kps_per_im[idx][idx0],  # (n,2)
            "kps1": self.kps_per_im[idx + self.step][idx1],  # (n,2)
            "R01": self.R01s[idx],
            "t01": self.t01s[idx],
            "K0": self.Ks[idx],
            "K1": self.Ks[idx + self.step],
        }
        return data

    def plot_matches(self, idx, kps0_m, kps1_m, ransac_mask, out_dir: Path):
        fig, ax = plt.subplots(1, 2)

        ax[0].imshow(self.ims[idx])
        ax[1].imshow(self.ims[idx + self.step])

        # all keypoints (including those that are not matched).
        kps0, kps1 = self.kps_per_im[idx], self.kps_per_im[idx + self.step]
        ax[0].scatter(kps0[:, 0], kps0[:, 1], c="b", s=1, alpha=0.5)
        ax[1].scatter(kps1[:, 0], kps1[:, 1], c="b", s=1, alpha=0.5)

        # plot all (pre and post ransac) matches.
        kw_pre_p = {"s": 1.0, "c": [(1.0, 0.0, 0.0)]}
        kw_pre_l = {"lw": 0.1, "color": (1.0, 0.0, 0.0)}
        plot_matches(fig, ax[0], ax[1], kps0_m.T, kps1_m.T, kw_pre_p, kw_pre_l)
        # plot ransac inliers.
        kps0_m, kps1_m = kps0_m[ransac_mask], kps1_m[ransac_mask]
        kw_post_p = {"s": 1.0, "c": [(0.0, 1.0, 0.0)]}
        kw_post_l = {"lw": 0.1, "color": (0.0, 1.0, 0.0), "alpha": 1.0}
        plot_matches(fig, ax[0], ax[1], kps0_m.T, kps1_m.T, kw_post_p, kw_post_l)

        # save.
        out_dir = out_dir / "matches"
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = str(out_dir / f"{idx}_{idx+self.step}.png")
        ax[0].set_axis_off()
        ax[1].set_axis_off()
        fig.tight_layout(pad=0.5)
        fig.savefig(fname, bbox_inches="tight", pad_inches=0.0)

        plt.close(fig)


def pairwise_matching(d0, d1, ratio_th: float = 0.8):
    """Pairwise matching of SIFT descriptors with Lowe's ratio test and MNN check."""
    ind0 = np.arange(len(d0))  # (n0,)
    mask0 = np.ones(len(d0), dtype=bool)
    # similarity matrix.
    sim = d0 @ d1.T
    # nearest neighbours for each descriptor in image 0 and 1.
    nn0 = np.argpartition(sim, -2 if ratio_th else -1, axis=1)  # (n0, n1)
    nn1 = np.argpartition(sim.T, -2 if ratio_th else -1, axis=1)  # (n1, n0)
    if ratio_th:
        # Lowe's ratio test.
        dist = 2.0 * (1.0 - sim)
        mask0 &= dist[ind0, nn0[:, -1]] <= ratio_th**2 * dist[ind0, nn0[:, -2]]
    # MNN check.
    mask0 &= ind0 == nn1[nn0[ind0, -1], -1]
    # final matched idx.
    matches = np.stack((ind0[mask0], nn0[mask0, -1]), axis=1)  # (n, 2)
    return matches


def filter_outliers_opengv(data, method="NISTER", th=5.0):
    """Outlier filtering with OpenGV's 3D based ransac.

    The most important parameter is the angular threshold, which is explained in:
    https://laurentkneip.github.io/opengv/page_how_to_use.html#sec_ransac
    """
    threshold = 1.0 - cos(atan(th / 2_700))
    max_iters = 1_000
    prob = 0.99

    f0, f1 = data["f0"], data["f1"]
    Rt01 = gv.relative_pose_ransac(
        b1=f0.T,
        b2=f1.T,
        algo_name=method,
        threshold=threshold,
        iterations=max_iters,
        probability=prob,
    )
    R01 = Rt01[:3, :3].copy()
    t01 = Rt01[:3, 3:] / np.linalg.norm(Rt01[:3, 3])

    # (estimated) bearings with the estimated relative pose.
    p3d_0 = gv.triangulation_triangulate2(f0.T, f1.T, t01, R01).T  # (3, n)
    f0_reproj = p3d_0 / np.linalg.norm(p3d_0, axis=0, keepdims=True)
    f1_reproj = R01.T @ p3d_0 - R01.T @ t01
    f1_reproj /= np.linalg.norm(f1_reproj, axis=0, keepdims=True)

    # angular threshold to select inliers (transformed to be \in [0, 2]).
    ang_error0 = 1.0 - np.einsum("dn,dn->n", f0, f0_reproj)
    ang_error1 = 1.0 - np.einsum("dn,dn->n", f1, f1_reproj)
    inliers_mask = (ang_error0 + ang_error1) <= threshold

    f0_inliers = f0[:, inliers_mask]
    f1_inliers = f1[:, inliers_mask]

    pose_est = {"R01": R01, "t01": t01, "E01": skew(t01) @ R01}
    return pose_est, f0_inliers, f1_inliers, inliers_mask


def main(args, methods):
    assert len(methods) > 0, "No methods to evaluate."
    # the dataset {will be downloaded / is expected} at this location.
    data_dir = Path(__file__).parent / "data" / "strecha" / args.sequence
    results_dir = Path(__file__).parent / "results" / "strecha" / args.sequence
    dataset = StrechaDataset(data_dir, args.sequence, step=1, download=args.download)
    ransac_fun = filter_outliers_opengv

    r_errors = np.zeros((len(methods), dataset.last_idx))
    t_errors = np.zeros((len(methods), dataset.last_idx))

    for i in tqdm(range(dataset.last_idx), desc="Evaluating methods"):
        # tentative matches + ground-truth.
        data = dataset[i]
        # filter potential outliers with ransac.
        _, f0_inliers, f1_inliers, inliers_mask = ransac_fun(data)
        dataset.plot_matches(i, data["kps0"], data["kps1"], inliers_mask, results_dir)
        data["f0"] = f0_inliers
        data["f1"] = f1_inliers

        for k, method in enumerate(methods):
            pose_est = method(f0_inliers, f1_inliers)
            r_errors[k, i], t_errors[k, i] = compute_error_metrics(data, pose_est)

    print([method.label for method in methods])
    print(r_errors.mean(1), np.median(r_errors, 1))
    print(t_errors.mean(1), np.median(t_errors, 1))

    # plot boxplots.
    fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(15, 6))
    x_labels = [method.label for method in methods]
    plot_boxplots(ax[0], r_errors.T, x_labels, sample_points=False, showfliers=False)
    plot_boxplots(ax[1], t_errors.T, x_labels, sample_points=False, showfliers=False)

    fs = 20
    ax[0].set_ylabel("rotation error (deg)", fontsize=fs)
    ax[1].set_ylabel("translation error (deg)", fontsize=fs)
    for axi in ax:
        # axi.set_xlabel("method", fontsize=fs)
        axi.tick_params(axis="both", which="major", labelsize=16)

    fname = str(results_dir / "boxplots")
    fig.savefig(fname + ".png", bbox_inches="tight")
    fig.savefig(fname + ".pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--sequence", type=str, default=SEQUENCE)
    args = parser.parse_args()

    # nonminimal models to evaluate.
    essmat_zhao = EssentialZhao()
    essmat_gsalguero = EssentialGSalguero()
    relpose_ours = C2P()
    relpose_ours_f = C2PFast()
    # plot labels.
    essmat_zhao.label = "Zhao"  # type: ignore
    essmat_gsalguero.label = "G-Salg."  # type: ignore
    relpose_ours.label = "C2P"  # type: ignore
    relpose_ours_f.label = "C2P-fast."  # type: ignore

    # gather solvers.
    methods = [
        essmat_zhao,
        essmat_gsalguero,
        relpose_ours,
        relpose_ours_f,
    ]

    main(args, methods)
