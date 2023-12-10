from pathlib import Path

import cv2
import numpy as np
import perfplot

from exp_utils import SyntheticData
from nonmin_pose import C2P, C2PFast, EssentialGSalguero, EssentialZhao
from nonmin_pose.utils import compute_data_matrix_C, decompose_essmat


def zhao_midpoint(f0, f1):
    return ess_zhao(f0, f1, do_disambiguation=True, use_opencv=False)  # type: ignore


def zhao_triangulation(f0, f1):
    return ess_zhao(f0, f1, do_disambiguation=True, use_opencv=True)  # type: ignore


def salguero_midpoint(f0, f1):
    return ess_salguero(f0, f1, do_disambiguation=True, use_opencv=False)  # type: ignore


def salguero_triangulation(f0, f1):
    return ess_salguero(f0, f1, do_disambiguation=True, use_opencv=True)  # type: ignore


def c2p(f0, f1):
    return nonmin_relpose(f0, f1)


def c2p_fast(f0, f1):
    return nonmin_relpose_fast(f0, f1)


def sample_data(n):
    data = dataset.generate_data(max_npoints=n, noise_level=0.0)
    return data


def monkeypatch_call(
    self, f0, f1, do_disambiguation=True, already_normalized=False, use_opencv=False
):
    """function for monkey-patching the __call__ method of the base class to include
    OpenCV's recoverPose() disambiguation, which triangulates the correspondences."""
    sh0, sh1 = f0.shape, f1.shape
    assert sh0 == sh1 and len(sh0) == 2 and sh0[0] == 3 and sh0[1] >= 5
    if not already_normalized:
        f0 = f0 / np.linalg.norm(f0, axis=0)
        f1 = f1 / np.linalg.norm(f1, axis=0)
    C = compute_data_matrix_C(f0, f1)
    self.solveSDP(C, f0, f1)
    sol = self.retrieve_solution()
    if not self.SDP_COMPUTES_POSE:
        U, _, Vt = np.linalg.svd(sol["E01"])
        sol["E01"] = U[:, :2] @ Vt[:2]
        if do_disambiguation and not use_opencv:
            (
                sol["R01"],
                sol["t01"],
                sol["is_pure_rot"],
                sol["ret"],
            ) = decompose_essmat(U, Vt, f0, f1, self.cfg["th_pure_rot_post"])
        elif do_disambiguation and use_opencv:
            f0 = f0[:2] / f0[2:]
            f1 = f1[:2] / f1[2:]
            sol = cv2.recoverPose(sol["E01"], f0.T, f1.T)
    return sol  # type: ignore


EssentialZhao.__call__ = monkeypatch_call  # type: ignore
EssentialGSalguero.__call__ = monkeypatch_call  # type: ignore
C2P.__call__ = monkeypatch_call  # type: ignore
C2PFast.__call__ = monkeypatch_call  # type: ignore

if __name__ == "__main__":
    dataset = SyntheticData(1)
    ess_zhao = EssentialZhao()
    ess_salguero = EssentialGSalguero()
    nonmin_relpose = C2P()
    nonmin_relpose_fast = C2PFast()

    labels = [
        "zhao_midpoint",
        "salguero_midpoint",
        "zhao_triangulation",
        "salguero_triangulation",
        "C2P",
        "C2P-fast",
    ]
    n_to_test = [2**k for k in range(6, 17)]

    out = perfplot.bench(
        setup=lambda n: sample_data(n),
        kernels=[
            lambda data: zhao_midpoint(data["f0"], data["f1"]),
            lambda data: salguero_midpoint(data["f0"], data["f1"]),
            lambda data: zhao_triangulation(data["f0"], data["f1"]),
            lambda data: salguero_triangulation(data["f0"], data["f1"]),
            lambda data: c2p(data["f0"], data["f1"]),
            lambda data: c2p_fast(data["f0"], data["f1"]),
        ],
        labels=labels,
        n_range=n_to_test,
        xlabel="#correspondences",
        # More optional arguments with their default values:
        # logx="auto",  # set to True or False to force scaling
        # logy="auto",
        equality_check=None,  # set to None to disable "correctness" assertion
        show_progress=True,
        # target_time_per_measurement=1.0,
        # max_time=None,  # maximum time per measurement
        # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
        # relative_to=1,  # plot the timings relative to one of the measurements
        # flops=lambda n: 3*n,  # FLOPS plots
    )

    print(f"n samples:\n{n_to_test}\n")
    for timing, label in zip(out.timings_s, labels):
        print(f"{label}:\n{timing}\n")

    out_dir = Path(__file__).parent / "results" / "runtimes"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = str(out_dir / "runtimes")
    filepath_log = str(out_dir / "runtimes_log")

    for ext in [".png", ".pdf"]:
        out.save(
            filepath + ext,
            transparent=True,
            bbox_inches="tight",
            logy=False,
        )
        out.save(
            filepath_log + ext,
            transparent=True,
            bbox_inches="tight",
            logy=True,
        )
