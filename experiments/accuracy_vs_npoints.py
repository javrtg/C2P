from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exp_utils import SyntheticData, compute_error_metrics
from nonmin_pose import C2P, C2PFast, EssentialGSalguero, EssentialZhao

# magnitude of the translational offset of the second frame.
TRANSL_MAGNITUDE = 2.0
# angle bounds orientation of the second frame.
EULER_ANG_MAGNITUDE = 0.5
FOCAL_LENGTH = 800.0

NREPEATS = 1000
NOISE_LEVEL = 1.0  # pixels.


def savefig(fig, outdir, npoints_levels):
    """Save figure."""
    m, M = npoints_levels.min(), npoints_levels.max()
    filepath = str(
        outdir / f"acc_vs_npoints_noise{NOISE_LEVEL}_tmag{TRANSL_MAGNITUDE}_m{m}_M{M}"
    )
    fig.savefig(filepath + ".png", bbox_inches="tight")
    fig.savefig(filepath + ".pdf", bbox_inches="tight")


def plot_results(ax, r_errors, t_errors, methods, npoints_levels):
    """Plot results."""
    for i, method in enumerate(methods):
        ax[0].plot(npoints_levels, r_errors[:, i], label=method.label, linewidth=2)
        ax[1].plot(npoints_levels, t_errors[:, i], label=method.label, linewidth=2)

    xlim = (npoints_levels.min(), npoints_levels.max())
    # xticks = np.arange(xlim[0], xlim[1] + 1, 5)
    fs = 20
    ax[0].set(
        xlim=xlim,
        # xticks=xticks,
    )
    ax[0].set_xlabel("#correspondences", fontsize=fs)
    ax[0].set_ylabel("mean rot. error (deg)", fontsize=fs)
    ax[1].set(
        xlim=xlim,
        # xticks=xticks,
    )
    ax[1].set_xlabel("#correspondences", fontsize=fs)
    ax[1].set_ylabel("mean tran. error (deg)", fontsize=fs)

    for axi in ax:
        axi.grid(True, linestyle="--", linewidth=0.5, color="gray")
        axi.legend(prop=dict(size=18))
        axi.tick_params(axis="both", which="major", labelsize=16)


def main(dataset, methods, outdir):
    """Accuracy vs npoints experiment."""
    regime1 = np.arange(12, 30 + 1, 1)
    regime2 = np.arange(100, 10_000 + 400, 400)

    for ri, regime in enumerate((regime1, regime2)):
        npoints_levels = regime
        r_errors = np.zeros((len(npoints_levels), len(methods)))
        t_errors = np.zeros((len(npoints_levels), len(methods)))

        for i, npoints in enumerate(tqdm(npoints_levels, desc=f"Regime {ri}")):
            n = 0
            for j in range(NREPEATS):
                data = dataset.generate_data(
                    transl_magnitude=TRANSL_MAGNITUDE,
                    euler_ang_magnitude=EULER_ANG_MAGNITUDE,
                    max_npoints=npoints,
                    noise_level=NOISE_LEVEL,
                )

                # run each method.
                for k, method in enumerate(methods):
                    pose_est = method(data["f0"], data["f1"])
                    r_err, t_err = compute_error_metrics(data, pose_est)
                    # cumulative average.
                    r_errors[i, k] = (r_err + n * r_errors[i, k]) / (n + 1)
                    t_errors[i, k] = (t_err + n * t_errors[i, k]) / (n + 1)
                n += 1

        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        plot_results(ax, r_errors, t_errors, methods, npoints_levels)
        if ri == 1:
            ax[1].set_xscale("log", base=2)
            ax[0].set_xscale("log", base=2)
        savefig(fig, outdir, npoints_levels)


if __name__ == "__main__":
    seed = 0
    dataset = SyntheticData(seed=seed, focal=FOCAL_LENGTH)

    # nonminimal models to evaluate.
    essmat_zhao = EssentialZhao()
    essmat_gsalguero = EssentialGSalguero()
    relpose_ours = C2P()
    relpose_ours_f = C2PFast()
    # labels for plotting.
    essmat_zhao.label = "Zhao"  # type: ignore
    essmat_gsalguero.label = "García-Salguero"  # type: ignore
    relpose_ours.label = "C2P"  # type: ignore
    relpose_ours_f.label = "C2P-fast"  # type: ignore

    # gather solvers.
    methods = [
        essmat_zhao,
        essmat_gsalguero,
        relpose_ours_f,
        relpose_ours,
    ]

    # output folder.
    outdir = Path(__file__).parent / "results" / "accuracy_vs_npoints"
    outdir.mkdir(parents=True, exist_ok=True)

    # run experiment
    main(dataset, methods, outdir)
