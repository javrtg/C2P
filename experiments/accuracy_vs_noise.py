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

NREPEATS = 500
NPOINTS = 1000
NOISE_STEP = 0.5


def main(dataset, methods, noise_levels, outdir):
    """Noise resilience experiment."""
    r_errors = np.zeros((len(noise_levels), len(methods)))
    t_errors = np.zeros((len(noise_levels), len(methods)))

    for i, noise_level in enumerate(tqdm(noise_levels)):
        n = 0
        for j in range(NREPEATS):
            data = dataset.generate_data(
                transl_magnitude=TRANSL_MAGNITUDE,
                euler_ang_magnitude=EULER_ANG_MAGNITUDE,
                max_npoints=NPOINTS,
                noise_level=noise_level,
            )
            # run each method.
            for k, method in enumerate(methods):
                pose_est = method(data["f0"], data["f1"])
                r_err, t_err = compute_error_metrics(data, pose_est)
                # cumulative average.
                r_errors[i, k] = (r_err + n * r_errors[i, k]) / (n + 1)
                t_errors[i, k] = (t_err + n * t_errors[i, k]) / (n + 1)
            n += 1

    # plot results.
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i, method in enumerate(methods):
        ax[0].plot(noise_levels, r_errors[:, i], label=method.label, linewidth=2)
        ax[1].plot(noise_levels, t_errors[:, i], label=method.label, linewidth=2)

    xlim = (noise_levels.min(), noise_levels.max())
    ax[0].set(
        xlabel="noise level (pixels)",
        ylabel="mean rot. error (deg)",
        xlim=xlim,
    )
    ax[1].set(
        xlabel="noise level (pixels)",
        ylabel="mean tran. error (deg)",
        xlim=xlim,
    )

    for axi in ax:
        axi.grid(True, linestyle="--", linewidth=0.5, color="gray")
        axi.legend()

    # save the plots
    filepath = outdir / f"noise_resilience_npoints{NPOINTS}"
    fig.savefig(filepath.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(filepath.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    dataset = SyntheticData(seed=0, focal=FOCAL_LENGTH)
    noise_levels = np.arange(0, 10 + NOISE_STEP, NOISE_STEP)

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
    outdir = Path(__file__).parent / "results" / "accuracy_vs_noise"
    outdir.mkdir(parents=True, exist_ok=True)

    # run experiment
    main(dataset, methods, noise_levels, outdir)
