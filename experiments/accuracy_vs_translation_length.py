from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exp_utils import SyntheticData, compute_error_metrics
from nonmin_pose import C2P, EssentialGSalguero, EssentialZhao

# angle bounds orientation of the second frame.
EULER_ANG_MAGNITUDE = 0.5
NREPEATS = 1000
FOCAL_LENGTH = 800.0

NPOINTS = 100
NOISE_LEVEL = 0.5  # 0.5 pixels.


def create_boxplots(translation_lengths, r_errors, t_errors):
    nm = r_errors.shape[0]
    nt = len(translation_lengths)

    # plot results.
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # style.
    colors = plt.get_cmap("turbo")(np.linspace(0.1, 0.9, nm))  # type: ignore
    labels = translation_lengths.astype(str)

    # position of the boxes for the first method.
    box_w = 0.5
    box_sep = 0.2 * box_w
    group_sep = box_w
    positions = np.arange(
        0,
        nt * (nm * box_w + (nm - 1) * box_sep + group_sep),
        nm * box_w + (nm - 1) * box_sep + group_sep,
    )
    group_centers = np.zeros(nt)  # for cumulative average

    # add point samples to the boxplots to show the distribution.
    samples_idx = np.random.choice(NREPEATS, min(50, NREPEATS), replace=False)
    x_jitter = np.random.uniform(-0.4 * box_w, 0.4 * box_w, samples_idx.shape)
    idx = np.ones_like(samples_idx)

    for k, c in enumerate(colors):
        positions_ = positions + k * (box_w + box_sep)
        group_centers += positions_

        ax[0].boxplot(
            r_errors[k].T,
            positions=positions_,
            widths=box_w,
            # labels=labels,
            patch_artist=True,
            boxprops=dict(color=c, facecolor="none"),
            capprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            whiskerprops=dict(color=c),
            medianprops=dict(color="black"),
        )

        ax[1].boxplot(
            t_errors[k].T,
            positions=positions_,
            widths=box_w,
            # labels=labels,
            patch_artist=True,
            boxprops=dict(color=c, facecolor="none"),
            capprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            whiskerprops=dict(color=c),
            medianprops=dict(color="black"),
        )

        for i, pos in enumerate(positions_):
            idx_i = i * idx
            samples_r_err = r_errors[k, idx_i, samples_idx]
            samples_t_err = t_errors[k, idx_i, samples_idx]

            xvals = pos + x_jitter

            ax[0].scatter(xvals, samples_r_err, alpha=0.5, color="0.8", s=5)
            ax[1].scatter(xvals, samples_t_err, alpha=0.5, color="0.8", s=5)

    group_centers /= nm
    ax[0].set(
        xlabel="translation length",
        ylabel="rotation error (deg)",
        xticks=group_centers,
        xticklabels=labels,
    )
    ax[1].set(
        xlabel="translation length",
        ylabel="translation error (deg)",
        yscale="log",
        xticks=group_centers,
        xticklabels=labels,
    )
    ax[1].legend(
        handles=[
            ax[1].plot([], [], c=c, label=m.label)[0] for m, c in zip(methods, colors)
        ]
    )
    return fig


def main(dataset, methods, translation_lengths, outdir):
    """Accuracy vs translation length experiment."""
    r_errors = np.zeros((len(methods), len(translation_lengths), NREPEATS))
    t_errors = np.zeros((len(methods), len(translation_lengths), NREPEATS))

    for i, translation_length in enumerate(tqdm(translation_lengths)):
        for j in range(NREPEATS):
            data = dataset.generate_data(
                scale_t=translation_length,
                euler_ang_magnitude=EULER_ANG_MAGNITUDE,
                max_npoints=NPOINTS,
                noise_level=NOISE_LEVEL,
            )

            # run each method.
            for k, method in enumerate(methods):
                pose_est = method(data["f0"], data["f1"])
                # error metrics.
                r_errors[k, i, j], t_errors[k, i, j] = compute_error_metrics(
                    data, pose_est
                )
    # plot and save results.
    fig = create_boxplots(translation_lengths, r_errors, t_errors)
    # save the plots
    filepath = str(outdir / f"acc_vs_translen_noise{NOISE_LEVEL}")
    fig.savefig(filepath + ".png", bbox_inches="tight")
    fig.savefig(filepath + ".pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    seed = 0
    dataset = SyntheticData(seed=seed, focal=FOCAL_LENGTH)
    translation_lengths = np.array([0, 1e-3, 1e-2, 1e-1, 0.5, 1, 2, 3])

    # models to evaluate.
    essmat_zhao = EssentialZhao()
    essmat_gsalguero = EssentialGSalguero()
    relpose_ours = C2P()

    # prepare non-minimal methods.
    essmat_zhao.label = "Zhao"  # type: ignore
    essmat_gsalguero.label = "García-Salguero"  # type: ignore
    relpose_ours.label = "C2P"  # type: ignore

    methods = [
        essmat_zhao,
        essmat_gsalguero,
        relpose_ours,
    ]

    # output folder.
    outdir = Path(__file__).parent / "results" / "accuracy_vs_translation_length"
    outdir.mkdir(parents=True, exist_ok=True)

    # run experiment
    main(dataset, methods, translation_lengths, outdir)
