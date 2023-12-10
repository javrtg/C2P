from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exp_utils import SyntheticData, compute_error_metrics
from nonmin_pose import C2P
from nonmin_pose.utils import rot_given_Etq

# angle bounds orientation of the second frame.
EULER_ANG_MAGNITUDE = 0.5
NREPEATS = 1000
FOCAL_LENGTH = 800.0

NPOINTS = 100
NOISE_LEVEL = 10.0  # pixels.


class C2PWrap(C2P):
    def retrieve_solution(self):
        X = self.npt_problem.getResultYMat(self.params["h"].block)
        _, svals, Vt = np.linalg.svd(X)
        idx = 0
        E01 = Vt[idx, :9].reshape(3, 3)
        t01 = Vt[idx, 9:12].reshape(3, 1)
        q = Vt[idx, 12:15].reshape(3, 1)
        h = Vt[idx, 15]
        E01, t01, q = (E01, t01, q) if h > 0 else (-E01, -t01, -q)

        sct = self.npt_problem.getResultYMat(2)[0, 0]
        is_pure_rot = sct < self.cfg["th_pure_rot_sdp"]

        if sct < self.cfg["th_pure_rot_noisefree_sdp"]:
            # improve numerical conditioning.
            _, _, Vt = np.linalg.svd(X[:15, :15])
            E01_ = Vt[idx, :9].reshape(3, 3)
            t01_ = Vt[idx, 9:12].reshape(3, 1)
            q_ = Vt[idx, 12:15].reshape(3, 1)
            # correct sign.
            id_mx = np.abs(t01).argmax()
            E01_c, t01_c, q_c = (
                (E01_, t01_, q_)
                if t01[id_mx, 0] * t01_[id_mx, 0] > 0
                else (-E01_, -t01_, -q_)
            )
        else:
            E01_c, t01_c, q_c = E01, t01, q

        # manifold projections (without rot correction).
        Ue, _, Vte = np.linalg.svd(E01)
        E01 = Ue[:, :2] @ Vte[:2]
        t01 = t01 / np.linalg.norm(t01)
        q = q / np.linalg.norm(q)
        R01 = rot_given_Etq(E01, t01, q)

        # manifold projections (with maybe a rot correction).
        Ue, _, Vte = np.linalg.svd(E01_c)
        E01_c = Ue[:, :2] @ Vte[:2]
        t01_c = t01_c / np.linalg.norm(t01_c)
        q_c = q_c / np.linalg.norm(q_c)
        R01_c = rot_given_Etq(E01_c, t01_c, q_c)

        # check optimality.
        eps = self.cfg["th_rank_optimality"]
        is_optimal = (svals > eps).sum() <= 3
        return {
            "R01": R01,
            "t01": t01,
            "E01": E01,
            "R01_c": R01_c,
            "t01_c": t01_c,
            "E01_c": E01_c,
            "is_optimal": is_optimal,
            "is_pure_rot": is_pure_rot,
            "h": h,
        }


def create_plot(translation_lengths, h_vals, r_errors, r_errors_c):
    nt = len(translation_lengths)
    x = np.arange(nt)
    labels = [
        f"$10^{{{np.log10(i):.0f}}}$" if i > 0 else "0" for i in translation_lengths
    ]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # bar chart of homogenization values.
    color_bars = [i / 255 for i in (167, 199, 231)]
    color_bars = [0.0, 0.0, 1.0]
    h_mean = h_vals.mean(1)
    h_std = h_vals.std(axis=1)
    ax1.bar(
        x,
        h_mean,
        # yerr=h_std,
        color=color_bars,
        alpha=0.5,
        # label="homogenization var. $h$",
    )
    ax1.set_ylabel("homogenization var. $h$", color=color_bars, fontsize=20)
    ax1.tick_params(axis="y", labelcolor=color_bars)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_ylim(0, 1.0)

    # line plot of rotation errors.
    color_line = [i / 255 for i in (255, 150, 79)]
    color_line = [1.0, 0.0, 0.0]
    r_mean_c = r_errors_c.mean(axis=1)
    r_std_c = r_errors_c.std(axis=1)
    ax2.plot(
        x,
        r_mean_c,
        color=color_line,
        linestyle="--",
        label="num. improved rotation",
        lw=3,
    )
    # ax2.fill_between(
    #     x,
    #     r_mean_c - 3 * r_std_c,
    #     r_mean_c + 3 * r_std_c,
    #     color=color_line,
    #     alpha=0.5,
    # )

    r_mean = r_errors.mean(axis=1)
    r_std = r_errors.std(axis=1)
    ax2.plot(
        x,
        r_mean,
        color=color_line,
        label="raw rotation",
        lw=3,
    )
    # ax2.fill_between(
    #     x,
    #     r_mean - 3 * r_std,
    #     r_mean + 3 * r_std,
    #     color=color_line,
    #     alpha=0.5,
    # )

    if r_mean[0] / r_mean[-1] > 1e2:
        ax2.set_yscale("log")
    else:
        ax2.set_ylim(r_mean.min() * 0.9, r_mean.max() * 1.1)

    ax2.set_ylabel("rotation error (deg)", color=color_line, fontsize=20)
    ax2.tick_params(axis="y", labelcolor=color_line)
    ax2.tick_params(axis="y", which="major", labelsize=16)

    ax2.legend(fontsize=16, frameon=False)
    ax1.set_xlabel("relative scale of $\\mathbf{t}$ vs scene", fontsize=20)
    return fig


def main(dataset, method, translation_lengths, outdir):
    """Accuracy vs translation length experiment."""
    h_vals, r_errors, r_errors_c = np.zeros((3, len(translation_lengths), NREPEATS))

    for i, translation_length in enumerate(tqdm(translation_lengths)):
        for j in range(NREPEATS):
            data = dataset.generate_data(
                scale_t=translation_length,
                euler_ang_magnitude=EULER_ANG_MAGNITUDE,
                max_npoints=NPOINTS,
                noise_level=NOISE_LEVEL,
            )
            pose_est = method(data["f0"], data["f1"])
            h_vals[i, j] = pose_est["h"]
            r_errors[i, j], _ = compute_error_metrics(data, pose_est)

            # errors in case of a rotation correction.
            pose_est["R01"] = pose_est["R01_c"]
            pose_est["t01"] = pose_est["t01_c"]
            pose_est["E01"] = pose_est["E01_c"]
            r_errors_c[i, j], _ = compute_error_metrics(data, pose_est)
    h_vals = np.abs(h_vals)

    # plot and save results.
    fig = create_plot(translation_lengths, h_vals, r_errors, r_errors_c)
    # fig = create_boxplots(translation_lengths, slack_st_vals)

    # save the plots
    filepath = str(outdir / f"homog_rotacc_vs_tlen_noise{NOISE_LEVEL:.2f}")
    fig.savefig(filepath + ".png", bbox_inches="tight")
    fig.savefig(filepath + ".pdf", bbox_inches="tight")

    plt.close(fig)


if __name__ == "__main__":
    seed = 0
    dataset = SyntheticData(seed=seed, focal=FOCAL_LENGTH)
    translation_lengths = np.array([0, 1e-5, 1e-4, 1e-3, 1e-2, 1])

    # models to evaluate.
    relpose_ours = C2PWrap()
    relpose_ours.label = "C2P"  # type: ignore

    # output folder.
    outdir = Path(__file__).parent / "results" / "homog_vs_tlen"
    outdir.mkdir(parents=True, exist_ok=True)

    # run experiment
    main(dataset, relpose_ours, translation_lengths, outdir)
