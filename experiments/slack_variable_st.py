from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exp_utils import SyntheticData
from nonmin_pose import C2P
from nonmin_pose.utils import rot_given_Etq

# angle bounds orientation of the second frame.
EULER_ANG_MAGNITUDE = 0.5
NREPEATS = 1000
FOCAL_LENGTH = 800.0

NPOINTS = 100
NOISE_LEVEL = 0.5  # pixels.


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
            E01, t01, q = (
                (E01_, t01_, q_)
                if t01[id_mx, 0] * t01_[id_mx, 0] > 0
                else (-E01_, -t01_, -q_)
            )

        # manifold projections.
        Ue, _, Vte = np.linalg.svd(E01)
        E01 = Ue[:, :2] @ Vte[:2]
        t01 = t01 / np.linalg.norm(t01)
        q = q / np.linalg.norm(q)

        R01 = rot_given_Etq(E01, t01, q)

        # check optimality.
        eps = self.cfg["th_rank_optimality"]
        is_optimal = (svals > eps).sum() <= 3
        return {
            "R01": R01,
            "t01": t01,
            "E01": E01,
            "is_optimal": is_optimal,
            "is_pure_rot": is_pure_rot,
            "sct": sct,
        }


def create_boxplots(translation_lengths, slack_st_vals):
    nt = len(translation_lengths)
    fig, ax = plt.subplots()
    labels = translation_lengths.astype(str)

    ax.boxplot(slack_st_vals.T)

    samples_idx = np.random.choice(NREPEATS, min(50, NREPEATS), replace=False)
    x_jitter = np.random.uniform(-0.4 * 0.5, 0.4 * 0.5, samples_idx.shape)
    for i in range(1, nt + 1):
        xvals = i + x_jitter
        yvals = slack_st_vals[i - 1, samples_idx]
        ax.scatter(xvals, yvals, alpha=0.5, color="0.8", s=5)

    ax.set(
        yscale="log",
        xticklabels=labels,
        # xticks=group_centers,
    )
    fs = 20
    ax.set_xlabel("translation length", fontsize=fs)
    ax.set_ylabel("slack variable $s_t^2$", fontsize=fs)
    ax.tick_params(axis="both", which="major", labelsize=16)
    return fig


def main(dataset, method, translation_lengths, outdir):
    """Accuracy vs translation length experiment."""
    slack_st_vals = np.zeros((len(translation_lengths), NREPEATS))

    for i, translation_length in enumerate(tqdm(translation_lengths)):
        for j in range(NREPEATS):
            data = dataset.generate_data(
                scale_t=translation_length,
                euler_ang_magnitude=EULER_ANG_MAGNITUDE,
                max_npoints=NPOINTS,
                noise_level=NOISE_LEVEL,
            )
            pose_est = method(data["f0"], data["f1"])
            slack_st_vals[i, j] = pose_est["sct"]

    fig = create_boxplots(translation_lengths, slack_st_vals)
    # save the plots
    filepath = str(outdir / f"slack_st_noise{NOISE_LEVEL:.2f}")
    fig.savefig(filepath + ".png", bbox_inches="tight")
    fig.savefig(filepath + ".pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    seed = 0
    dataset = SyntheticData(seed=seed, focal=FOCAL_LENGTH)
    translation_lengths = np.array([0, 1e-3, 1e-2, 1e-1, 0.5, 1, 2, 3])

    relpose_ours = C2PWrap()
    relpose_ours.label = "C2P"  # type: ignore

    # output folder.
    outdir = Path(__file__).parent / "results" / "slack_variable_st"
    outdir.mkdir(parents=True, exist_ok=True)
    # run experiment
    main(dataset, relpose_ours, translation_lengths, outdir)
