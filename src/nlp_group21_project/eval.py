import torch
from dataset import LABEL_NAMES
from utils import to_device
import numpy as np
import matplotlib.pyplot as plt
import baycomp
import scipy.stats as st
from itertools import combinations


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=3):
    model.eval()
    total_loss = torch.zeros((), device=device)
    n = 0
    conf = torch.zeros(num_classes, num_classes,
                       dtype=torch.long, device=device)
    for batch in loader:
        batch = to_device(batch, device)
        logits = model(batch)
        labels = batch["labels"]
        total_loss += criterion(logits, labels).detach() * labels.size(0)
        n += labels.size(0)
        idx = labels * num_classes + logits.argmax(-1)
        conf += torch.bincount(idx, minlength=num_classes**2).view(
            num_classes, num_classes
        )

    total_loss = total_loss.item()
    conf = conf.cpu().float()
    tp = conf.diag()
    precision = tp / conf.sum(0).clamp(min=1)
    recall = tp / conf.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {
        "loss": total_loss / n,
        "accuracy": (tp.sum() / conf.sum()).item(),
        "macro_f1": f1.mean().item(),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "f1": f1[i].item(),
            }
            for i in range(num_classes)
        },
        "confusion_matrix": conf.long().tolist(),
    }


class BenavoliCorrelatedT:
    """
    A convenient wrapper around baycomp.

    Estimates the posterior mean difference (B-A) in performance between system A and system B.
    """

    def __init__(self, systems: list, system_names: list = None, runs=1):
        # the ROPE isn't actually needed at this point,
        # we can compute probs in relation to any given ROPE
        # after we get the posterior samples from baycomp (see the methods plot and probs below)
        # for sys in systems:
        #     assert len(sys) == runs, f"Each system should have {runs} runs, but got {len(sys)}"
        self._cts = []
        # for each system combination we want to compute the posterior mean difference (B-A)
        combinations = [(i, j) for i in range(len(systems))
                        for j in range(i + 1, len(systems))]

        for comb in combinations:
            sys_a = systems[comb[0]]
            sys_b = systems[comb[1]]
            self._cts.append(baycomp.CorrelatedTTest(
                np.array(sys_a),
                np.array(sys_b),
                rope=0.,
                runs=runs
            ))
            self._names = system_names

    @property
    def samples(self):
        """Sampled mean differences"""
        samples = []
        for cts in self._cts:
            samples.append(cts.samples)
        return samples

    @property
    def mean_diffs(self):
        """Posterior mean difference"""
        mean_diffs = []
        for cts in self._cts:
            mean_diffs.append(cts.mean)
        return mean_diffs

    @property
    def stds(self):
        """Posterior standard deviation"""
        stds = []
        for cts in self._cts:
            stds.append(np.sqrt(cts.var))
        return stds

    @property
    def dfs(self):
        """Posterior degrees of freedom"""
        dfs = []
        for cts in self._cts:
            dfs.append(cts.df)
        return dfs

    @property
    def effect_size(self):
        """
        Return mean difference divided by posterior std.
        This can be interpreted as improvement in "units of stds".
        There is no universal way to read this, 
         but some people like to think that < 0.2 indicates a minor effect
         more than 0.8 indicates a large effect
         and in between indicates moderate effect
        (Don't take these 'rules' too seriously)
        """
        return self.mean_diffs / self.stds

    def plot(self, rope_param, hdi=0.95, wider_hdi=0.999,
             num_points=1000, rope_factor=2.5):
        """
        Plot the posterior distributions for all pairwise system comparisons.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : numpy.ndarray
            One axis per pairwise comparison.
        """
        assert wider_hdi > hdi, "wider_hdi should be wider than hdi"

        pairs = list(combinations(range(len(self._names)), 2))
        n_comparisons = len(self._cts)

        fig, axes = plt.subplots(
            n_comparisons, 1,
            figsize=(10, 4 * n_comparisons),
            squeeze=False
        )
        axes = axes.flatten()

        for i, (cts, (a, b)) in enumerate(zip(self._cts, pairs)):
            ax = axes[i]

            name_a = self._names[a]
            name_b = self._names[b]

            # Posterior distribution for this comparison
            t = st.t(
                df=self.dfs[i],
                loc=self.mean_diffs[i],
                scale=self.stds[i]
            )

            # Plot range
            ci = t.interval(wider_hdi)
            x = np.linspace(
                min(ci[0], -rope_param * rope_factor),
                max(ci[1], rope_param * rope_factor),
                num_points
            )

            ax.set_xlabel(r"Mean difference: $\mu$ (B - A)")
            ax.plot(x, t.pdf(x), label=r"$p(\mu \mid \mathcal{D})$")

            # ROPE
            ax.axvspan(
                -rope_param, rope_param,
                color="teal", alpha=0.25,
                label=f"ROPE ({rope_param})"
            )

            # Region probabilities
            p_a = t.cdf(-rope_param)
            p_tie = t.cdf(rope_param) - p_a
            p_b = 1 - t.cdf(rope_param)

            transform = ax.get_xaxis_transform()

            ax.text(
                -rope_param, 0.2,
                f"{p_a * 100:.1f}%\n{name_a} wins",
                ha="right", va="center",
                color="gray", transform=transform, fontsize=9
            )

            ax.text(
                0, 0.5,
                f"{p_tie * 100:.1f}%\nTie",
                ha="center", va="center",
                color="gray", transform=transform, fontsize=9
            )

            ax.text(
                rope_param, 0.7,
                f"{p_b * 100:.1f}%\n{name_b} wins",
                ha="left", va="center",
                color="gray", transform=transform, fontsize=9
            )

            # HDI and median
            ci = t.interval(hdi)

            ax.hlines(
                y=0.02,
                xmin=ci[0], xmax=ci[1],
                color="black", linewidth=2,
                label=f"{hdi * 100:.1f}% HDI"
            )

            ax.plot(
                t.median(), 0.02, "d",
                color="black", label="Median"
            )

            ax.set_title(f"{name_a} vs. {name_b}")
            ax.legend()

        fig.tight_layout()
        return fig, axes

    def plot_all_systems(self, rope_param, hdi=0.95, wider_hdi=0.999,
            num_points=1000, rope_factor=2.5):
        """
        Plot all pairwise posterior distributions on a single axis.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        assert 0 < hdi < wider_hdi < 1
        assert rope_param >= 0

        pairs = list(combinations(self._names, 2))

        assert len(pairs) == len(self._cts), \
            "Number of comparisons does not match self._cts"

        # Construct a Student's t distribution for each comparison
        distributions = [
            st.t(df=df, loc=mean, scale=std)
            for df, mean, std in zip(
                self.dfs, self.mean_diffs, self.stds
            )
        ]

        # Shared x-axis range
        intervals = [t.interval(wider_hdi) for t in distributions]

        x_min = min(
            min(ci[0] for ci in intervals),
            -rope_param * rope_factor
        )
        x_max = max(
            max(ci[1] for ci in intervals),
            rope_param * rope_factor
        )

        x = np.linspace(x_min, x_max, num_points)

        fig, ax = plt.subplots(figsize=(11, 6))

        colors = plt.cm.tab10(
            np.linspace(0, 1, len(distributions))
        )

        # Draw the ROPE once
        ax.axvspan(
            -rope_param, rope_param,
            color="teal", alpha=0.15,
            label=f"ROPE (±{rope_param})"
        )

        # Draw all pairwise posterior distributions
        for i, (t, (name_a, name_b)) in enumerate(
            zip(distributions, pairs)
        ):
            color = colors[i]

            p_a = t.cdf(-rope_param)
            p_tie = t.cdf(rope_param) - p_a
            p_b = t.sf(rope_param)

            label = (
                f"{name_a} vs {name_b} "
                f"(A: {p_a:.1%}, "
                f"Tie: {p_tie:.1%}, "
                f"B: {p_b:.1%})"
            )

            ax.plot(
                x, t.pdf(x),
                color=color,
                linewidth=2,
                label=label
            )

            # 95% credible interval and posterior median
            ci = t.interval(hdi)

            # Place intervals below the density curves
            y_pos = -0.05 * (i + 1)

            ax.hlines(
                y=y_pos,
                xmin=ci[0],
                xmax=ci[1],
                color=color,
                linewidth=3
            )

            ax.plot(
                t.median(), y_pos,
                marker="d",
                color=color,
                markersize=6
            )

        ax.axvline(0, color="gray", linestyle=":", alpha=0.6)

        ax.set_xlabel("Mean performance difference (B - A)")
        ax.set_ylabel("Posterior density")
        ax.set_title("Pairwise Bayesian Model Comparisons")

        # Leave room for credible intervals below zero
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(
            -0.05 * (len(distributions) + 1),
            ymax
        )

        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.2)

        fig.tight_layout()

        return fig, ax




    def sampled_probs(self, rope_size: float):
        """
        rope_size: defines a ROPE around 0 (ie, -rope_size, +rope_size)
        
        Return 
            - prob that mean diff is to the left of the ROPE (ie, A wins)
            - prob that mean diff is inside the ROPE (ie, tie)
            - prob that mean diff is to the right of the ROPE (ie, B wins)
        """
        p_A = np.mean(self.samples < - rope_size)
        p_tie = np.mean(np.abs(self.samples) < rope_size)
        p_B = np.mean(self.samples > rope_size)

        return p_A, p_tie, p_B

    def probs(self, rope_param: float):
        """
        rope_param: defines a ROPE around 0 (ie, -rope_param, +rope_param)
        
        Return 
            - prob that mean diff is to the left of the ROPE (ie, A wins)
            - prob that mean diff is inside the ROPE (ie, tie)
            - prob that mean diff is to the right of the ROPE (ie, B wins)
        """
        p_A = st.t.cdf(-rope_param, self.dfs,
                       loc=self.mean_diffs, scale=self.stds)
        p_tie = st.t.cdf(rope_param, self.dfs, loc=self.mean_diffs, scale=self.stds) - \
            st.t.cdf(-rope_param, self.dfs,
                     loc=self.mean_diffs, scale=self.stds)
        p_B = 1 - st.t.cdf(rope_param, self.dfs,
                           loc=self.mean_diffs, scale=self.stds)
        return p_A, p_tie, p_B



