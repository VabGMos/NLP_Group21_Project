"""Matplotlib helpers for the embedding / dataset analysis notebook."""

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def _finish(fig, path=None):
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_label_counts(splits: dict, label_names, path=None):
    """Grouped bars: examples per label in every split."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.8 / len(splits)
    for k, (name, df) in enumerate(splits.items()):
        counts = df["label"].value_counts().sort_index()
        pos = np.arange(len(counts)) + k * width
        ax.bar(pos, counts.values, width, label=name, color=COLORS[k])
    ax.set_xticks(np.arange(len(label_names)) + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(label_names)
    ax.set_ylabel("examples")
    ax.set_title("Label distribution per split")
    ax.legend()
    _finish(fig, path)


def plot_length_distributions(lengths: dict, max_length: int, path=None):
    """Token-length histograms; ``lengths`` maps a title to {'premise', 'hypothesis'}."""
    fig, axes = plt.subplots(
        1, len(lengths), figsize=(4.5 * len(lengths), 3.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    for ax, (title, parts) in zip(axes, lengths.items()):
        top = int(max(arr.max() for arr in parts.values()))
        bins = np.arange(0, top + 2, 2)
        for k, (part, arr) in enumerate(parts.items()):
            ax.hist(arr, bins=bins, alpha=0.6, label=f"{part} (max {arr.max()})",
                    color=COLORS[k])
        ax.set_title(f"{title} (max_length={max_length} never reached)"
                     if top < max_length else title)
        ax.set_xlabel("tokens (incl. special)")
    axes[0].set_ylabel("texts")
    axes[0].legend()
    _finish(fig, path)


def plot_distribution_by_label(
    panels: dict, labels, label_names, xlabel, path=None, n_bins: int = 40
):
    """One histogram panel per entry in ``panels`` (title -> values), split by label."""
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.2 * len(panels), 3.5), sharey=True
    )
    axes = np.atleast_1d(axes)
    labels = np.asarray(labels)
    for ax, (title, values) in zip(axes, panels.items()):
        values = np.asarray(values)
        bins = np.linspace(values.min(), values.max(), n_bins)
        for k, name in enumerate(label_names):
            ax.hist(
                values[labels == k],
                bins=bins,
                histtype="step",
                density=True,
                lw=1.8,
                label=name,
                color=COLORS[k],
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("density")
    axes[0].legend()
    _finish(fig, path)


def plot_pieces_per_word(pieces, freq, max_pieces: int = 5, path=None):
    """Share of unique words and of running words split into 1, 2, ... subword pieces."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    capped = np.minimum(pieces, max_pieces)
    series = {
        "unique words": np.bincount(capped, minlength=max_pieces + 1)[1:] / len(pieces),
        "running words": np.bincount(capped, weights=freq, minlength=max_pieces + 1)[1:]
        / freq.sum(),
    }
    width = 0.4
    for k, (name, share) in enumerate(series.items()):
        ax.bar(np.arange(1, max_pieces + 1) + k * width, share, width, label=name,
               color=COLORS[k])
    ax.set_xticks(np.arange(1, max_pieces + 1) + width / 2)
    ax.set_xticklabels([str(i) for i in range(1, max_pieces)] + [f"{max_pieces}+"])
    ax.set_xlabel("subword pieces per word")
    ax.set_ylabel("share")
    ax.set_title("Word fragmentation")
    ax.legend()
    _finish(fig, path)


def plot_coverage(counts: dict, path=None):
    """Cumulative corpus mass covered by the k most frequent vocabulary entries."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k, (name, c) in enumerate(counts.items()):
        ordered = np.sort(c)[::-1]
        cum = np.cumsum(ordered) / ordered.sum()
        used = int((c > 0).sum())
        ax.plot(np.arange(1, len(c) + 1), cum, color=COLORS[k],
                label=f"{name} ({used / len(c):.0%} of vocab used)")
    ax.set_xscale("log")
    ax.set_xlabel("vocabulary entries, most frequent first")
    ax.set_ylabel("cumulative share of tokens")
    ax.set_title("Corpus coverage by vocabulary rank")
    ax.legend(fontsize=8)
    _finish(fig, path)


def plot_norm_vs_frequency(tables: dict, counts: dict, path=None):
    """Embedding L2 norm against corpus frequency for every token seen in the corpus."""
    fig, axes = plt.subplots(
        1, len(tables), figsize=(4.2 * len(tables), 3.5), sharey=False
    )
    axes = np.atleast_1d(axes)
    for ax, (name, w) in zip(axes, tables.items()):
        c = counts[name]
        norms = w.norm(dim=1).numpy()
        seen = c > 0
        ax.scatter(c[seen], norms[seen], s=3, alpha=0.3, color=COLORS[0])
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("corpus frequency")
    axes[0].set_ylabel("embedding L2 norm")
    _finish(fig, path)


def plot_pair_similarities(sims: dict, synonyms, antonyms, path=None):
    """Per-pair cosine similarity (synonym row / antonym row) for every model."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharey=True)
    for ax, key, pairs in zip(axes, ["synonyms", "antonyms"], [synonyms, antonyms]):
        width = 0.8 / len(sims)
        x = np.arange(len(pairs))
        for k, (name, per_kind) in enumerate(sims.items()):
            ax.bar(x + k * width, per_kind[key], width, label=name, color=COLORS[k])
        ax.set_xticks(x + width * (len(sims) - 1) / 2)
        ax.set_xticklabels([f"{a}/{b}" for a, b in pairs], rotation=30, ha="right")
        ax.set_title(key)
        ax.set_ylabel("cosine")
    axes[0].legend(fontsize=8)
    _finish(fig, path)


def plot_relation_summary(summary: dict, path=None):
    """Mean cosine of synonym / antonym / random pairs per model."""
    kinds = ["synonyms", "antonyms", "random pairs"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.8 / len(summary)
    for k, (name, row) in enumerate(summary.items()):
        ax.bar(np.arange(len(kinds)) + k * width, [row[t] for t in kinds], width,
               label=name, color=COLORS[k])
    ax.set_xticks(np.arange(len(kinds)) + width * (len(summary) - 1) / 2)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("mean cosine")
    ax.set_title("Similarity by word relation")
    ax.legend(fontsize=8)
    _finish(fig, path)


def plot_word_maps(maps: dict, groups: dict, path=None):
    """2-D PCA maps of word groups, one panel per model.

    ``maps`` maps a model name to {word: (x, y)}; ``groups`` maps word -> group name.
    """
    names = sorted(set(groups.values()))
    fig, axes = plt.subplots(1, len(maps), figsize=(5.5 * len(maps), 5))
    axes = np.atleast_1d(axes)
    for ax, (title, points) in zip(axes, maps.items()):
        for word, (x, y) in points.items():
            color = COLORS[names.index(groups[word])]
            ax.scatter(x, y, color=color, s=18)
            ax.annotate(word, (x, y), fontsize=8, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLORS[i], label=n)
        for i, n in enumerate(names)
    ]
    axes[0].legend(handles=handles, fontsize=8)
    _finish(fig, path)


def plot_probe(probe, path=None):
    """Mean probe accuracy per embedding with the individual runs as dots."""
    run_cols = [c for c in probe.columns if c.startswith("seed")]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(probe))
    ax.bar(x, probe["mean"], yerr=probe["std"], capsize=4, color=COLORS[0], alpha=0.8)
    for col in run_cols:
        ax.scatter(x, probe[col], color="k", s=10, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(probe.index, rotation=15, ha="right")
    ax.set_ylim(0.5, max(0.7, probe["mean"].max() + 0.03))
    ax.set_ylabel("validation accuracy")
    ax.set_title("Frozen embed-and-pool probe (mean +/- std over runs)")
    _finish(fig, path)
