from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from .model import FEATURE_INDEX, PosteriorSnapshot

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

FEATURE_NAMES = list(FEATURE_INDEX.keys())


def _ensure_dir(output_dir: str | Path) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_weight_evolution(
    history: list[PosteriorSnapshot],
    output_dir: str | Path = "output",
) -> Path:
    """3x3 grid: posterior mean +/- 95% credible interval for each audio feature."""
    out = _ensure_dir(output_dir)

    steps = np.array([s.step for s in history])
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    axes_flat = axes.flatten()

    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes_flat[i]
        idx = FEATURE_INDEX[feat]

        means = np.array([s.mu[idx] for s in history])
        stds = np.array([np.sqrt(s.sigma_diag[idx]) for s in history])

        ax.plot(steps, means, linewidth=1.5)
        ax.fill_between(
            steps,
            means - 1.96 * stds,
            means + 1.96 * stds,
            alpha=0.25,
        )
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_title(feat)
        ax.set_ylabel("weight")

    for ax in axes_flat[6:]:
        ax.set_xlabel("observation")

    fig.suptitle("Posterior Weight Evolution with 95% Credible Intervals", y=1.01)
    fig.tight_layout()
    path = out / "weight_evolution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prior_vs_posterior(
    history: list[PosteriorSnapshot],
    output_dir: str | Path = "output",
) -> Path:
    """Overlay prior and final posterior densities for each feature."""
    out = _ensure_dir(output_dir)
    prior = history[0]
    posterior = history[-1]

    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes_flat = axes.flatten()

    for i, feat in enumerate(FEATURE_NAMES):
        ax = axes_flat[i]
        idx = FEATURE_INDEX[feat]

        mu0, sd0 = prior.mu[idx], np.sqrt(prior.sigma_diag[idx])
        muT, sdT = posterior.mu[idx], np.sqrt(posterior.sigma_diag[idx])

        lo = min(mu0 - 3.5 * sd0, muT - 3.5 * sdT)
        hi = max(mu0 + 3.5 * sd0, muT + 3.5 * sdT)
        xs = np.linspace(lo, hi, 300)

        ax.plot(xs, norm.pdf(xs, mu0, sd0), label="prior", linestyle="--")
        ax.plot(xs, norm.pdf(xs, muT, sdT), label="posterior")
        ax.axvline(mu0, color="C0", linewidth=0.5, linestyle=":")
        ax.axvline(muT, color="C1", linewidth=0.5, linestyle=":")
        ax.set_title(feat)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Prior vs Posterior Density for Each Feature Weight", y=1.01)
    fig.tight_layout()
    path = out / "prior_vs_posterior.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_entropy(
    history: list[PosteriorSnapshot],
    output_dir: str | Path = "output",
) -> Path:
    """Posterior entropy (information gain) over observations."""
    out = _ensure_dir(output_dir)

    steps = [s.step for s in history]
    entropies = [s.entropy for s in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, entropies, marker="o", markersize=4, linewidth=1.5)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Posterior Entropy (nats)")
    ax.set_title("Posterior Entropy Over Time (Information Gain)")
    fig.tight_layout()
    path = out / "entropy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_map_vs_posterior_predictions(
    history: list[PosteriorSnapshot],
    output_dir: str | Path = "output",
) -> Path:
    """Scatter: MAP prediction vs predictive-posterior prediction, colored by outcome."""
    out = _ensure_dir(output_dir)

    obs = [s for s in history if s.pred_map is not None and s.y is not None]
    if not obs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No observations recorded", ha="center", va="center")
        path = out / "map_vs_posterior.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    maps = np.array([s.pred_map for s in obs])
    posts = np.array([s.pred_posterior for s in obs])
    ys = np.array([s.y for s in obs])
    steps = np.array([s.step for s in obs])

    fig, ax = plt.subplots(figsize=(7, 6))

    played = ys == 1
    ax.scatter(
        maps[played], posts[played],
        c=steps[played], cmap="Greens", edgecolors="green",
        label="played", s=50, alpha=0.8,
    )
    ax.scatter(
        maps[~played], posts[~played],
        c=steps[~played], cmap="Reds",
        marker="x", label="skipped", s=50, alpha=0.8,
    )

    lims = [0, 1]
    ax.plot(lims, lims, "k--", linewidth=0.5, label="MAP = posterior")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("MAP P(like)")
    ax.set_ylabel("Predictive Posterior P(like)")
    ax.set_title("MAP vs Predictive Posterior Predictions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "map_vs_posterior.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_diagnostics(
    history: list[PosteriorSnapshot],
    output_dir: str | Path = "output",
) -> list[Path]:
    """Run all four diagnostic plots and return the saved file paths."""
    return [
        plot_weight_evolution(history, output_dir),
        plot_prior_vs_posterior(history, output_dir),
        plot_entropy(history, output_dir),
        plot_map_vs_posterior_predictions(history, output_dir),
    ]
