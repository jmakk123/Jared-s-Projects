"""Headless simulation framework for evaluating Bayesian recommendation strategies.

Replaces the human with a synthetic user whose preferences are defined by a
known ground-truth beta vector, enabling reproducible quantitative comparison
of Thompson sampling vs greedy vs random vs epsilon-greedy policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit as sigmoid

from .model import N_FEATURES, BayesianLogisticRegression, FEATURE_INDEX
from .song_pool import SongPool, AUDIO_FEATURES

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

# ── Synthetic user profiles ──────────────────────────────────────────────────

SYNTHETIC_USERS: dict[str, np.ndarray] = {
    "chill_listener": np.array([
        0.0,    # bias
        -0.5,   # danceability (slight negative)
        -2.0,   # energy (strongly prefers low)
        0.0,    # loudness
        -0.3,   # speechiness
        1.8,    # acousticness (strongly prefers high)
        0.0,    # instrumentalness
        0.0,    # liveness
        -0.8,   # valence (slightly melancholic)
        -0.5,   # tempo (prefers slower)
    ]),
    "party_lover": np.array([
        0.0,    # bias
        2.0,    # danceability
        1.5,    # energy
        0.5,    # loudness
        0.0,    # speechiness
        -1.5,   # acousticness
        -0.5,   # instrumentalness
        0.3,    # liveness
        1.5,    # valence (happy)
        0.8,    # tempo (faster)
    ]),
}


@dataclass
class SimulationResult:
    strategy: str
    played: list[bool] = field(default_factory=list)
    cumulative_play_rate: list[float] = field(default_factory=list)
    regret: list[float] = field(default_factory=list)


# ── Selection strategies ─────────────────────────────────────────────────────

def _select_thompson(model: BayesianLogisticRegression, X: np.ndarray) -> int:
    scores = model.thompson_sample_scores(X)
    return int(np.argmax(scores))


def _select_greedy(model: BayesianLogisticRegression, X: np.ndarray) -> int:
    scores = model.predict_proba(X)
    return int(np.argmax(scores))


def _select_random(model: BayesianLogisticRegression, X: np.ndarray) -> int:
    return int(np.random.randint(X.shape[0]))


def _select_epsilon_greedy(
    model: BayesianLogisticRegression, X: np.ndarray, epsilon: float = 0.1
) -> int:
    if np.random.rand() < epsilon:
        return int(np.random.randint(X.shape[0]))
    return _select_greedy(model, X)


STRATEGIES = {
    "thompson": _select_thompson,
    "greedy": _select_greedy,
    "random": _select_random,
    "epsilon_greedy": _select_epsilon_greedy,
}


# ── Core simulation loop ────────────────────────────────────────────────────

def _run_single(
    strategy_name: str,
    true_beta: np.ndarray,
    pool: SongPool,
    constraints: dict[str, tuple[float, float]],
    n_rounds: int,
    scale: float = 2.0,
    constrained_var: float = 0.5,
    rng_seed: int | None = None,
) -> SimulationResult:
    if rng_seed is not None:
        np.random.seed(rng_seed)

    model = BayesianLogisticRegression.from_constraints(
        constraints, scale=scale, constrained_var=constrained_var
    )
    select_fn = STRATEGIES[strategy_name]
    result = SimulationResult(strategy=strategy_name)

    available = pool.available_indices().copy()
    np.random.shuffle(available)

    feat_matrix_full = pool.get_feature_matrix()
    all_indices = pool.available_indices()
    idx_to_local = {int(idx): i for i, idx in enumerate(all_indices)}

    used = set()
    total_played = 0

    optimal_probs = sigmoid(feat_matrix_full @ true_beta)

    for t in range(n_rounds):
        mask = np.array([i for i in range(feat_matrix_full.shape[0])
                         if int(all_indices[i]) not in used])
        if len(mask) == 0:
            break

        X_avail = feat_matrix_full[mask]

        local_choice = select_fn(model, X_avail)
        global_local = mask[local_choice]
        pool_idx = int(all_indices[global_local])

        x = feat_matrix_full[global_local]
        p_play = float(sigmoid(true_beta @ x))
        played = np.random.rand() < p_play

        model.update(x, int(played))
        used.add(pool_idx)

        total_played += int(played)
        result.played.append(played)
        result.cumulative_play_rate.append(total_played / (t + 1))

        optimal_in_pool = np.max(optimal_probs[mask])
        result.regret.append(optimal_in_pool - p_play)

    return result


def run_strategy_comparison(
    pool: SongPool,
    constraints: dict[str, tuple[float, float]],
    true_beta: np.ndarray | None = None,
    user_profile: str = "chill_listener",
    n_rounds: int = 50,
    n_repeats: int = 20,
    seed: int = 42,
) -> dict[str, list[SimulationResult]]:
    """Run all four strategies n_repeats times and collect results."""
    if true_beta is None:
        true_beta = SYNTHETIC_USERS[user_profile]

    results: dict[str, list[SimulationResult]] = {s: [] for s in STRATEGIES}

    for strategy in STRATEGIES:
        for rep in range(n_repeats):
            r = _run_single(
                strategy_name=strategy,
                true_beta=true_beta,
                pool=pool,
                constraints=constraints,
                n_rounds=n_rounds,
                rng_seed=seed + rep * 100 + hash(strategy) % 1000,
            )
            results[strategy].append(r)

    return results


def run_prior_sensitivity(
    pool: SongPool,
    constraints: dict[str, tuple[float, float]],
    true_beta: np.ndarray | None = None,
    user_profile: str = "chill_listener",
    n_rounds: int = 50,
    n_repeats: int = 20,
    seed: int = 42,
    scales: list[float] | None = None,
    constrained_vars: list[float] | None = None,
) -> dict[str, list[SimulationResult]]:
    """Re-run Thompson sampling with different prior configurations."""
    if true_beta is None:
        true_beta = SYNTHETIC_USERS[user_profile]
    if scales is None:
        scales = [0.5, 1.0, 2.0, 4.0]
    if constrained_vars is None:
        constrained_vars = [0.1, 0.5, 1.0, 3.0]

    results: dict[str, list[SimulationResult]] = {}

    for s in scales:
        label = f"scale={s}"
        results[label] = []
        for rep in range(n_repeats):
            r = _run_single(
                strategy_name="thompson",
                true_beta=true_beta,
                pool=pool,
                constraints=constraints,
                n_rounds=n_rounds,
                scale=s,
                rng_seed=seed + rep * 100,
            )
            results[label].append(r)

    for cv in constrained_vars:
        label = f"var={cv}"
        results[label] = []
        for rep in range(n_repeats):
            r = _run_single(
                strategy_name="thompson",
                true_beta=true_beta,
                pool=pool,
                constraints=constraints,
                n_rounds=n_rounds,
                constrained_var=cv,
                rng_seed=seed + rep * 100,
            )
            results[label].append(r)

    return results


# ── Plotting ─────────────────────────────────────────────────────────────────

def _avg_curves(
    results: list[SimulationResult], attr: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lo_ci, hi_ci) across repeats for a given attribute."""
    max_len = max(len(getattr(r, attr)) for r in results)
    matrix = np.full((len(results), max_len), np.nan)
    for i, r in enumerate(results):
        vals = getattr(r, attr)
        matrix[i, : len(vals)] = vals
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    return mean, mean - 1.96 * std, mean + 1.96 * std


def plot_strategy_comparison(
    results: dict[str, list[SimulationResult]],
    output_dir: str | Path = "output",
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    strategy_labels = {
        "thompson": "Thompson Sampling",
        "greedy": "Greedy (MAP)",
        "random": "Random",
        "epsilon_greedy": "Epsilon-Greedy (0.1)",
    }

    for attr, ylabel, title, fname in [
        ("cumulative_play_rate", "Cumulative Play Rate", "Strategy Comparison: Cumulative Play Rate", "strategy_play_rate.png"),
        ("regret", "Instantaneous Regret", "Strategy Comparison: Regret", "strategy_regret.png"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for strategy, runs in results.items():
            if not runs:
                continue
            mean, lo, hi = _avg_curves(runs, attr)
            x = np.arange(1, len(mean) + 1)
            label = strategy_labels.get(strategy, strategy)
            ax.plot(x, mean, linewidth=1.5, label=label)
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths


def plot_prior_sensitivity(
    results: dict[str, list[SimulationResult]],
    output_dir: str | Path = "output",
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    scale_results = {k: v for k, v in results.items() if k.startswith("scale=")}
    var_results = {k: v for k, v in results.items() if k.startswith("var=")}

    for subset, title_suffix, fname in [
        (scale_results, "Prior Scale", "sensitivity_scale.png"),
        (var_results, "Prior Variance", "sensitivity_variance.png"),
    ]:
        if not subset:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for label, runs in sorted(subset.items()):
            mean, lo, hi = _avg_curves(runs, "cumulative_play_rate")
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, linewidth=1.5, label=label)
            ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_xlabel("Round")
        ax.set_ylabel("Cumulative Play Rate")
        ax.set_title(f"Prior Sensitivity Analysis: {title_suffix}")
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    return paths


def run_full_simulation(
    csv_path: str | Path,
    prompt_constraints: dict[str, tuple[float, float]] | None = None,
    genres: list[str] | None = None,
    user_profile: str = "chill_listener",
    n_rounds: int = 50,
    n_repeats: int = 20,
    output_dir: str | Path = "output",
    seed: int = 42,
) -> list[Path]:
    """One-call entry point: load data, run strategies + sensitivity, plot everything."""
    if prompt_constraints is None:
        prompt_constraints = {"energy": (0.2, 0.55), "acousticness": (0.35, 1.0)}

    pool = SongPool(csv_path)
    if genres:
        pool.filter_by_genres(genres)

    print(f"Simulation pool: {pool.n_available:,} songs")
    print(f"Synthetic user: {user_profile}")
    print(f"Running {n_repeats} repeats x {n_rounds} rounds per strategy...\n")

    strat_results = run_strategy_comparison(
        pool, prompt_constraints,
        user_profile=user_profile,
        n_rounds=n_rounds, n_repeats=n_repeats, seed=seed,
    )
    print("Strategy comparison complete.")

    sens_results = run_prior_sensitivity(
        pool, prompt_constraints,
        user_profile=user_profile,
        n_rounds=n_rounds, n_repeats=n_repeats, seed=seed,
    )
    print("Prior sensitivity analysis complete.\n")

    paths = plot_strategy_comparison(strat_results, output_dir)
    paths += plot_prior_sensitivity(sens_results, output_dir)

    for p in paths:
        print(f"  Saved: {p}")

    return paths
