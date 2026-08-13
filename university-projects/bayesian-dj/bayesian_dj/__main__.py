from __future__ import annotations

import argparse
from pathlib import Path

from .session import DJSession, DEFAULT_CSV
from .simulation import run_full_simulation


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bayesian DJ -- interactive playlist builder with Bayesian logistic regression."
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Music prompt describing what you want to hear.",
    )
    ap.add_argument(
        "--playlist-length",
        type=int,
        default=30,
        help="Maximum number of songs in the session (default: 30).",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the kaggle_dataset.csv file (auto-detected by default).",
    )
    ap.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for the query parser.",
    )
    ap.add_argument(
        "--cache-dir",
        type=str,
        default=".cache/music_query_parser",
        help="Cache directory for parser model / embeddings.",
    )
    ap.add_argument(
        "--analyze",
        action="store_true",
        help="Generate diagnostic plots after the session ends.",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for diagnostic and simulation plots (default: output/).",
    )
    ap.add_argument(
        "--simulate",
        action="store_true",
        help="Run headless simulation (strategy comparison + prior sensitivity) instead of interactive mode.",
    )
    ap.add_argument(
        "--sim-rounds",
        type=int,
        default=50,
        help="Number of rounds per simulation run (default: 50).",
    )
    ap.add_argument(
        "--sim-repeats",
        type=int,
        default=20,
        help="Number of repeat trials per strategy (default: 20).",
    )
    ap.add_argument(
        "--user-profile",
        type=str,
        default="chill_listener",
        help="Synthetic user profile for simulation: chill_listener or party_lover.",
    )
    args = ap.parse_args()

    csv_path = args.csv or str(DEFAULT_CSV)

    if args.simulate:
        genres = None
        constraints = None
        if args.prompt:
            from music_query_parser.parser import MusicQueryParser
            parser = MusicQueryParser(
                model_name=args.model_name, cache_dir=args.cache_dir
            )
            spec = parser.parse(args.prompt)
            genres = spec.genres or None
            constraints = dict(spec.constraints) if spec.constraints else None

        run_full_simulation(
            csv_path=csv_path,
            prompt_constraints=constraints,
            genres=genres,
            user_profile=args.user_profile,
            n_rounds=args.sim_rounds,
            n_repeats=args.sim_repeats,
            output_dir=args.output_dir,
        )
        return

    session_kwargs: dict = dict(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        playlist_length=args.playlist_length,
        analyze=args.analyze,
        output_dir=args.output_dir,
    )
    if args.csv:
        session_kwargs["csv_path"] = args.csv

    session = DJSession(**session_kwargs)

    if args.prompt:
        session.run_interactive(args.prompt)
    else:
        print("Bayesian DJ -- enter a music prompt to start.\n")
        while True:
            try:
                prompt = input("Prompt > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not prompt:
                continue
            if prompt.lower() in {"quit", "exit"}:
                break
            session = DJSession(**session_kwargs)
            session.run_interactive(prompt)
            print()


if __name__ == "__main__":
    main()
