"""Command-line entry point for churn-uplift."""

from __future__ import annotations

import argparse

from .pipeline import run_pipeline


def _format_result(r) -> str:
    return (
        f"  qini_coefficient       : {r.qini_coefficient:+.4f}\n"
        f"  auuc                   : {r.auuc:+.4f}\n"
        f"  uplift@10%             : {r.uplift_at_10pct:+.4f}\n"
        f"  corr(pred, true uplift): {r.true_uplift_correlation:+.4f}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print metrics for both learners."""
    parser = argparse.ArgumentParser(
        prog="upliftkit",
        description="Train S/T-learner uplift models on synthetic retention data "
        "and report uplift metrics.",
    )
    parser.add_argument("-n", "--n-samples", type=int, default=4000, help="number of customers")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument("--test-size", type=float, default=0.3, help="held-out fraction")
    args = parser.parse_args(argv)

    result = run_pipeline(n=args.n_samples, seed=args.seed, test_size=args.test_size)

    print("churn-uplift :: retention uplift modeling")
    print(f"(synthetic customers: {args.n_samples}, seed: {args.seed})\n")
    for name, r in result.results.items():
        print(name)
        print(_format_result(r))
        print()
    print(f"Best learner by Qini coefficient: {result.best_learner}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
