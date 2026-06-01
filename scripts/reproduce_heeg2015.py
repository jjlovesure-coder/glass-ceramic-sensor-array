from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thermal_history.experiments import (  # noqa: E402
    main_reconstruction_estimates,
    main_reconstruction_summary,
    regularization_sweep,
    spike_reconstruction_summary,
)
from thermal_history.plotting import (  # noqa: E402
    save_fend_reconstruction,
    save_main_histograms,
    save_regularization_sweep,
    save_spike_reconstruction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Heeg 2015 thermal history simulations.")
    parser.add_argument("--samples", type=int, default=2000, help="Monte Carlo samples per experiment.")
    parser.add_argument("--output", type=Path, default=Path("outputs/heeg2015"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=2015, help="Base random seed.")
    return parser.parse_args()


def write_output_readme(output_dir: Path, samples: int) -> None:
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Heeg 2015 Reproduction Outputs",
                "",
                f"Monte Carlo samples per primary experiment: {samples}",
                "",
                "Files:",
                "",
                "- `main_reconstruction_summary.csv`: five-interval reconstruction statistics.",
                "- `regularization_sweep.csv`: alpha sweep and sensor 21 end-crystallinity checks.",
                "- `spike_reconstruction_summary.csv`: short spike reconstruction statistics.",
                "- `fig_main_histograms.png`: histograms comparable to the paper's main examples.",
                "- `fig_regularization_sweep.png`: duration sensitivity to regularization.",
                "- `fig_fend_reconstruction.png`: reconstructed sensor 21 crystallinity.",
                "- `fig_spike_reconstruction.png`: estimated high-temperature duration versus spike temperature.",
                "",
                "The model uses the paper's pre-nucleated two-dimensional simplification.",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    main_estimates = main_reconstruction_estimates(samples=args.samples, seed=args.seed, method="nnls")
    main_summary = main_reconstruction_summary(samples=args.samples, seed=args.seed, method="nnls")
    sweep = regularization_sweep(samples=max(50, args.samples // 5), seed=args.seed + 1)
    spike = spike_reconstruction_summary(samples=args.samples, seed=args.seed + 2)

    main_summary.to_csv(args.output / "main_reconstruction_summary.csv", index=False)
    sweep.to_csv(args.output / "regularization_sweep.csv", index=False)
    spike.to_csv(args.output / "spike_reconstruction_summary.csv", index=False)

    save_main_histograms(main_estimates, args.output / "fig_main_histograms.png")
    save_regularization_sweep(sweep, args.output / "fig_regularization_sweep.png")
    save_fend_reconstruction(sweep, args.output / "fig_fend_reconstruction.png")
    save_spike_reconstruction(spike, args.output / "fig_spike_reconstruction.png")
    write_output_readme(args.output, args.samples)

    print(f"Wrote Heeg 2015 reproduction outputs to {args.output}")


if __name__ == "__main__":
    main()
