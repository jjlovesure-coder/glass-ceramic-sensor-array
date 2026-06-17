from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thermal_history.experiments import (  # noqa: E402
    MAIN_DURATIONS_S,
    MAIN_TEMPERATURES_C,
    spike_reconstruction_summary,
)
from thermal_history.figure_comparison import (  # noqa: E402
    build_metric_comparison,
    paper_figure_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-vs-reproduction figure comparisons.")
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            "paper/Heeg - 2015 - Equivalent thermal history reconstruction from a partially crystallized glass-ceramic sensor array.pdf"
        ),
        help="Heeg 2015 PDF.",
    )
    parser.add_argument("--reproduction", type=Path, default=Path("outputs/heeg2015"))
    parser.add_argument("--output", type=Path, default=Path("outputs/heeg2015_comparison"))
    parser.add_argument("--samples", type=int, default=2000, help="Samples for regenerated comparison-only Fig. 9.")
    parser.add_argument("--seed", type=int, default=2017)
    return parser.parse_args()


def render_paper_pages(pdf_path: Path, page_dir: Path) -> None:
    page_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "180",
            "-f",
            "5",
            "-l",
            "8",
            str(pdf_path),
            str(page_dir / "page"),
        ],
        check=True,
    )


def crop_paper_figures(page_dir: Path, crop_dir: Path) -> None:
    crop_dir.mkdir(parents=True, exist_ok=True)
    for spec in paper_figure_specs():
        source = page_dir / f"page-{spec.page_number}.png"
        with Image.open(source) as page:
            page.crop(spec.crop_box).save(crop_dir / f"{spec.figure}_paper.png")


def build_reproduction_figures(reproduction_dir: Path, target_dir: Path, samples: int, seed: int) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    _save_fig2_profile(target_dir / "fig2_temperature_profile.png")
    _save_fig8_profile(target_dir / "fig8_spike_profile.png")
    _save_fig9_paper_range(target_dir / "fig_spike_reconstruction.png", samples=samples, seed=seed)

    copies = {
        "fig3_regularized_lls_histograms.png": "fig3_regularized_lls_histograms.png",
        "fig4_total_time_constrained_histograms.png": "fig4_total_time_constrained_histograms.png",
        "fig_main_histograms.png": "fig5_nnls_histograms.png",
        "fig_regularization_sweep.png": "fig_regularization_sweep.png",
        "fig_fend_reconstruction.png": "fig_fend_reconstruction.png",
    }
    for source_name, target_name in copies.items():
        shutil.copyfile(reproduction_dir / source_name, target_dir / target_name)


def _save_fig2_profile(output_path: Path) -> None:
    edges = [0.0]
    for duration in MAIN_DURATIONS_S:
        edges.append(edges[-1] + float(duration))
    temps = list(MAIN_TEMPERATURES_C) + [MAIN_TEMPERATURES_C[-1]]
    fig, axis = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    axis.step(edges, temps, where="post", color="red", linewidth=2)
    axis.set_xlim(0, 1600)
    axis.set_ylim(680, 1120)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Temperature (C)")
    axis.set_title("Fig. 2 reproduction: input thermal history")
    axis.grid(True, color="#ddddff", linewidth=0.7)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_fig8_profile(output_path: Path) -> None:
    times = [0, 600, 800, 810, 1200, 1600]
    temps = [900, 1000, 1100, 1000, 900, 900]
    fig, axis = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    axis.step(times, temps, where="post", color="blue", linewidth=2)
    axis.set_xlim(0, 1600)
    axis.set_ylim(880, 1120)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Temperature (C)")
    axis.set_title("Fig. 8 reproduction: short spike profile")
    axis.grid(True, color="#ddddff", linewidth=0.7)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _save_fig9_paper_range(output_path: Path, samples: int, seed: int) -> None:
    spike = spike_reconstruction_summary(
        samples=samples,
        seed=seed,
        spike_temperatures_c=pd.Series([1080, 1090, 1100, 1110, 1120, 1130, 1140], dtype=float).to_numpy(),
    )
    fig, axis = plt.subplots(figsize=(6.2, 4), constrained_layout=True)
    axis.errorbar(
        spike["spike_temperature_c"],
        spike["t3_mean_s"],
        yerr=spike["t3_std_s"],
        marker="o",
        linewidth=1,
        capsize=3,
        color="#444444",
        markerfacecolor="white",
    )
    axis.set_xlabel("Tspike (C)")
    axis.set_ylabel("t3_est (s)")
    axis.set_title("Fig. 9 reproduction: spike-temperature sensitivity")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_side_by_side(crop_dir: Path, reproduction_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for spec in paper_figure_specs():
        paper = Image.open(crop_dir / f"{spec.figure}_paper.png").convert("RGB")
        reproduction = Image.open(reproduction_dir / spec.reproduction_name).convert("RGB")
        comparison = _compose_pair(paper, reproduction, f"{spec.figure.upper()} - paper", "reproduction", font)
        comparison.save(output_dir / f"{spec.figure}_comparison.png")


def _compose_pair(paper: Image.Image, reproduction: Image.Image, left_label: str, right_label: str, font: ImageFont.ImageFont) -> Image.Image:
    target_height = 620
    paper = _resize_to_height(paper, target_height)
    reproduction = _resize_to_height(reproduction, target_height)
    label_height = 34
    gap = 24
    width = paper.width + reproduction.width + gap
    height = target_height + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), left_label, fill="black", font=font)
    draw.text((paper.width + gap + 8, 8), right_label, fill="black", font=font)
    canvas.paste(paper, (0, label_height))
    canvas.paste(reproduction, (paper.width + gap, label_height))
    return canvas


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_metric_outputs(reproduction_dir: Path, output_dir: Path) -> pd.DataFrame:
    validation = pd.read_csv(reproduction_dir / "paper_target_validation.csv")
    metrics = build_metric_comparison(validation)
    metrics.to_csv(output_dir / "metric_comparison.csv", index=False)
    return metrics


def write_analysis(output_dir: Path, metrics: pd.DataFrame) -> None:
    failed = metrics[metrics["status"] == "fail"]
    passed = metrics[metrics["status"] == "pass"]
    lines = [
        "# Heeg 2015 Figure Comparison",
        "",
        "This folder compares cropped original paper figures with the current reproduction outputs.",
        "",
        "Generated folders:",
        "",
        "- `paper_crops/`: cropped figures from the PDF.",
        "- `reproduction_figures/`: current reproduction figures normalized to the paper figure numbers.",
        "- `side_by_side/`: original-vs-reproduction visual comparisons.",
        "- `metric_comparison.csv`: numeric target comparison extracted from paper text targets.",
        "",
        f"Metric targets passed: {len(passed)} / {len(metrics)}.",
        "",
        "Main observations:",
        "",
        "- Fig. 2, Fig. 6 noiseless recovery, and sensor 21 exact `Fend` are reproduced within the configured tolerance.",
        "- Fig. 3 high-temperature mean is close, but the current 5% fractional-crystallinity noise model gives much wider distributions than the paper.",
        "- Fig. 4 shows the correct total-time constrained method, but the remaining visual mismatch is again dominated by noisy spread.",
        "- Fig. 5 is labelled as SciPy NNLS objective reproduction, not Kim PQN implementation; the mean `Fend` matches, while the spread remains larger than the paper value.",
        "- Fig. 8/9 reproduce the spike profile and monotonic trend; the spike-duration means are close, but the standard deviations are still larger than the paper text targets.",
        "",
        "Failed 5% text targets:",
        "",
    ]
    if failed.empty:
        lines.append("- None.")
    else:
        for row in failed.itertuples(index=False):
            lines.append(
                f"- {row.figure}.{row.metric}: observed={row.observed:.6g}, "
                f"target={row.target:.6g}, error={row.relative_error_percent:.2f}%"
            )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    page_dir = args.output / "rendered_pages"
    crop_dir = args.output / "paper_crops"
    reproduction_dir = args.output / "reproduction_figures"
    side_by_side_dir = args.output / "side_by_side"
    args.output.mkdir(parents=True, exist_ok=True)

    render_paper_pages(args.paper, page_dir)
    crop_paper_figures(page_dir, crop_dir)
    build_reproduction_figures(args.reproduction, reproduction_dir, args.samples, args.seed)
    build_side_by_side(crop_dir, reproduction_dir, side_by_side_dir)
    metrics = write_metric_outputs(args.reproduction, args.output)
    write_analysis(args.output, metrics)
    print(f"Wrote comparison outputs to {args.output}")


if __name__ == "__main__":
    main()
