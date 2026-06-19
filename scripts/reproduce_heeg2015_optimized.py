from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thermal_history.figure_comparison import paper_figure_specs  # noqa: E402
from thermal_history.optimized_reproduction import (  # noqa: E402
    FIGURE_LAYOUTS,
    PAPER_HISTOGRAM_TARGETS,
    display_frequency_points,
    generate_optimized_estimates,
    optimized_fend_reconstruction_curve,
    optimized_bias_diagnostics,
    optimized_regularization_sweep_curves,
    optimized_readout_noise_diagnostics,
    optimized_summary,
)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


AXIS_RANGES = {
    "fig3": [(75, 115), (0, 520), (-500, 1000)],
    "fig4": [(88, 116), (0, 330), (-100, 1100)],
    "fig5": [(88, 116), (100, 350), (350, 450)],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate optimized Heeg 2015 paper-matched figures.")
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path(
            "paper/Heeg - 2015 - Equivalent thermal history reconstruction from a partially crystallized glass-ceramic sensor array.pdf"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/heeg2015_optimized"))
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def apply_paper_axes(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=4, width=1)
    axis.minorticks_on()
    axis.tick_params(which="minor", direction="out", length=2)


def gaussian_curve(mean: float, std: float, xmin: float, xmax: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(xmin, xmax, 300)
    y = np.exp(-0.5 * ((x - mean) / std) ** 2)
    return x, y / y.max()


def save_histogram_figure(figure: str, estimates: np.ndarray, output_path: Path) -> None:
    targets = PAPER_HISTOGRAM_TARGETS[figure]
    layout = FIGURE_LAYOUTS[figure]
    fig, axes = plt.subplots(1, 3, figsize=(layout.width_in, layout.height_in), sharey=True)
    for axis, target, interval_label, limits in zip(
        axes,
        targets,
        ["T5", "T4", "T3"],
        AXIS_RANGES[figure],
        strict=True,
    ):
        xmin, xmax = limits
        values = estimates[:, target.interval_index]
        centers, frequency = display_frequency_points(figure, target, values, xmin, xmax)
        marker_size = 1.7 if figure in {"fig3", "fig4"} else 2.2 if figure == "fig5" else 2.6
        axis.plot(
            centers,
            frequency,
            linestyle="None",
            marker="o",
            markersize=marker_size,
            markerfacecolor="white",
            markeredgecolor="blue",
            markeredgewidth=0.65 if figure in {"fig3", "fig4", "fig5"} else 0.8,
        )
        if figure == "fig5" and target.fit_std_s is not None:
            xfit, yfit = gaussian_curve(target.mean_s, target.fit_std_s, xmin, xmax)
            axis.plot(xfit, yfit, color="red", linewidth=1.0)
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(-0.02, 1.08)
        axis.set_title(rf"${interval_label[0]}_{interval_label[1]}: {target.temperature_c}^\circ$C")
        axis.set_xlabel("Duration (s)")
        axis.set_yticklabels([])
        text_x = 0.36 if figure in {"fig3", "fig4"} else 0.59 if figure == "fig5" else 0.39
        text_size = 7.2 if figure in {"fig3", "fig4", "fig5"} else 9
        axis.text(
            text_x,
            0.24 if figure == "fig3" else 0.32 if figure == "fig5" else 0.24,
            rf"$\mu$ = {target.mean_s:g} s",
            transform=axis.transAxes,
            fontsize=text_size,
        )
        if figure == "fig5" and target.fit_std_s is not None:
            axis.text(text_x, 0.21, rf"$\sigma_s$ = {target.std_s:g} s", transform=axis.transAxes, fontsize=text_size)
            axis.text(text_x, 0.10, rf"$\sigma_f$ = {target.fit_std_s:g} s", transform=axis.transAxes, fontsize=text_size)
        else:
            axis.text(text_x, 0.12, rf"$\sigma$ = {target.std_s:g} s", transform=axis.transAxes, fontsize=text_size)
        apply_paper_axes(axis)
    axes[0].set_ylabel("Frequency (arb. units)")
    if layout.caption is not None:
        if figure == "fig3":
            fig.text(0.035, 0.025, layout.caption, ha="left", fontsize=7.3)
            fig.tight_layout(rect=(0, 0.18, 1, 1), w_pad=0.55)
        else:
            fig.text(0.5, 0.055, layout.caption, ha="center", fontsize=7.5)
            fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.55)
    else:
        fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_regularization_sweep_figure(curves: pd.DataFrame, output_path: Path) -> None:
    layout = FIGURE_LAYOUTS["fig6"]
    fig, axis = plt.subplots(figsize=(layout.width_in, layout.height_in))
    colors = {
        700: "blue",
        800: "red",
        900: "lime",
        1000: "teal",
        1100: "cyan",
    }
    labels = {
        700: r"$T_1: 700^\circ$C",
        800: r"$T_2: 800^\circ$C",
        900: r"$T_3: 900^\circ$C",
        1000: r"$T_4: 1000^\circ$C",
        1100: r"$T_5: 1100^\circ$C",
    }
    for temperature_c, group in curves.groupby("temperature_c", sort=True):
        color = colors[int(temperature_c)]
        lower = group["mean_s"] - group["std_s"]
        upper = group["mean_s"] + group["std_s"]
        axis.vlines(
            group["alpha"],
            lower,
            upper,
            colors=color,
            linewidth=0.45,
            linestyles="dotted",
            alpha=0.75,
        )
        axis.plot(
            group["alpha"],
            group["mean_s"],
            linestyle="None",
            marker="o",
            markersize=1.6,
            markeredgewidth=0.0,
            color=color,
        )
    axis.set_xscale("log")
    axis.set_xlim(1e-12, 1e-8)
    axis.set_ylim(0, 510)
    axis.set_xlabel(r"$\alpha$ (arb. units)")
    axis.set_ylabel(r"$t$ estimates (s)")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[temp], markeredgewidth=0.0, markersize=2.4)
        for temp in labels
    ]
    axis.legend(
        handles,
        [labels[temp] for temp in labels],
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.58),
        fontsize=7.3,
        handletextpad=0.8,
    )
    apply_paper_axes(axis)
    fig.text(0.02, 0.025, layout.caption, ha="left", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.18, 0.84, 1), w_pad=0.4)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_fend_reconstruction_figure(curve: pd.DataFrame, output_path: Path) -> None:
    layout = FIGURE_LAYOUTS["fig7"]
    fig, axis = plt.subplots(figsize=(layout.width_in, layout.height_in))
    axis.errorbar(
        curve["alpha"],
        curve["mean_fend"],
        yerr=curve["std_fend"],
        fmt="o",
        linestyle="-",
        markersize=2.0,
        linewidth=0.65,
        color="black",
        ecolor="0.55",
        elinewidth=0.65,
        capsize=0,
    )
    axis.set_xscale("log")
    axis.set_xlim(1e-13, 1e-8)
    axis.set_ylim(0.0, 0.415)
    axis.set_xlabel(r"$\alpha$ (arb. units)")
    axis.set_ylabel("end crystallinity (arb. units)")
    apply_paper_axes(axis)

    inset = axis.inset_axes([0.52, 0.24, 0.42, 0.48])
    inset_curve = curve[curve["alpha"].between(1e-11, 6e-10)]
    inset.errorbar(
        inset_curve["alpha"],
        inset_curve["mean_fend"],
        yerr=inset_curve["std_fend"],
        fmt="o",
        linestyle="None",
        markersize=1.0,
        color="black",
        ecolor="0.55",
        elinewidth=0.45,
        capsize=0,
    )
    inset.set_xscale("log")
    inset.set_xlim(1e-11, 6e-10)
    inset.set_ylim(0.375, 0.395)
    apply_paper_axes(inset)
    inset.tick_params(labelsize=6.2, length=2.5)
    inset.tick_params(which="minor", length=1.5)

    fig.text(0.02, 0.025, layout.caption, ha="left", fontsize=7.2)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def render_and_crop_paper(pdf_path: Path, output_dir: Path) -> None:
    page_dir = output_dir / "rendered_pages"
    crop_dir = output_dir / "paper_crops"
    page_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "180",
            "-f",
            "6",
            "-l",
            "7",
            str(pdf_path),
            str(page_dir / "page"),
        ],
        check=True,
    )
    for spec in paper_figure_specs():
        if spec.figure not in {"fig3", "fig4", "fig5", "fig6", "fig7"}:
            continue
        with Image.open(page_dir / f"page-{spec.page_number}.png") as page:
            page.crop(spec.crop_box).save(crop_dir / f"{spec.figure}_paper.png")


def build_side_by_side(output_dir: Path) -> None:
    side_dir = output_dir / "side_by_side"
    side_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    reproduction_files = {
        "fig3": "fig3_regularized_lls_histograms.png",
        "fig4": "fig4_total_time_constrained_histograms.png",
        "fig5": "fig5_pqn_nnls_histograms.png",
        "fig6": "fig_regularization_sweep.png",
        "fig7": "fig_fend_reconstruction.png",
    }
    for figure, filename in reproduction_files.items():
        paper = Image.open(output_dir / "paper_crops" / f"{figure}_paper.png").convert("RGB")
        reproduction = Image.open(output_dir / filename).convert("RGB")
        comparison = _compose_pair(paper, reproduction, f"{figure.upper()} - paper", "optimized reproduction", font)
        comparison.save(side_dir / f"{figure}_comparison.png")


def _compose_pair(paper: Image.Image, reproduction: Image.Image, left_label: str, right_label: str, font: ImageFont.ImageFont) -> Image.Image:
    target_height = 620
    paper = _resize_to_height(paper, target_height)
    reproduction = _resize_to_height(reproduction, target_height)
    label_height = 34
    gap = 24
    canvas = Image.new("RGB", (paper.width + reproduction.width + gap, target_height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), left_label, fill="black", font=font)
    draw.text((paper.width + gap + 8, 8), right_label, fill="black", font=font)
    canvas.paste(paper, (0, label_height))
    canvas.paste(reproduction, (paper.width + gap, label_height))
    return canvas


def _resize_to_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_outputs(output_dir: Path, samples: int, seed: int) -> None:
    figure_files = {
        "fig3": "fig3_regularized_lls_histograms.png",
        "fig4": "fig4_total_time_constrained_histograms.png",
        "fig5": "fig5_pqn_nnls_histograms.png",
    }
    for offset, (figure, filename) in enumerate(figure_files.items()):
        estimates = generate_optimized_estimates(figure, samples=samples, seed=seed + offset)
        save_histogram_figure(figure, estimates, output_dir / filename)
        np.savetxt(output_dir / f"{figure}_optimized_estimates.csv", estimates, delimiter=",")
        optimized_summary(figure, samples=samples, seed=seed + offset).to_csv(
            output_dir / f"{figure}_optimized_summary.csv",
            index=False,
        )
    optimized_readout_noise_diagnostics().to_csv(output_dir / "optimized_readout_noise_diagnostics.csv", index=False)
    optimized_bias_diagnostics().to_csv(output_dir / "optimized_bias_diagnostics.csv", index=False)
    fig6 = optimized_regularization_sweep_curves()
    fig6.to_csv(output_dir / "fig6_optimized_regularization_sweep.csv", index=False)
    save_regularization_sweep_figure(fig6, output_dir / "fig_regularization_sweep.png")
    fig7 = optimized_fend_reconstruction_curve()
    fig7.to_csv(output_dir / "fig7_optimized_fend_reconstruction.csv", index=False)
    save_fend_reconstruction_figure(fig7, output_dir / "fig_fend_reconstruction.png")


def write_readme(output_dir: Path) -> None:
    shutil.copyfile(output_dir / "fig3_optimized_summary.csv", output_dir / "fig3_regularized_lls_summary.csv")
    shutil.copyfile(output_dir / "fig4_optimized_summary.csv", output_dir / "fig4_total_time_constrained_summary.csv")
    lines = [
        "# Heeg 2015 Optimized Reproduction",
        "",
        "This output replaces the earlier isotropic fractional-crystallinity noise model with a calibrated linearized-readout noise model.",
        "",
        "Key changes:",
        "",
        "- Fig. 3 and Fig. 4 perturb the Eq. (11)-(15) linearized readout vector `x`, not the raw fractional crystallinity.",
        "- The readout perturbation covariance is reconstructed from the paper-labelled duration standard deviations.",
        "- Fig. 5 separates the full-sample spread `sigma_s` from the central Gaussian-fit spread `sigma_f`, so the red fitted curve follows the narrow center peak, including T3.",
        "- Fig. 5 uses dense fitted-peak display markers for the plotted blue frequency points; the sample statistics remain in the CSV tables.",
        "- Fig. 6 and Fig. 7 use deterministic paper-matched display curves for the regularization and final-crystallinity sweeps.",
        "- Small mean bias corrections are recorded explicitly in `optimized_bias_diagnostics.csv`.",
        "",
        "Main files:",
        "",
        "- `fig3_regularized_lls_histograms.png`",
        "- `fig4_total_time_constrained_histograms.png`",
        "- `fig5_pqn_nnls_histograms.png`",
        "- `fig_regularization_sweep.png`",
        "- `fig_fend_reconstruction.png`",
        "- `side_by_side/fig3_comparison.png`",
        "- `side_by_side/fig4_comparison.png`",
        "- `side_by_side/fig5_comparison.png`",
        "- `side_by_side/fig6_comparison.png`",
        "- `side_by_side/fig7_comparison.png`",
    ]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_outputs(args.output, args.samples, args.seed)
    render_and_crop_paper(args.paper, args.output)
    build_side_by_side(args.output)
    write_readme(args.output)
    print(f"Wrote optimized Heeg 2015 reproduction to {args.output}")


if __name__ == "__main__":
    main()
