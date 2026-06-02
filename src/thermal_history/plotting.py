from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_history.experiments import MAIN_DURATIONS_S, MAIN_TEMPERATURES_C


def save_main_histograms(
    estimates: np.ndarray,
    output_path: Path,
    title: str = "Heeg 2015 main reconstruction histograms",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    selected = [4, 3, 2]
    for axis, index in zip(axes, selected, strict=True):
        axis.hist(estimates[:, index], bins=30, color="#4c78a8", edgecolor="white")
        axis.axvline(MAIN_DURATIONS_S[index], color="#d62728", linewidth=1.5)
        axis.set_title(f"{MAIN_TEMPERATURES_C[index]:.0f} C")
        axis.set_xlabel("Estimated duration (s)")
        axis.set_ylabel("Count")
    fig.suptitle(title)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_regularization_sweep(sweep: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for temperature_c, group in sweep.groupby("temperature_c"):
        if temperature_c in {900.0, 1000.0, 1100.0}:
            axis.errorbar(
                group["alpha"],
                group["mean_s"],
                yerr=group["std_s"],
                marker="o",
                linewidth=1,
                capsize=2,
                label=f"{temperature_c:.0f} C",
            )
    axis.set_xscale("log")
    axis.set_xlabel("Regularization alpha")
    axis.set_ylabel("Estimated duration (s)")
    axis.legend()
    axis.set_title("Regularization sensitivity")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_fend_reconstruction(sweep: pd.DataFrame, output_path: Path) -> None:
    reduced = sweep.drop_duplicates("alpha").sort_values("alpha")
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.errorbar(
        reduced["alpha"],
        reduced["sensor21_fend_mean"],
        yerr=reduced["sensor21_fend_std"],
        marker="o",
        linewidth=1,
        capsize=2,
        color="#54a24b",
    )
    axis.axhline(0.386, color="#d62728", linewidth=1.5, label="Paper exact Fend")
    axis.set_xscale("log")
    axis.set_xlabel("Regularization alpha")
    axis.set_ylabel("Sensor 21 Fend")
    axis.set_title("End crystallinity from reconstructed histories")
    axis.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_spike_reconstruction(spike_summary: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.errorbar(
        spike_summary["spike_temperature_c"],
        spike_summary["t3_mean_s"],
        yerr=spike_summary["t3_std_s"],
        marker="o",
        linewidth=1,
        capsize=2,
        color="#f58518",
    )
    axis.axhline(10, color="#d62728", linewidth=1.5, label="True 10 s at 1100 C")
    axis.set_xlabel("Actual spike temperature (C)")
    axis.set_ylabel("Estimated 1100 C duration (s)")
    axis.set_title("Short spike reconstruction")
    axis.legend()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
