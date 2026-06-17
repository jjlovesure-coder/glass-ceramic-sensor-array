from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PaperFigureSpec:
    figure: str
    page_number: int
    crop_box: tuple[int, int, int, int]
    reproduction_name: str
    title: str


def paper_figure_specs() -> list[PaperFigureSpec]:
    return [
        PaperFigureSpec("fig2", 5, (860, 1385, 1390, 1900), "fig2_temperature_profile.png", "Thermal history input"),
        PaperFigureSpec("fig3", 6, (105, 1325, 740, 1920), "fig3_regularized_lls_histograms.png", "Regularized LLS histograms"),
        PaperFigureSpec("fig4", 6, (830, 1390, 1400, 1930), "fig4_total_time_constrained_histograms.png", "Total-time constrained histograms"),
        PaperFigureSpec("fig5", 7, (125, 145, 740, 680), "fig5_nnls_histograms.png", "NNLS histograms"),
        PaperFigureSpec("fig6", 7, (120, 1490, 750, 1915), "fig_regularization_sweep.png", "Regularization sweep"),
        PaperFigureSpec("fig7", 7, (790, 1490, 1405, 1915), "fig_fend_reconstruction.png", "Sensor 21 end crystallinity"),
        PaperFigureSpec("fig8", 8, (90, 150, 760, 650), "fig8_spike_profile.png", "Spike input profile"),
        PaperFigureSpec("fig9", 8, (120, 1440, 745, 1920), "fig_spike_reconstruction.png", "Spike-temperature sensitivity"),
    ]


def build_metric_comparison(validation: pd.DataFrame) -> pd.DataFrame:
    required = {"figure", "metric", "target", "observed", "relative_error", "passed"}
    missing = required.difference(validation.columns)
    if missing:
        raise ValueError(f"Validation table is missing columns: {sorted(missing)}")

    comparison = validation.copy()
    comparison["status"] = comparison["passed"].map({True: "pass", False: "fail"})
    comparison["relative_error_percent"] = comparison["relative_error"] * 100.0
    return comparison[
        [
            "figure",
            "metric",
            "target",
            "observed",
            "relative_error_percent",
            "status",
        ]
    ]
