from __future__ import annotations

from pathlib import Path

import pandas as pd

from thermal_history.experiments import MAIN_DURATIONS_S, MAIN_TEMPERATURES_C
from thermal_history.model import crystallinity_from_durations
from thermal_history.paper_data import CELSIUS_TO_KELVIN, get_sensor
from thermal_history.solvers import solve_lls
from thermal_history.targets import load_text_targets
from thermal_history.validation import validate_targets


def observations_from_output_directory(output_dir: str | Path) -> pd.DataFrame:
    output_path = Path(output_dir)
    rows: list[dict[str, float | str]] = []

    fig3_path = output_path / "fig3_regularized_lls_summary.csv"
    if fig3_path.exists():
        fig3 = pd.read_csv(fig3_path)
        rows.extend(_duration_summary_rows(fig3, "fig3"))

    fig4_path = output_path / "fig4_total_time_constrained_summary.csv"
    if fig4_path.exists():
        fig4 = pd.read_csv(fig4_path)
        rows.extend(_duration_summary_rows(fig4, "fig4"))

    spike_path = output_path / "spike_reconstruction_summary.csv"
    if spike_path.exists():
        spike = pd.read_csv(spike_path)
        rows.extend(_spike_rows(spike))

    nnls_fend_path = output_path / "nnls_fend_summary.csv"
    if nnls_fend_path.exists():
        nnls_fend = pd.read_csv(nnls_fend_path)
        rows.extend(_nnls_fend_rows(nnls_fend))

    return pd.DataFrame(rows, columns=["figure", "metric", "observed"])


def model_observations() -> pd.DataFrame:
    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = MAIN_TEMPERATURES_C + CELSIUS_TO_KELVIN
    fractions = crystallinity_from_durations(sensors, temperatures_k, MAIN_DURATIONS_S)
    noiseless = solve_lls(sensors, temperatures_k, fractions)
    sensor21_fend = crystallinity_from_durations(
        [get_sensor(21)],
        temperatures_k,
        MAIN_DURATIONS_S,
    )[0]

    rows: list[dict[str, float | str]] = []
    for index, value in enumerate(MAIN_DURATIONS_S, start=1):
        rows.append(
            {
                "figure": "fig2",
                "metric": f"t{index}_truth_s",
                "observed": float(value),
            }
        )
    for index, value in enumerate(noiseless, start=1):
        rows.append(
            {
                "figure": "fig6",
                "metric": f"noiseless_t{index}_s",
                "observed": float(value),
            }
        )
    rows.append(
        {
            "figure": "fig7",
            "metric": "fend_sensor21_exact",
            "observed": float(sensor21_fend),
        }
    )
    return pd.DataFrame(rows, columns=["figure", "metric", "observed"])


def combined_observations(output_dir: str | Path) -> pd.DataFrame:
    return pd.concat(
        [observations_from_output_directory(output_dir), model_observations()],
        ignore_index=True,
    )


def build_validation_report(
    output_dir: str | Path,
    target_path: str | Path,
    max_relative_error: float = 0.05,
) -> pd.DataFrame:
    targets = load_text_targets(target_path)
    observations = combined_observations(output_dir)
    return validate_targets(targets, observations, max_relative_error=max_relative_error)


def write_validation_outputs(
    report: pd.DataFrame,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    csv_target = Path(csv_path)
    markdown_target = Path(markdown_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)

    report.to_csv(csv_target, index=False)
    markdown_target.write_text(_validation_markdown(report), encoding="utf-8")


def _validation_markdown(report: pd.DataFrame) -> str:
    total = len(report)
    passed = int(report["passed"].sum()) if total else 0
    failed = report[~report["passed"]]
    lines = [
        "# Heeg 2015 Target Validation",
        "",
        f"Passed targets: {passed} / {total}",
        "",
    ]
    if failed.empty:
        lines.append("All available targets are within tolerance.")
    else:
        lines.extend(["Failed targets:", ""])
        for row in failed.itertuples(index=False):
            observed = "missing" if pd.isna(row.observed) else f"{row.observed:.6g}"
            lines.append(
                "- "
                f"{row.figure}.{row.metric}: observed={observed}, "
                f"target={row.target:.6g} {row.unit}, "
                f"relative_error={row.relative_error:.3g}"
            )
    return "\n".join(lines) + "\n"


def _duration_summary_rows(frame: pd.DataFrame, figure: str) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for row in frame.itertuples(index=False):
        interval_index = int(round((float(row.temperature_c) - 600.0) / 100.0))
        rows.append({"figure": figure, "metric": f"t{interval_index}_mean_s", "observed": row.mean_s})
        rows.append({"figure": figure, "metric": f"t{interval_index}_std_s", "observed": row.std_s})
    return rows


def _nnls_fend_rows(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    if frame.empty:
        return []
    row = frame.iloc[0]
    return [
        {
            "figure": "fig7",
            "metric": "nnls_fend_mean",
            "observed": float(row["sensor21_fend_mean"]),
        },
        {
            "figure": "fig7",
            "metric": "nnls_fend_std",
            "observed": float(row["sensor21_fend_std"]),
        },
    ]


def _spike_rows(frame: pd.DataFrame) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    selected = frame[frame["spike_temperature_c"].round(6) == 1100.0]
    if selected.empty:
        return rows

    row = selected.iloc[0]
    for interval_index in range(1, 4):
        rows.append(
            {
                "figure": "fig8_spike_1100",
                "metric": f"t{interval_index}_mean_s",
                "observed": float(row[f"t{interval_index}_mean_s"]),
            }
        )
        rows.append(
            {
                "figure": "fig8_spike_1100",
                "metric": f"t{interval_index}_std_s",
                "observed": float(row[f"t{interval_index}_std_s"]),
            }
        )
    return rows
