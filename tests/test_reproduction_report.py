from pathlib import Path

import pandas as pd
import pytest

from thermal_history.reproduction_report import (
    build_validation_report,
    model_observations,
    observations_from_output_directory,
    write_validation_outputs,
)


def test_observations_from_output_directory_maps_key_paper_metrics(tmp_path: Path):
    pd.DataFrame(
        [
            {"temperature_c": 1000.0, "mean_s": 260.0, "std_s": 20.0, "median_s": 261.0},
            {"temperature_c": 1100.0, "mean_s": 93.0, "std_s": 7.0, "median_s": 92.0},
        ]
    ).to_csv(tmp_path / "fig3_regularized_lls_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "spike_temperature_c": 1100.0,
                "t1_mean_s": 999.7,
                "t1_std_s": 12.0,
                "t2_mean_s": 590.4,
                "t2_std_s": 15.0,
                "t3_mean_s": 9.9,
                "t3_std_s": 3.5,
            }
        ]
    ).to_csv(tmp_path / "spike_reconstruction_summary.csv", index=False)
    pd.DataFrame(
        [{"method": "scipy_nnls", "sensor21_fend_mean": 0.385, "sensor21_fend_std": 0.003}]
    ).to_csv(tmp_path / "nnls_fend_summary.csv", index=False)

    observations = observations_from_output_directory(tmp_path)

    indexed = observations.set_index(["figure", "metric"])["observed"]
    assert indexed[("fig3", "t4_mean_s")] == 260.0
    assert indexed[("fig3", "t5_mean_s")] == 93.0
    assert indexed[("fig3", "t5_std_s")] == 7.0
    assert indexed[("fig7", "nnls_fend_mean")] == 0.385
    assert indexed[("fig7", "nnls_fend_std")] == 0.003
    assert indexed[("fig8_spike_1100", "t3_std_s")] == 3.5


def test_model_observations_include_noiseless_and_sensor21_targets():
    observations = model_observations()

    indexed = observations.set_index(["figure", "metric"])["observed"]
    assert indexed[("fig2", "t1_truth_s")] == 600.0
    assert indexed[("fig2", "t5_truth_s")] == 100.0
    assert indexed[("fig6", "noiseless_t1_s")] == pytest.approx(600.05, rel=5e-4)
    assert indexed[("fig6", "noiseless_t5_s")] == pytest.approx(100.0, rel=5e-4)
    assert indexed[("fig7", "fend_sensor21_exact")] == pytest.approx(0.386, rel=5e-3)


def test_write_validation_outputs_creates_machine_and_human_reports(tmp_path: Path):
    target_path = tmp_path / "targets.csv"
    pd.DataFrame(
        [
            {
                "figure": "fig3",
                "metric": "t5_mean_s",
                "target": 93.0,
                "unit": "s",
                "tolerance_relative": 0.05,
                "tolerance_absolute": 0.0,
                "source": "test",
                "note": "test target",
            }
        ]
    ).to_csv(target_path, index=False)
    pd.DataFrame(
        [{"temperature_c": 1100.0, "mean_s": 93.0, "std_s": 7.0, "median_s": 92.0}]
    ).to_csv(tmp_path / "fig3_regularized_lls_summary.csv", index=False)

    report = build_validation_report(tmp_path, target_path)
    write_validation_outputs(report, tmp_path / "validation.csv", tmp_path / "validation.md")

    assert report["passed"].all()
    assert (tmp_path / "validation.csv").exists()
    assert "Passed targets: 1 / 1" in (tmp_path / "validation.md").read_text(encoding="utf-8")
