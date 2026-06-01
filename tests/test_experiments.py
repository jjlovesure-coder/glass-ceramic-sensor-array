import numpy as np

from thermal_history.experiments import (
    main_reconstruction_summary,
    regularization_sweep,
    spike_reconstruction_summary,
)


def test_main_reconstruction_summary_contains_expected_intervals():
    summary = main_reconstruction_summary(samples=50, seed=11, method="nnls")

    assert list(summary["temperature_c"]) == [700, 800, 900, 1000, 1100]
    assert summary.loc[summary["temperature_c"] == 1100, "mean_s"].iloc[0] > 50.0
    assert summary.loc[summary["temperature_c"] == 1100, "mean_s"].iloc[0] < 150.0


def test_regularization_sweep_reports_sensor_21_end_crystallinity():
    sweep = regularization_sweep(samples=20, seed=12, alphas=np.array([1e-12, 1e-11]))

    assert set(["alpha", "temperature_c", "mean_s", "std_s", "sensor21_fend_mean"]).issubset(sweep.columns)
    assert np.all(sweep["sensor21_fend_mean"] > 0.2)
    assert np.all(sweep["sensor21_fend_mean"] < 0.8)


def test_spike_reconstruction_recovers_ten_second_1100c_spike():
    summary = spike_reconstruction_summary(
        samples=200,
        seed=13,
        spike_temperatures_c=np.array([1100.0]),
    )

    row = summary.iloc[0]

    assert row["spike_temperature_c"] == 1100.0
    assert 5.0 <= row["t3_mean_s"] <= 15.0
    assert 0.5 <= row["t3_std_s"] <= 8.0
