import numpy as np

from thermal_history.experiments import (
    fig3_fig4_histogram_outputs,
    main_reconstruction_summary,
    noise_model_comparison,
    numerical_diagnostics,
    regularization_sweep,
    spike_reconstruction_summary,
)


def test_main_reconstruction_summary_contains_expected_intervals():
    summary = main_reconstruction_summary(samples=50, seed=11, method="nnls")

    assert list(summary["temperature_c"]) == [700, 800, 900, 1000, 1100]
    assert summary.loc[summary["temperature_c"] == 1100, "mean_s"].iloc[0] > 50.0
    assert summary.loc[summary["temperature_c"] == 1100, "mean_s"].iloc[0] < 150.0


def test_fig3_and_fig4_histogram_outputs_are_distinct_reconstruction_methods():
    outputs = fig3_fig4_histogram_outputs(samples=50, seed=21)
    fig3_mean = outputs["fig3_regularized_lls"].estimates.mean(axis=0)
    fig4_estimates = outputs["fig4_total_time_constrained"].estimates
    fig4_mean = fig4_estimates.mean(axis=0)

    assert set(outputs) == {"fig3_regularized_lls", "fig4_total_time_constrained"}
    assert outputs["fig3_regularized_lls"].estimates.shape == (50, 5)
    assert fig4_estimates.shape == (50, 5)
    assert np.any(outputs["fig3_regularized_lls"].estimates < 0.0)
    assert np.any(fig4_estimates < 0.0)
    np.testing.assert_allclose(
        fig4_estimates.sum(axis=1),
        1600.0,
        atol=1e-8,
    )
    assert abs(fig4_mean[3] - 200.0) < abs(fig3_mean[3] - 200.0)
    assert 70.0 <= fig4_mean[4] <= 120.0


def test_regularization_sweep_reports_sensor_21_end_crystallinity():
    sweep = regularization_sweep(samples=20, seed=12, alphas=np.array([1e-12, 1e-11]))

    assert set(["alpha", "temperature_c", "mean_s", "std_s", "sensor21_fend_mean"]).issubset(sweep.columns)
    assert np.all(sweep["sensor21_fend_mean"] > 0.2)
    assert np.all(sweep["sensor21_fend_mean"] < 0.8)


def test_numerical_diagnostics_record_solver_and_noise_assumptions():
    diagnostics = numerical_diagnostics()

    assert diagnostics["alpha"] == 1e-11
    assert diagnostics["noise_placement"] == "fractional_crystallinity"
    assert diagnostics["fig4_allows_negative_durations"] is True
    assert diagnostics["condition_number"] > 1e6
    assert diagnostics["noiseless_lls_error_norm_s"] < 1e-3
    assert diagnostics["noiseless_total_time_alpha0_error_norm_s"] < 1e-3


def test_noise_model_comparison_reports_fraction_and_linearized_noise():
    comparison = noise_model_comparison(samples=20, seed=31)

    assert set(comparison["noise_placement"]) == {"fractional_crystallinity", "linearized_observation"}
    assert set(["fig4_t3_mean_s", "fig4_t3_std_s", "fig4_t4_mean_s", "fig4_t4_std_s"]).issubset(
        comparison.columns
    )


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
