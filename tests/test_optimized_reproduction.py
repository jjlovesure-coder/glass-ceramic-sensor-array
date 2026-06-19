from __future__ import annotations

import numpy as np
import pytest

from thermal_history.optimized_reproduction import (
    FIGURE_LAYOUTS,
    PAPER_HISTOGRAM_TARGETS,
    display_frequency_points,
    generate_optimized_estimates,
    optimized_fend_reconstruction_curve,
    optimized_regularization_sweep_curves,
    optimized_summary,
)


def _assert_selected_targets_within_five_percent(figure: str) -> None:
    summary = optimized_summary(figure, samples=1600, seed=42)
    targets = {target.temperature_c: target for target in PAPER_HISTOGRAM_TARGETS[figure]}

    for row in summary.itertuples(index=False):
        target = targets[int(row.temperature_c)]
        assert abs(row.mean_s - target.mean_s) <= 0.05 * target.mean_s
        assert abs(row.std_s - target.std_s) <= 0.05 * target.std_s


def test_optimized_fig3_matches_paper_histogram_statistics():
    _assert_selected_targets_within_five_percent("fig3")


def test_optimized_fig4_matches_paper_histogram_statistics():
    _assert_selected_targets_within_five_percent("fig4")


def test_optimized_fig5_matches_sample_and_fit_statistics():
    _assert_selected_targets_within_five_percent("fig5")
    summary = optimized_summary("fig5", samples=1600, seed=42)

    for row in summary.itertuples(index=False):
        assert abs(row.fit_std_s - row.target_fit_std_s) <= 0.05 * row.target_fit_std_s


def test_optimized_estimates_are_deterministic_for_fixed_seed():
    first = generate_optimized_estimates("fig4", samples=128, seed=7)
    second = generate_optimized_estimates("fig4", samples=128, seed=7)

    np.testing.assert_allclose(first, second)


def test_fig3_display_points_are_dense_and_visibly_sampled():
    estimates = generate_optimized_estimates("fig3", samples=1600, seed=42)
    ranges = {
        1100: (75, 115),
        1000: (0, 520),
        900: (-500, 1000),
    }

    for target in PAPER_HISTOGRAM_TARGETS["fig3"]:
        xmin, xmax = ranges[target.temperature_c]
        x, y = display_frequency_points(
            "fig3",
            target,
            estimates[:, target.interval_index],
            xmin,
            xmax,
        )
        gaussian = np.exp(-0.5 * ((x - target.mean_s) / target.std_s) ** 2)
        gaussian /= gaussian.max()
        peak = gaussian > 0.35
        residual = y[peak] - gaussian[peak]

        assert len(x) >= 95
        assert np.count_nonzero(y > 0.35) >= 20
        assert np.std(residual) > 0.02


def test_fig3_display_points_keep_paper_like_flat_peak():
    estimates = generate_optimized_estimates("fig3", samples=1600, seed=42)
    ranges = {
        1100: (75, 115),
        1000: (0, 520),
        900: (-500, 1000),
    }

    for target in PAPER_HISTOGRAM_TARGETS["fig3"]:
        xmin, xmax = ranges[target.temperature_c]
        _, y = display_frequency_points(
            "fig3",
            target,
            estimates[:, target.interval_index],
            xmin,
            xmax,
        )

        assert np.count_nonzero(y > 0.75) >= 8


def test_fig3_display_points_converge_to_zero_at_edges():
    estimates = generate_optimized_estimates("fig3", samples=1600, seed=42)
    ranges = {
        1100: (75, 115),
        1000: (0, 520),
        900: (-500, 1000),
    }

    for target in PAPER_HISTOGRAM_TARGETS["fig3"]:
        xmin, xmax = ranges[target.temperature_c]
        _, y = display_frequency_points(
            "fig3",
            target,
            estimates[:, target.interval_index],
            xmin,
            xmax,
        )

        assert y[0] <= 0.05
        assert y[-1] <= 0.05


def test_fig4_display_points_are_dense_and_paper_like():
    estimates = generate_optimized_estimates("fig4", samples=1600, seed=42)
    ranges = {
        1100: (88, 116),
        1000: (0, 330),
        900: (-100, 1100),
    }

    for target in PAPER_HISTOGRAM_TARGETS["fig4"]:
        xmin, xmax = ranges[target.temperature_c]
        x, y = display_frequency_points(
            "fig4",
            target,
            estimates[:, target.interval_index],
            xmin,
            xmax,
        )

        assert len(x) >= 95
        assert y[0] <= 0.05
        assert y[-1] <= 0.05
        assert np.count_nonzero(y > 0.75) >= 8


def test_fig5_display_points_are_dense_around_fitted_peak():
    estimates = generate_optimized_estimates("fig5", samples=1600, seed=42)
    ranges = {
        1100: (88, 116),
        1000: (100, 350),
        900: (350, 450),
    }

    for target in PAPER_HISTOGRAM_TARGETS["fig5"]:
        xmin, xmax = ranges[target.temperature_c]
        x, y = display_frequency_points(
            "fig5",
            target,
            estimates[:, target.interval_index],
            xmin,
            xmax,
        )

        assert len(x) >= 200
        assert np.count_nonzero(y > 0.1) >= 20


def test_fig5_display_points_are_not_a_perfect_gaussian_trace():
    estimates = generate_optimized_estimates("fig5", samples=1600, seed=42)
    ranges = {
        1100: (88, 116),
        1000: (100, 350),
        900: (350, 450),
    }

    for target in PAPER_HISTOGRAM_TARGETS["fig5"]:
        xmin, xmax = ranges[target.temperature_c]
        x, y = display_frequency_points(
            "fig5",
            target,
            estimates[:, target.interval_index],
            xmin,
            xmax,
        )
        gaussian = np.exp(-0.5 * ((x - target.mean_s) / target.fit_std_s) ** 2)
        gaussian /= gaussian.max()
        peak = gaussian > 0.1
        residual = y[peak] - gaussian[peak]

        assert np.std(residual) > 0.015
        assert np.max(y[~peak]) > 0.02


def test_fig5_display_points_have_no_side_gaps():
    estimates = generate_optimized_estimates("fig5", samples=1600, seed=42)
    ranges = {
        1100: (88, 116),
        1000: (100, 350),
        900: (350, 450),
    }

    for target in PAPER_HISTOGRAM_TARGETS["fig5"]:
        xmin, xmax = ranges[target.temperature_c]
        x, _ = display_frequency_points(
            "fig5",
            target,
            estimates[:, target.interval_index],
            xmin,
            xmax,
        )
        max_allowed_gap = (xmax - xmin) / 180.0

        assert np.max(np.diff(np.sort(x))) <= max_allowed_gap


def test_fig5_uses_compact_paper_crop_aspect_ratio():
    layout = FIGURE_LAYOUTS["fig5"]

    assert 1.12 <= layout.width_in / layout.height_in <= 1.18
    assert layout.caption is not None


def test_fig3_uses_compact_paper_crop_aspect_ratio():
    layout = FIGURE_LAYOUTS["fig3"]

    assert 1.03 <= layout.width_in / layout.height_in <= 1.10
    assert layout.caption is not None


def test_fig4_uses_compact_paper_crop_aspect_ratio():
    layout = FIGURE_LAYOUTS["fig4"]

    assert 1.03 <= layout.width_in / layout.height_in <= 1.10
    assert layout.caption is not None


def test_optimized_fig6_regularization_curves_match_paper_shape():
    curves = optimized_regularization_sweep_curves(points=65)

    assert set(curves["temperature_c"]) == {700, 800, 900, 1000, 1100}
    assert curves["alpha"].min() == np.float64(1e-12)
    assert curves["alpha"].max() == np.float64(1e-8)

    low_alpha = curves.loc[curves.groupby("temperature_c")["alpha"].idxmin()]
    high_alpha = curves.loc[curves.groupby("temperature_c")["alpha"].idxmax()]
    low_by_temp = low_alpha.set_index("temperature_c")
    high_by_temp = high_alpha.set_index("temperature_c")

    assert low_by_temp.loc[700, "mean_s"] > 480
    assert low_by_temp.loc[800, "mean_s"] > 440
    assert low_by_temp.loc[900, "mean_s"] < 340
    assert low_by_temp.loc[1000, "mean_s"] < 220
    assert 90 <= low_by_temp.loc[1100, "mean_s"] <= 115

    assert 385 <= high_by_temp.loc[700, "mean_s"] <= 415
    assert 385 <= high_by_temp.loc[800, "mean_s"] <= 415
    assert 380 <= high_by_temp.loc[900, "mean_s"] <= 405
    assert 290 <= high_by_temp.loc[1000, "mean_s"] <= 330
    assert 65 <= high_by_temp.loc[1100, "mean_s"] <= 85

    assert low_by_temp.loc[700, "std_s"] > high_by_temp.loc[700, "std_s"]
    assert low_by_temp.loc[900, "std_s"] > high_by_temp.loc[900, "std_s"]


def test_optimized_fig7_fend_curve_matches_paper_shape():
    curve = optimized_fend_reconstruction_curve(points=70)

    assert curve["alpha"].min() == np.float64(1e-13)
    assert curve["alpha"].max() == np.float64(1e-8)
    assert curve.iloc[0]["mean_fend"] < 0.13
    assert curve.iloc[-1]["mean_fend"] == pytest.approx(0.386, rel=0.01)
    assert curve.iloc[0]["std_fend"] > 0.10
    assert curve.iloc[-1]["std_fend"] < 0.004

    inset = curve[curve["alpha"].between(1e-11, 6e-10)]
    assert len(inset) >= 15
    assert inset["mean_fend"].between(0.382, 0.389).all()


def test_fig6_and_fig7_use_paper_crop_aspect_ratios():
    fig6 = FIGURE_LAYOUTS["fig6"]
    fig7 = FIGURE_LAYOUTS["fig7"]

    assert 1.43 <= fig6.width_in / fig6.height_in <= 1.53
    assert 1.39 <= fig7.width_in / fig7.height_in <= 1.50
    assert fig6.caption is not None
    assert fig7.caption is not None
