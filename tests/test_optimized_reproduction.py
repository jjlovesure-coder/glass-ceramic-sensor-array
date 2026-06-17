from __future__ import annotations

import numpy as np

from thermal_history.optimized_reproduction import (
    FIGURE_LAYOUTS,
    PAPER_HISTOGRAM_TARGETS,
    display_frequency_points,
    generate_optimized_estimates,
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
