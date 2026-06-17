from __future__ import annotations

import numpy as np

from thermal_history.optimized_reproduction import (
    PAPER_HISTOGRAM_TARGETS,
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
