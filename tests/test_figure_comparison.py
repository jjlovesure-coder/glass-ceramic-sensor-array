import pandas as pd

from thermal_history.figure_comparison import build_metric_comparison, paper_figure_specs


def test_paper_figure_specs_cover_result_figures_with_valid_crop_boxes():
    specs = paper_figure_specs()

    assert [spec.figure for spec in specs] == [f"fig{number}" for number in range(2, 10)]
    assert len({spec.reproduction_name for spec in specs}) == len(specs)
    for spec in specs:
        left, top, right, bottom = spec.crop_box
        assert right > left
        assert bottom > top
        assert spec.page_number >= 5


def test_build_metric_comparison_groups_failed_targets_by_figure():
    validation = pd.DataFrame(
        [
            {
                "figure": "fig3",
                "metric": "t5_mean_s",
                "target": 93.0,
                "observed": 92.4,
                "relative_error": 0.006,
                "passed": True,
            },
            {
                "figure": "fig3",
                "metric": "t5_std_s",
                "target": 7.0,
                "observed": 66.5,
                "relative_error": 8.5,
                "passed": False,
            },
        ]
    )

    comparison = build_metric_comparison(validation)

    assert list(comparison["figure"]) == ["fig3", "fig3"]
    assert list(comparison["status"]) == ["pass", "fail"]
    assert comparison.loc[1, "relative_error_percent"] == 850.0
