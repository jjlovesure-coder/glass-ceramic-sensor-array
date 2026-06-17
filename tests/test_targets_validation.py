from pathlib import Path

import pandas as pd

from thermal_history.targets import load_text_targets
from thermal_history.validation import validate_targets


def test_text_targets_include_core_paper_metrics():
    targets = load_text_targets(Path("data/paper_targets/heeg2015/text_targets.csv"))

    required = {
        ("fig3", "t5_mean_s"),
        ("fig3", "t5_std_s"),
        ("fig7", "fend_sensor21_exact"),
        ("fig8_spike_1100", "t1_mean_s"),
        ("fig8_spike_1100", "t3_std_s"),
    }

    assert required.issubset({(row.figure, row.metric) for row in targets})


def test_validation_reports_pass_fail_and_error_fraction():
    targets = load_text_targets(Path("data/paper_targets/heeg2015/text_targets.csv"))
    observations = pd.DataFrame(
        [
            {"figure": "fig3", "metric": "t5_mean_s", "observed": 93.0},
            {"figure": "fig3", "metric": "t5_std_s", "observed": 10.0},
        ]
    )

    report = validate_targets(targets, observations, max_relative_error=0.05)
    fig3_t5_mean = report[(report["figure"] == "fig3") & (report["metric"] == "t5_mean_s")].iloc[0]
    fig3_t5_std = report[(report["figure"] == "fig3") & (report["metric"] == "t5_std_s")].iloc[0]

    assert bool(fig3_t5_mean["passed"]) is True
    assert bool(fig3_t5_std["passed"]) is False
    assert fig3_t5_std["relative_error"] > 0.05
