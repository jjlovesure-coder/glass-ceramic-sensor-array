from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from thermal_history.targets import PaperTarget


def validate_targets(
    targets: Iterable[PaperTarget],
    observations: pd.DataFrame,
    max_relative_error: float = 0.05,
) -> pd.DataFrame:
    required = {"figure", "metric", "observed"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Observations are missing columns: {sorted(missing)}")

    target_frame = pd.DataFrame([asdict(target) for target in targets])
    observed_frame = observations.copy()
    report = target_frame.merge(observed_frame, on=["figure", "metric"], how="left")
    report["abs_error"] = (report["observed"] - report["target"]).abs()
    denominator = report["target"].abs().replace(0.0, np.nan)
    report["relative_error"] = report["abs_error"] / denominator
    relative_limit = np.minimum(report["tolerance_relative"], max_relative_error)
    report["allowed_error"] = np.maximum(
        report["target"].abs() * relative_limit,
        report["tolerance_absolute"],
    )
    report["passed"] = report["observed"].notna() & (report["abs_error"] <= report["allowed_error"])
    return report
