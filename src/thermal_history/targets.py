from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class PaperTarget:
    figure: str
    metric: str
    target: float
    unit: str
    tolerance_relative: float
    tolerance_absolute: float
    source: str
    note: str


def load_text_targets(path: str | Path) -> list[PaperTarget]:
    frame = pd.read_csv(path)
    required = {
        "figure",
        "metric",
        "target",
        "unit",
        "tolerance_relative",
        "tolerance_absolute",
        "source",
        "note",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Target file is missing columns: {sorted(missing)}")

    return [
        PaperTarget(
            figure=str(row.figure),
            metric=str(row.metric),
            target=float(row.target),
            unit=str(row.unit),
            tolerance_relative=float(row.tolerance_relative),
            tolerance_absolute=float(row.tolerance_absolute),
            source=str(row.source),
            note=str(row.note),
        )
        for row in frame.itertuples(index=False)
    ]
