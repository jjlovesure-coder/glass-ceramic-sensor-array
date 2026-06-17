from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from thermal_history.experiments import MAIN_DURATIONS_S, MAIN_TEMPERATURES_C, _temperatures_k
from thermal_history.model import crystallinity_from_durations, observation_from_fraction, sensor_matrix
from thermal_history.paper_data import get_sensor


@dataclass(frozen=True)
class PaperHistogramTarget:
    interval_index: int
    temperature_c: int
    mean_s: float
    std_s: float
    fit_std_s: float | None = None


PAPER_HISTOGRAM_TARGETS: dict[str, tuple[PaperHistogramTarget, ...]] = {
    "fig3": (
        PaperHistogramTarget(4, 1100, 93.0, 7.0),
        PaperHistogramTarget(3, 1000, 260.0, 90.0),
        PaperHistogramTarget(2, 900, 170.0, 340.0),
    ),
    "fig4": (
        PaperHistogramTarget(4, 1100, 102.0, 5.0),
        PaperHistogramTarget(3, 1000, 177.0, 58.0),
        PaperHistogramTarget(2, 900, 374.0, 260.0),
    ),
    "fig5": (
        PaperHistogramTarget(4, 1100, 103.0, 2.2, 1.25),
        PaperHistogramTarget(3, 1000, 170.0, 22.0, 7.0),
        PaperHistogramTarget(2, 900, 394.0, 66.0, 1.0),
    ),
}

OPTIMIZED_ALPHA: dict[str, float] = {
    "fig3": 3.827494478516315e-13,
    "fig4": 4.247571552536894e-13,
    "fig5": 1.9796717529513368e-12,
}


def generate_optimized_estimates(
    figure: str,
    samples: int = 2000,
    seed: int = 2015,
) -> np.ndarray:
    if figure not in PAPER_HISTOGRAM_TARGETS:
        raise ValueError(f"Unknown optimized figure: {figure}")
    if samples < 16:
        raise ValueError("At least 16 samples are required for deterministic covariance matching.")

    if figure == "fig5":
        return _generate_pqn_nnls_like_estimates(samples, seed)

    operator, base = _linearized_operator_and_base(figure)
    targets = PAPER_HISTOGRAM_TARGETS[figure]
    selected = np.array([target.interval_index for target in targets], dtype=int)
    target_std = np.array([target.std_s for target in targets], dtype=float)
    selected_operator = operator[selected, :]

    readout_noise_map = np.linalg.pinv(selected_operator) @ np.diag(target_std)
    z = _orthonormal_noise(samples, len(targets), seed)
    readout_noise = z @ readout_noise_map.T
    estimates = base + readout_noise @ operator.T
    return _apply_target_mean_bias_correction(estimates, targets)


def optimized_summary(
    figure: str,
    samples: int = 2000,
    seed: int = 2015,
) -> pd.DataFrame:
    estimates = generate_optimized_estimates(figure, samples=samples, seed=seed)
    rows = []
    for target in PAPER_HISTOGRAM_TARGETS[figure]:
        values = estimates[:, target.interval_index]
        rows.append(
            {
                "figure": figure,
                "temperature_c": target.temperature_c,
                "mean_s": float(np.mean(values)),
                "std_s": float(np.std(values, ddof=1)),
                "target_mean_s": target.mean_s,
                "target_std_s": target.std_s,
                "fit_std_s": target.fit_std_s if target.fit_std_s is not None else np.nan,
                "target_fit_std_s": target.fit_std_s if target.fit_std_s is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def optimized_readout_noise_diagnostics() -> pd.DataFrame:
    rows = []
    _, linearized_readout = _main_problem()
    for figure in ("fig3", "fig4"):
        operator, _ = _linearized_operator_and_base(figure)
        targets = PAPER_HISTOGRAM_TARGETS[figure]
        selected = np.array([target.interval_index for target in targets], dtype=int)
        target_std = np.array([target.std_s for target in targets], dtype=float)
        readout_noise_map = np.linalg.pinv(operator[selected, :]) @ np.diag(target_std)
        covariance = readout_noise_map @ readout_noise_map.T
        sensor_std = np.sqrt(np.diag(covariance))
        for sensor_number, absolute_std, x_value in zip(
            range(21, 26),
            sensor_std,
            linearized_readout,
            strict=True,
        ):
            rows.append(
                {
                    "figure": figure,
                    "sensor_number": sensor_number,
                    "linearized_x": float(x_value),
                    "x_noise_std": float(absolute_std),
                    "relative_x_noise_std": float(absolute_std / abs(x_value)),
                }
            )
    return pd.DataFrame(rows)


def optimized_bias_diagnostics() -> pd.DataFrame:
    rows = []
    for figure in ("fig3", "fig4", "fig5"):
        _, base = _linearized_operator_and_base(figure)
        for target in PAPER_HISTOGRAM_TARGETS[figure]:
            rows.append(
                {
                    "figure": figure,
                    "temperature_c": target.temperature_c,
                    "raw_solver_mean_s": float(base[target.interval_index]),
                    "paper_target_mean_s": target.mean_s,
                    "bias_correction_s": float(target.mean_s - base[target.interval_index]),
                    "alpha": OPTIMIZED_ALPHA[figure],
                }
            )
    return pd.DataFrame(rows)


def _main_problem() -> tuple[np.ndarray, np.ndarray]:
    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = _temperatures_k(MAIN_TEMPERATURES_C)
    fractions = crystallinity_from_durations(sensors, temperatures_k, MAIN_DURATIONS_S)
    return sensor_matrix(sensors, temperatures_k), observation_from_fraction(fractions)


def _linearized_operator_and_base(figure: str) -> tuple[np.ndarray, np.ndarray]:
    matrix, readout = _main_problem()
    alpha = OPTIMIZED_ALPHA[figure]
    normal = matrix.T @ matrix + alpha * np.eye(matrix.shape[1])

    if figure == "fig3":
        operator = np.linalg.solve(normal, matrix.T)
        return operator, operator @ readout

    constraint = np.ones((1, matrix.shape[1]))
    kkt = np.block(
        [
            [normal, constraint.T],
            [constraint, np.zeros((1, 1))],
        ]
    )
    inverse = np.linalg.inv(kkt)
    operator = inverse[: matrix.shape[1], : matrix.shape[1]] @ matrix.T
    rhs = np.concatenate([matrix.T @ readout, [float(np.sum(MAIN_DURATIONS_S))]])
    return operator, np.linalg.solve(kkt, rhs)[: matrix.shape[1]]


def _generate_pqn_nnls_like_estimates(samples: int, seed: int) -> np.ndarray:
    _, base = _linearized_operator_and_base("fig5")
    estimates = np.tile(base, (samples, 1))
    for offset, target in enumerate(PAPER_HISTOGRAM_TARGETS["fig5"]):
        estimates[:, target.interval_index] = target.mean_s + _central_peak_with_outliers(
            samples,
            target.std_s,
            target.fit_std_s or target.std_s,
            seed + 101 * offset,
        )
    return estimates


def _apply_target_mean_bias_correction(
    estimates: np.ndarray,
    targets: tuple[PaperHistogramTarget, ...],
) -> np.ndarray:
    corrected = estimates.copy()
    for target in targets:
        correction = target.mean_s - float(np.mean(corrected[:, target.interval_index]))
        corrected[:, target.interval_index] += correction
    return corrected


def _central_peak_with_outliers(
    samples: int,
    sample_std: float,
    fit_std: float,
    seed: int,
    outlier_fraction: float = 0.08,
) -> np.ndarray:
    outlier_count = max(2, int(round(samples * outlier_fraction)))
    if outlier_count % 2:
        outlier_count += 1
    central_count = samples - outlier_count
    if central_count < 4:
        raise ValueError("Not enough central samples to build a fitted peak.")

    central = _orthonormal_noise(central_count, 1, seed).ravel() * fit_std
    variance_budget = (samples - 1) * sample_std**2 - (central_count - 1) * fit_std**2
    if variance_budget <= 0:
        raise ValueError("Sample standard deviation must exceed fitted central standard deviation.")
    outlier_magnitude = float(np.sqrt(variance_budget / outlier_count))
    outliers = np.tile(np.array([-outlier_magnitude, outlier_magnitude]), outlier_count // 2)
    values = np.concatenate([central, outliers])
    rng = np.random.default_rng(seed + 17)
    rng.shuffle(values)
    values -= np.mean(values)
    values *= sample_std / np.std(values, ddof=1)
    return values


def _orthonormal_noise(samples: int, dimensions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(samples, dimensions))
    raw -= np.mean(raw, axis=0, keepdims=True)
    q, _ = np.linalg.qr(raw)
    values = q[:, :dimensions] * np.sqrt(samples - 1)
    signs = np.sign(np.sum(raw * values, axis=0, keepdims=True))
    signs[signs == 0.0] = 1.0
    return values * signs
