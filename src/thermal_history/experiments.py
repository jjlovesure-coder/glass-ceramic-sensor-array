from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from thermal_history.model import crystallinity_from_durations
from thermal_history.paper_data import CELSIUS_TO_KELVIN, get_sensor
from thermal_history.solvers import (
    add_multiplicative_noise,
    solve_nnls,
    solve_tikhonov_total_time,
)


MAIN_TEMPERATURES_C = np.array([700, 800, 900, 1000, 1100], dtype=float)
MAIN_DURATIONS_S = np.array([600, 400, 300, 200, 100], dtype=float)
SPIKE_RECON_TEMPERATURES_C = np.array([900, 1000, 1100], dtype=float)
SPIKE_BASE_DURATIONS_S = np.array([1000, 590], dtype=float)


def _temperatures_k(temperatures_c: Sequence[float]) -> np.ndarray:
    return np.asarray(temperatures_c, dtype=float) + CELSIUS_TO_KELVIN


def _summarize_by_interval(
    temperatures_c: Sequence[float],
    estimates: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature_c": np.asarray(temperatures_c, dtype=float),
            "mean_s": np.mean(estimates, axis=0),
            "std_s": np.std(estimates, axis=0, ddof=1),
            "median_s": np.median(estimates, axis=0),
        }
    )


def _monte_carlo_estimates(
    fractions: np.ndarray,
    samples: int,
    seed: int,
    solve: Callable[[np.ndarray], np.ndarray],
    noise_fraction: float,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        noisy = add_multiplicative_noise(fractions, noise_fraction, rng)
        estimates.append(solve(noisy))
    return np.vstack(estimates)


def main_reconstruction_estimates(
    samples: int = 2000,
    seed: int = 2015,
    method: str = "nnls",
    noise_fraction: float = 0.05,
) -> np.ndarray:
    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = _temperatures_k(MAIN_TEMPERATURES_C)
    fractions = crystallinity_from_durations(sensors, temperatures_k, MAIN_DURATIONS_S)
    total_time = float(np.sum(MAIN_DURATIONS_S))

    if method == "nnls":
        solve = lambda noisy: solve_nnls(sensors, temperatures_k, noisy)
    elif method == "constrained_tikhonov":
        solve = lambda noisy: solve_tikhonov_total_time(
            sensors,
            temperatures_k,
            noisy,
            alpha=1e-11,
            total_time_s=total_time,
        )
    else:
        raise ValueError(f"Unknown main reconstruction method: {method}")

    return _monte_carlo_estimates(fractions, samples, seed, solve, noise_fraction)


def main_reconstruction_summary(
    samples: int = 2000,
    seed: int = 2015,
    method: str = "nnls",
    noise_fraction: float = 0.05,
) -> pd.DataFrame:
    estimates = main_reconstruction_estimates(samples, seed, method, noise_fraction)
    return _summarize_by_interval(MAIN_TEMPERATURES_C, estimates)


def regularization_sweep(
    samples: int = 400,
    seed: int = 2016,
    alphas: np.ndarray | None = None,
    noise_fraction: float = 0.05,
) -> pd.DataFrame:
    if alphas is None:
        alphas = np.logspace(-13, -9, 17)

    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = _temperatures_k(MAIN_TEMPERATURES_C)
    fractions = crystallinity_from_durations(sensors, temperatures_k, MAIN_DURATIONS_S)
    total_time = float(np.sum(MAIN_DURATIONS_S))
    sensor21 = [get_sensor(21)]
    rows = []

    for alpha in alphas:
        solve = lambda noisy, a=float(alpha): solve_tikhonov_total_time(
            sensors,
            temperatures_k,
            noisy,
            alpha=a,
            total_time_s=total_time,
        )
        estimates = _monte_carlo_estimates(fractions, samples, seed, solve, noise_fraction)
        fends = np.array(
            [
                crystallinity_from_durations(sensor21, temperatures_k, estimate)[0]
                for estimate in estimates
            ]
        )
        for interval_index, temperature_c in enumerate(MAIN_TEMPERATURES_C):
            rows.append(
                {
                    "alpha": float(alpha),
                    "temperature_c": float(temperature_c),
                    "mean_s": float(np.mean(estimates[:, interval_index])),
                    "std_s": float(np.std(estimates[:, interval_index], ddof=1)),
                    "sensor21_fend_mean": float(np.mean(fends)),
                    "sensor21_fend_std": float(np.std(fends, ddof=1)),
                }
            )

    return pd.DataFrame(rows)


def _spike_truth(spike_temperature_c: float) -> tuple[np.ndarray, np.ndarray]:
    temperatures_c = np.array([900.0, 1000.0, spike_temperature_c], dtype=float)
    durations_s = np.array([1000.0, 590.0, 10.0], dtype=float)
    return temperatures_c, durations_s


def spike_reconstruction_summary(
    samples: int = 2000,
    seed: int = 2017,
    spike_temperatures_c: np.ndarray | None = None,
    noise_fraction: float = 0.05,
) -> pd.DataFrame:
    if spike_temperatures_c is None:
        spike_temperatures_c = np.array([1050, 1075, 1100, 1125, 1150], dtype=float)

    sensors = [get_sensor(number) for number in range(1, 4)]
    recon_temperatures_k = _temperatures_k(SPIKE_RECON_TEMPERATURES_C)
    rows = []

    for spike_temperature_c in spike_temperatures_c:
        truth_temperatures_c, truth_durations_s = _spike_truth(float(spike_temperature_c))
        fractions = crystallinity_from_durations(
            sensors,
            _temperatures_k(truth_temperatures_c),
            truth_durations_s,
        )
        solve = lambda noisy: solve_tikhonov_total_time(
            sensors,
            recon_temperatures_k,
            noisy,
            alpha=1e-11,
            total_time_s=float(np.sum(truth_durations_s)),
        )
        estimates = _monte_carlo_estimates(fractions, samples, seed, solve, noise_fraction)
        rows.append(
            {
                "spike_temperature_c": float(spike_temperature_c),
                "t1_mean_s": float(np.mean(estimates[:, 0])),
                "t1_std_s": float(np.std(estimates[:, 0], ddof=1)),
                "t2_mean_s": float(np.mean(estimates[:, 1])),
                "t2_std_s": float(np.std(estimates[:, 1], ddof=1)),
                "t3_mean_s": float(np.mean(estimates[:, 2])),
                "t3_std_s": float(np.std(estimates[:, 2], ddof=1)),
            }
        )

    return pd.DataFrame(rows)
