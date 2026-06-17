from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from thermal_history.model import (
    crystallinity_from_durations,
    fraction_from_observation,
    observation_from_fraction,
    sensor_matrix,
)
from thermal_history.paper_data import CELSIUS_TO_KELVIN, get_sensor
from thermal_history.solvers import (
    add_multiplicative_noise,
    solve_lls,
    solve_nnls,
    solve_tikhonov,
    solve_tikhonov_total_time,
)


MAIN_TEMPERATURES_C = np.array([700, 800, 900, 1000, 1100], dtype=float)
MAIN_DURATIONS_S = np.array([600, 400, 300, 200, 100], dtype=float)
SPIKE_RECON_TEMPERATURES_C = np.array([900, 1000, 1100], dtype=float)
SPIKE_BASE_DURATIONS_S = np.array([1000, 590], dtype=float)


@dataclass(frozen=True)
class HistogramReconstruction:
    name: str
    title: str
    estimates: np.ndarray
    summary: pd.DataFrame


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
    noise_placement: str = "fractional_crystallinity",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    observation = observation_from_fraction(fractions)
    estimates = []
    for _ in range(samples):
        if noise_placement == "fractional_crystallinity":
            noisy = add_multiplicative_noise(fractions, noise_fraction, rng)
        elif noise_placement == "linearized_observation":
            noisy_observation = observation * (
                1.0 + rng.normal(0.0, noise_fraction, size=observation.shape)
            )
            noisy = fraction_from_observation(noisy_observation)
        else:
            raise ValueError(f"Unknown noise placement: {noise_placement}")
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
    elif method == "regularized_tikhonov":
        solve = lambda noisy: solve_tikhonov(
            sensors,
            temperatures_k,
            noisy,
            alpha=1e-11,
        )
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


def fig3_fig4_histogram_outputs(
    samples: int = 2000,
    seed: int = 2015,
    noise_fraction: float = 0.05,
) -> dict[str, HistogramReconstruction]:
    fig3_estimates = main_reconstruction_estimates(
        samples=samples,
        seed=seed,
        method="regularized_tikhonov",
        noise_fraction=noise_fraction,
    )
    fig4_estimates = main_reconstruction_estimates(
        samples=samples,
        seed=seed,
        method="constrained_tikhonov",
        noise_fraction=noise_fraction,
    )
    return {
        "fig3_regularized_lls": HistogramReconstruction(
            name="fig3_regularized_lls",
            title="Fig. 3 reproduction: regularized LLS",
            estimates=fig3_estimates,
            summary=_summarize_by_interval(MAIN_TEMPERATURES_C, fig3_estimates),
        ),
        "fig4_total_time_constrained": HistogramReconstruction(
            name="fig4_total_time_constrained",
            title="Fig. 4 reproduction: total-time constrained LLS",
            estimates=fig4_estimates,
            summary=_summarize_by_interval(MAIN_TEMPERATURES_C, fig4_estimates),
        ),
    }


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


def nnls_fend_summary(
    samples: int = 2000,
    seed: int = 2015,
    noise_fraction: float = 0.05,
) -> pd.DataFrame:
    estimates = main_reconstruction_estimates(
        samples=samples,
        seed=seed,
        method="nnls",
        noise_fraction=noise_fraction,
    )
    sensor21 = [get_sensor(21)]
    temperatures_k = _temperatures_k(MAIN_TEMPERATURES_C)
    fends = np.array(
        [
            crystallinity_from_durations(sensor21, temperatures_k, estimate)[0]
            for estimate in estimates
        ]
    )
    return pd.DataFrame(
        [
            {
                "method": "scipy_nnls",
                "sensor21_fend_mean": float(np.mean(fends)),
                "sensor21_fend_std": float(np.std(fends, ddof=1)),
            }
        ]
    )


def numerical_diagnostics(alpha: float = 1e-11) -> dict[str, object]:
    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = _temperatures_k(MAIN_TEMPERATURES_C)
    fractions = crystallinity_from_durations(sensors, temperatures_k, MAIN_DURATIONS_S)
    matrix = sensor_matrix(sensors, temperatures_k)
    total_time = float(np.sum(MAIN_DURATIONS_S))

    lls = solve_lls(sensors, temperatures_k, fractions)
    regularized = solve_tikhonov(sensors, temperatures_k, fractions, alpha=alpha)
    total_time_alpha0 = solve_tikhonov_total_time(
        sensors,
        temperatures_k,
        fractions,
        alpha=0.0,
        total_time_s=total_time,
    )
    total_time_regularized = solve_tikhonov_total_time(
        sensors,
        temperatures_k,
        fractions,
        alpha=alpha,
        total_time_s=total_time,
    )

    return {
        "condition_number": float(np.linalg.cond(matrix)),
        "alpha": float(alpha),
        "noise_placement": "fractional_crystallinity",
        "fig4_allows_negative_durations": True,
        "noiseless_lls_error_norm_s": float(np.linalg.norm(lls - MAIN_DURATIONS_S)),
        "noiseless_regularized_error_norm_s": float(np.linalg.norm(regularized - MAIN_DURATIONS_S)),
        "noiseless_total_time_alpha0_error_norm_s": float(
            np.linalg.norm(total_time_alpha0 - MAIN_DURATIONS_S)
        ),
        "noiseless_total_time_regularized_error_norm_s": float(
            np.linalg.norm(total_time_regularized - MAIN_DURATIONS_S)
        ),
        "noiseless_total_time_regularized_solution_s": " ".join(
            f"{value:.6g}" for value in total_time_regularized
        ),
    }


def noise_model_comparison(
    samples: int = 400,
    seed: int = 2018,
    noise_fraction: float = 0.05,
    alpha: float = 1e-11,
) -> pd.DataFrame:
    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = _temperatures_k(MAIN_TEMPERATURES_C)
    fractions = crystallinity_from_durations(sensors, temperatures_k, MAIN_DURATIONS_S)
    observation = observation_from_fraction(fractions)
    total_time = float(np.sum(MAIN_DURATIONS_S))
    rows = []

    for noise_placement in ["fractional_crystallinity", "linearized_observation"]:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(samples):
            if noise_placement == "fractional_crystallinity":
                noisy_fractions = add_multiplicative_noise(fractions, noise_fraction, rng)
            else:
                noisy_observation = observation * (
                    1.0 + rng.normal(0.0, noise_fraction, size=observation.shape)
                )
                noisy_fractions = fraction_from_observation(noisy_observation)
            estimates.append(
                solve_tikhonov_total_time(
                    sensors,
                    temperatures_k,
                    noisy_fractions,
                    alpha=alpha,
                    total_time_s=total_time,
                )
            )
        estimates = np.vstack(estimates)
        rows.append(
            {
                "noise_placement": noise_placement,
                "fig4_t3_mean_s": float(np.mean(estimates[:, 2])),
                "fig4_t3_std_s": float(np.std(estimates[:, 2], ddof=1)),
                "fig4_t4_mean_s": float(np.mean(estimates[:, 3])),
                "fig4_t4_std_s": float(np.std(estimates[:, 3], ddof=1)),
                "fig4_t5_mean_s": float(np.mean(estimates[:, 4])),
                "fig4_t5_std_s": float(np.std(estimates[:, 4], ddof=1)),
            }
        )

    return pd.DataFrame(rows)


def noise_amplitude_sweep(
    samples: int = 400,
    seed: int = 2019,
    noise_fractions: np.ndarray | None = None,
    alpha: float = 1e-11,
) -> pd.DataFrame:
    if noise_fractions is None:
        noise_fractions = np.array([0.0025, 0.005, 0.01, 0.02, 0.05], dtype=float)

    rows: list[dict[str, float | str]] = []
    placements = ["fractional_crystallinity", "linearized_observation"]

    main_sensors = [get_sensor(number) for number in range(21, 26)]
    main_temperatures_k = _temperatures_k(MAIN_TEMPERATURES_C)
    main_fractions = crystallinity_from_durations(
        main_sensors,
        main_temperatures_k,
        MAIN_DURATIONS_S,
    )
    solve_fig3 = lambda noisy: solve_tikhonov(
        main_sensors,
        main_temperatures_k,
        noisy,
        alpha=alpha,
    )

    spike_sensors = [get_sensor(number) for number in range(1, 4)]
    spike_recon_temperatures_k = _temperatures_k(SPIKE_RECON_TEMPERATURES_C)
    spike_truth_temperatures_c, spike_truth_durations_s = _spike_truth(1100.0)
    spike_fractions = crystallinity_from_durations(
        spike_sensors,
        _temperatures_k(spike_truth_temperatures_c),
        spike_truth_durations_s,
    )
    solve_spike = lambda noisy: solve_tikhonov_total_time(
        spike_sensors,
        spike_recon_temperatures_k,
        noisy,
        alpha=alpha,
        total_time_s=float(np.sum(spike_truth_durations_s)),
    )

    for placement in placements:
        for noise_fraction in noise_fractions:
            fig3 = _monte_carlo_estimates(
                main_fractions,
                samples,
                seed,
                solve_fig3,
                float(noise_fraction),
                noise_placement=placement,
            )
            rows.append(
                {
                    "experiment": "fig3_regularized_lls",
                    "noise_placement": placement,
                    "noise_fraction": float(noise_fraction),
                    "t4_mean_s": float(np.mean(fig3[:, 3])),
                    "t4_std_s": float(np.std(fig3[:, 3], ddof=1)),
                    "t5_mean_s": float(np.mean(fig3[:, 4])),
                    "t5_std_s": float(np.std(fig3[:, 4], ddof=1)),
                }
            )

            spike = _monte_carlo_estimates(
                spike_fractions,
                samples,
                seed,
                solve_spike,
                float(noise_fraction),
                noise_placement=placement,
            )
            rows.append(
                {
                    "experiment": "spike_1100",
                    "noise_placement": placement,
                    "noise_fraction": float(noise_fraction),
                    "t1_mean_s": float(np.mean(spike[:, 0])),
                    "t1_std_s": float(np.std(spike[:, 0], ddof=1)),
                    "t2_mean_s": float(np.mean(spike[:, 1])),
                    "t2_std_s": float(np.std(spike[:, 1], ddof=1)),
                    "t3_mean_s": float(np.mean(spike[:, 2])),
                    "t3_std_s": float(np.std(spike[:, 2], ddof=1)),
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
