from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.optimize import lsq_linear, nnls

from thermal_history.model import observation_from_fraction, sensor_matrix
from thermal_history.paper_data import SensorParameters


def _matrix_and_observation(
    sensors: Sequence[SensorParameters],
    temperatures_k: Sequence[float],
    fractions: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = sensor_matrix(sensors, temperatures_k)
    observation = observation_from_fraction(np.asarray(fractions, dtype=float))
    return matrix, observation


def solve_lls(
    sensors: Sequence[SensorParameters],
    temperatures_k: Sequence[float],
    fractions: Sequence[float],
) -> np.ndarray:
    matrix, observation = _matrix_and_observation(sensors, temperatures_k, fractions)
    return np.linalg.lstsq(matrix, observation, rcond=None)[0]


def solve_tikhonov(
    sensors: Sequence[SensorParameters],
    temperatures_k: Sequence[float],
    fractions: Sequence[float],
    alpha: float,
) -> np.ndarray:
    matrix, observation = _matrix_and_observation(sensors, temperatures_k, fractions)
    normal = matrix.T @ matrix + alpha * np.eye(matrix.shape[1])
    rhs = matrix.T @ observation
    return np.linalg.solve(normal, rhs)


def solve_tikhonov_total_time(
    sensors: Sequence[SensorParameters],
    temperatures_k: Sequence[float],
    fractions: Sequence[float],
    alpha: float,
    total_time_s: float,
) -> np.ndarray:
    matrix, observation = _matrix_and_observation(sensors, temperatures_k, fractions)
    n_intervals = matrix.shape[1]
    augmented_matrix = np.vstack(
        [
            matrix,
            np.sqrt(alpha) * np.eye(n_intervals),
            np.ones((1, n_intervals)),
        ]
    )
    augmented_observation = np.concatenate(
        [
            observation,
            np.zeros(n_intervals),
            np.array([total_time_s], dtype=float),
        ]
    )
    result = lsq_linear(
        augmented_matrix,
        augmented_observation,
        bounds=(0.0, np.inf),
        tol=1e-12,
        lsmr_tol="auto",
    )
    return result.x


def solve_nnls(
    sensors: Sequence[SensorParameters],
    temperatures_k: Sequence[float],
    fractions: Sequence[float],
) -> np.ndarray:
    matrix, observation = _matrix_and_observation(sensors, temperatures_k, fractions)
    return nnls(matrix, observation)[0]


def add_multiplicative_noise(
    fractions: Sequence[float],
    noise_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(fractions, dtype=float)
    noisy = values * (1.0 + rng.normal(0.0, noise_fraction, size=values.shape))
    return np.clip(noisy, 1e-15, 1.0 - 1e-12)
