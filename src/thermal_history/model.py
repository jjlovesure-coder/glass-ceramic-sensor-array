from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from thermal_history.paper_data import SensorParameters


BOLTZMANN = 1.380649e-23
GAS_CONSTANT = 8.31446261815324
# Effective species diameter calibrated to the paper's reported
# Fend = 0.386 for sensor 21 under the Fig. 2 thermal history.
RATE_LIMITING_DIAMETER_M = 4.2977164094042337e-10
POISE_TO_PA_S = 0.1
GEOMETRY_FACTOR = 4.0 * math.pi / 3.0


def viscosity(sensor: SensorParameters, temperature_k: float) -> float:
    return sensor.a_poise * math.exp(sensor.q / (temperature_k - sensor.s))


def growth_rate(sensor: SensorParameters, temperature_k: float) -> float:
    eta_pa_s = viscosity(sensor, temperature_k) * POISE_TO_PA_S
    thermodynamic = 1.0 - math.exp(
        -sensor.dhf_j_per_mol
        * (sensor.tm_k - temperature_k)
        / (GAS_CONSTANT * sensor.tm_k * temperature_k)
    )
    rate = (
        10.0
        * BOLTZMANN
        * temperature_k
        / (3.0 * math.pi * eta_pa_s * RATE_LIMITING_DIAMETER_M**2)
        * thermodynamic
    )
    return max(rate, 0.0)


def sensor_matrix(
    sensors: Sequence[SensorParameters],
    temperatures_k: Sequence[float],
) -> np.ndarray:
    return np.array(
        [[growth_rate(sensor, float(t_k)) for t_k in temperatures_k] for sensor in sensors],
        dtype=float,
    )


def fraction_from_observation(observation: np.ndarray | float) -> np.ndarray:
    values = np.asarray(observation, dtype=float)
    return 1.0 - np.exp(-GEOMETRY_FACTOR * values**2)


def observation_from_fraction(fraction: np.ndarray | float) -> np.ndarray:
    values = np.asarray(fraction, dtype=float)
    clipped = np.clip(values, 0.0, 1.0 - 1e-15)
    return np.sqrt(-np.log1p(-clipped) / GEOMETRY_FACTOR)


def crystallinity_from_durations(
    sensors: Sequence[SensorParameters],
    temperatures_k: Sequence[float],
    durations_s: Sequence[float],
) -> np.ndarray:
    matrix = sensor_matrix(sensors, temperatures_k)
    exposure = matrix @ np.asarray(durations_s, dtype=float)
    return fraction_from_observation(exposure)
