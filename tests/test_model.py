import math

import numpy as np
import pytest

from thermal_history.model import (
    crystallinity_from_durations,
    growth_rate,
    observation_from_fraction,
    fraction_from_observation,
)
from thermal_history.paper_data import CELSIUS_TO_KELVIN, get_sensor, sensors_by_number


def test_table_i_contains_required_sensor_parameters():
    sensors = sensors_by_number()

    assert sensors[1].q == 39000.0
    assert sensors[1].tm_k == 1411.0
    assert sensors[21].q == 55000.0
    assert sensors[25].barrier_b == 86.0
    assert get_sensor(32).dhf_j_per_mol == 320_000.0


def test_growth_rate_is_positive_and_varies_by_temperature():
    sensor = get_sensor(21)

    rate_900 = growth_rate(sensor, 900.0 + CELSIUS_TO_KELVIN)
    rate_1100 = growth_rate(sensor, 1100.0 + CELSIUS_TO_KELVIN)

    assert math.isfinite(rate_900)
    assert math.isfinite(rate_1100)
    assert rate_900 > 0.0
    assert rate_1100 > rate_900


def test_fraction_observation_round_trip():
    fractions = np.array([1e-9, 0.1, 0.386, 0.9])

    observed = observation_from_fraction(fractions)
    round_trip = fraction_from_observation(observed)

    np.testing.assert_allclose(round_trip, fractions, rtol=1e-12, atol=1e-12)


def test_crystallinity_from_durations_returns_valid_sensor_readouts():
    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = np.array([700, 800, 900, 1000, 1100], dtype=float) + CELSIUS_TO_KELVIN
    durations_s = np.array([600, 400, 300, 200, 100], dtype=float)

    fractions = crystallinity_from_durations(sensors, temperatures_k, durations_s)

    assert fractions.shape == (5,)
    assert np.all(fractions > 0.0)
    assert np.all(fractions < 1.0)
    assert fractions[0] == pytest.approx(0.386, abs=0.01)
