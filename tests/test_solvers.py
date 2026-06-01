import numpy as np
import pytest

from thermal_history.model import crystallinity_from_durations
from thermal_history.paper_data import CELSIUS_TO_KELVIN, get_sensor
from thermal_history.solvers import (
    add_multiplicative_noise,
    solve_lls,
    solve_nnls,
    solve_tikhonov_total_time,
)


def five_interval_case():
    sensors = [get_sensor(number) for number in range(21, 26)]
    temperatures_k = np.array([700, 800, 900, 1000, 1100], dtype=float) + CELSIUS_TO_KELVIN
    durations_s = np.array([600, 400, 300, 200, 100], dtype=float)
    fractions = crystallinity_from_durations(sensors, temperatures_k, durations_s)
    return sensors, temperatures_k, durations_s, fractions


def test_noiseless_lls_recovers_five_interval_history():
    sensors, temperatures_k, durations_s, fractions = five_interval_case()

    estimate = solve_lls(sensors, temperatures_k, fractions)

    np.testing.assert_allclose(estimate, durations_s, rtol=1e-8, atol=1e-6)


def test_nnls_returns_non_negative_duration_estimates_for_noisy_readout():
    sensors, temperatures_k, _durations_s, fractions = five_interval_case()
    noisy = add_multiplicative_noise(fractions, noise_fraction=0.05, rng=np.random.default_rng(7))

    estimate = solve_nnls(sensors, temperatures_k, noisy)

    assert estimate.shape == (5,)
    assert np.all(estimate >= 0.0)


def test_total_time_tikhonov_preserves_known_exposure_duration():
    sensors, temperatures_k, durations_s, fractions = five_interval_case()
    noisy = add_multiplicative_noise(fractions, noise_fraction=0.05, rng=np.random.default_rng(8))

    estimate = solve_tikhonov_total_time(
        sensors,
        temperatures_k,
        noisy,
        alpha=1e-11,
        total_time_s=float(np.sum(durations_s)),
    )

    assert np.sum(estimate) == pytest.approx(np.sum(durations_s), abs=1e-6)
    assert np.all(estimate >= -1e-8)
