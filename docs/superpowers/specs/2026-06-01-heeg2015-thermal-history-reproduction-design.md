# Heeg 2015 Thermal History Reconstruction Reproduction Design

## Goal

Reproduce the key simulation results from Heeg 2015 for inverse thermal history reconstruction of a partially crystallized glass-ceramic sensor array.

## Scope

The reproduction will implement the paper's pre-nucleated, two-dimensional growth simplification and inverse least-squares reconstruction workflow. It will focus on:

- The main five-temperature reconstruction example using sensors 21-25 from Table I.
- Noise-driven histograms comparable to Figs. 3-5.
- Regularization sensitivity and reconstructed end-crystallinity checks comparable to Figs. 6-7.
- The short high-temperature spike example comparable to Figs. 8-9.

The implementation will not model nucleation-order effects or the full Fair 2008 pattern-matching workflow, because Heeg 2015 explicitly simplifies the inverse problem by omitting nucleation.

## Architecture

Create a small Python package under `src/thermal_history/` with clear separation between paper data, forward model, inverse solvers, experiments, and plotting. A single script, `scripts/reproduce_heeg2015.py`, will run all configured experiments and write reproducible outputs under `outputs/heeg2015/`.

## Model

The forward model will use the paper's site-saturated two-dimensional expression:

```text
F = 1 - exp(-(4*pi/3) * (sum_i u(T_i) * t_i)^2)
```

The growth rate will use the growth expression and viscosity law from the paper:

```text
eta(T) = A * exp(Q / (T - s))
u(T) = 10*kB*T / (3*pi*eta(T)*a0^2) * (1 - exp(-DHf*(Tm - T)/(R*Tm*T)))
```

Temperatures will be handled internally in Kelvin. Table I values will be encoded in a data module, including at least sensors 1-32.

## Inverse Solvers

The implementation will provide:

- Standard linear least squares for noiseless sanity checks.
- Tikhonov-regularized least squares with optional total-time constraint.
- Non-negative least squares using SciPy.
- A constrained regularized method for the short-spike experiment.

The transformed observation vector will follow the paper:

```text
y = sqrt(-log(1 - F) * 3 / (4*pi))
```

The sensor matrix will contain growth rates `u(T)` for each sensor and reconstruction temperature interval.

## Experiments

### Main Reconstruction

Use the paper's thermal history:

```text
700 C: 600 s
800 C: 400 s
900 C: 300 s
1000 C: 200 s
1100 C: 100 s
```

Use sensors 21-25 from Table I. Generate noisy sensor readouts with 5% multiplicative noise, reconstruct repeated Monte Carlo estimates, and save summary CSVs and histogram PNGs.

### Regularization Sweep

Sweep alpha values around the paper's important region near `1e-11`, record mean and standard deviation of reconstructed durations, and recompute end crystallinity for sensor 21 from reconstructed histories.

### Short Spike

Use sensors 1-3 and reconstruction temperatures:

```text
900 C: 1000 s
1000 C: 590 s
1100 C: 10 s spike
```

With 5% multiplicative noise and alpha `1e-11`, reproduce the reported behavior that a `10 s` spike at `1100 C` is recovered near `10 s`, with standard deviation of a few seconds. Also sweep spike temperature to reproduce the monotonic increase in the inferred `1100 C` interval duration.

## Outputs

The reproduction script will write:

- `outputs/heeg2015/main_reconstruction_summary.csv`
- `outputs/heeg2015/regularization_sweep.csv`
- `outputs/heeg2015/spike_reconstruction_summary.csv`
- `outputs/heeg2015/fig_main_histograms.png`
- `outputs/heeg2015/fig_regularization_sweep.png`
- `outputs/heeg2015/fig_fend_reconstruction.png`
- `outputs/heeg2015/fig_spike_reconstruction.png`
- `outputs/heeg2015/README.md`

## Testing

Tests will verify:

- Growth-rate calculations return finite positive rates for paper temperatures.
- Noiseless five-temperature LLS reconstruction recovers the input durations within tight tolerance.
- Observation transform and inverse transform are numerically stable for non-saturated readouts.
- NNLS and constrained methods return non-negative durations.
- The short-spike Monte Carlo estimate recovers the `1100 C` interval near `10 s` for a fixed random seed.

## Assumptions

- Noise is multiplicative on fractional crystallinity, then clipped to a valid interval before transformation.
- Monte Carlo sample counts can be configured, with defaults high enough for stable plots but not excessive for local runs.
- Exact numerical reproduction of the paper's plotted histogram bins is not expected because the random seeds and implementation details are not published; the target is reproduction of the reported qualitative behavior and numeric summary values.
