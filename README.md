# Glass-Ceramic Sensor Array Reproduction

This repository reproduces key simulations from Heeg 2015, "Equivalent thermal history reconstruction from a partially crystallized glass-ceramic sensor array."

## Run

```powershell
python -m pytest -v
python scripts/reproduce_heeg2015.py --samples 2000 --output outputs/heeg2015
```

The script writes CSV summaries, PNG figures, and an output README under `outputs/heeg2015/`.

The generated `outputs/heeg2015/paper_target_validation.md` compares extracted paper-text targets against current outputs. The method reproduction currently passes deterministic and several mean-value targets, while some noisy Monte Carlo spread metrics remain outside the 5% target window; see `outputs/heeg2015/noise_amplitude_sweep.csv` for the effective-noise diagnostic sweep.

## Notes

The implementation follows the paper's pre-nucleated two-dimensional growth simplification. The effective rate-limiting species diameter is calibrated to reproduce the paper's reported `Fend = 0.386` for sensor 21 under the five-interval thermal history.
