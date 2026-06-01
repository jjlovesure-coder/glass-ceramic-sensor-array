# Glass-Ceramic Sensor Array Reproduction

This repository reproduces key simulations from Heeg 2015, "Equivalent thermal history reconstruction from a partially crystallized glass-ceramic sensor array."

## Run

```powershell
python -m pytest -v
python scripts/reproduce_heeg2015.py --samples 2000 --output outputs/heeg2015
```

The script writes CSV summaries, PNG figures, and an output README under `outputs/heeg2015/`.

## Notes

The implementation follows the paper's pre-nucleated two-dimensional growth simplification. The effective rate-limiting species diameter is calibrated to reproduce the paper's reported `Fend = 0.386` for sensor 21 under the five-interval thermal history.
