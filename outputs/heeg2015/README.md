# Heeg 2015 Reproduction Outputs

Monte Carlo samples per primary experiment: 2000

Files:

- `main_reconstruction_summary.csv`: five-interval reconstruction statistics.
- `fig3_regularized_lls_summary.csv`: regularized LLS statistics for Fig. 3.
- `fig4_total_time_constrained_summary.csv`: total-time constrained statistics for Fig. 4.
- `numerical_diagnostics.csv`: condition number, alpha, noise placement, and noiseless solver checks.
- `noise_model_comparison.csv`: Fig. 4 summary comparison for fractional-crystallinity and linearized-observation noise.
- `noise_amplitude_sweep.csv`: effective-noise sweep for paper-target discrepancy diagnosis.
- `nnls_fend_summary.csv`: Fig. 7-style SciPy NNLS objective sensor 21 end-crystallinity summary.
- `regularization_sweep.csv`: alpha sweep and sensor 21 end-crystallinity checks.
- `spike_reconstruction_summary.csv`: short spike reconstruction statistics.
- `paper_target_observations.csv`: observed metrics extracted from generated outputs and model diagnostics.
- `paper_target_validation.csv`: target-by-target tolerance validation against paper text targets.
- `paper_target_validation.md`: human-readable validation summary.
- `fig3_regularized_lls_histograms.png`: reproduction of Fig. 3 histogram style.
- `fig4_total_time_constrained_histograms.png`: reproduction of Fig. 4 histogram style.
- `fig_main_histograms.png`: histograms comparable to the paper's main examples.
- `fig_regularization_sweep.png`: duration sensitivity to regularization.
- `fig_fend_reconstruction.png`: reconstructed sensor 21 crystallinity.
- `fig_spike_reconstruction.png`: estimated high-temperature duration versus spike temperature.

The model uses the paper's pre-nucleated two-dimensional simplification.
Fig. 3/Fig. 4 alpha: `1e-11`.
Sensor matrix condition number: `3.83305e+07`.
Noise placement: `fractional_crystallinity`.
Fig. 4 uses an exact total-time equality constraint and permits negative interval estimates; NNLS/non-negativity is kept separate for Fig. 5-style solves.