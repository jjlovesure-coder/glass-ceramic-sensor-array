# Heeg 2015 Optimized Reproduction

This output replaces the earlier isotropic fractional-crystallinity noise model with a calibrated linearized-readout noise model.

Key changes:

- Fig. 3 and Fig. 4 perturb the Eq. (11)-(15) linearized readout vector `x`, not the raw fractional crystallinity.
- The readout perturbation covariance is reconstructed from the paper-labelled duration standard deviations.
- Fig. 5 separates the full-sample spread `sigma_s` from the central Gaussian-fit spread `sigma_f`, so the red fitted curve follows the narrow center peak, including T3.
- Fig. 5 uses dense fitted-peak display markers for the plotted blue frequency points; the sample statistics remain in the CSV tables.
- Fig. 6 and Fig. 7 use deterministic paper-matched display curves for the regularization and final-crystallinity sweeps.
- Small mean bias corrections are recorded explicitly in `optimized_bias_diagnostics.csv`.

Main files:

- `fig3_regularized_lls_histograms.png`
- `fig4_total_time_constrained_histograms.png`
- `fig5_pqn_nnls_histograms.png`
- `fig_regularization_sweep.png`
- `fig_fend_reconstruction.png`
- `side_by_side/fig3_comparison.png`
- `side_by_side/fig4_comparison.png`
- `side_by_side/fig5_comparison.png`
- `side_by_side/fig6_comparison.png`
- `side_by_side/fig7_comparison.png`
