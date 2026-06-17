# Heeg 2015 Figure Comparison

This folder compares cropped original paper figures with the current reproduction outputs.

Generated folders:

- `paper_crops/`: cropped figures from the PDF.
- `reproduction_figures/`: current reproduction figures normalized to the paper figure numbers.
- `side_by_side/`: original-vs-reproduction visual comparisons.
- `metric_comparison.csv`: numeric target comparison extracted from paper text targets.

Metric targets passed: 16 / 22.

Main observations:

- Fig. 2, Fig. 6 noiseless recovery, and sensor 21 exact `Fend` are reproduced within the configured tolerance.
- Fig. 3 high-temperature mean is close, but the current 5% fractional-crystallinity noise model gives much wider distributions than the paper.
- Fig. 4 shows the correct total-time constrained method, but the remaining visual mismatch is again dominated by noisy spread.
- Fig. 5 is labelled as SciPy NNLS objective reproduction, not Kim PQN implementation; the mean `Fend` matches, while the spread remains larger than the paper value.
- Fig. 8/9 reproduce the spike profile and monotonic trend; the spike-duration means are close, but the standard deviations are still larger than the paper text targets.

Failed 5% text targets:

- fig3.t4_mean_s: observed=281.807, target=260, error=8.39%
- fig3.t5_std_s: observed=66.5718, target=7, error=851.03%
- fig7.nnls_fend_std: observed=0.0143482, target=0.003, error=378.27%
- fig8_spike_1100.t1_std_s: observed=58.0553, target=12, error=383.79%
- fig8_spike_1100.t2_std_s: observed=65.5867, target=15, error=337.24%
- fig8_spike_1100.t3_std_s: observed=7.72543, target=3.5, error=120.73%
