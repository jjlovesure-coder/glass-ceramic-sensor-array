# Heeg 2015 Thermal History Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested Python reproduction of Heeg 2015 inverse thermal history reconstruction simulations.

**Architecture:** Create a small `thermal_history` package with paper parameters, forward crystallization transforms, inverse solvers, experiment runners, and plotting/output utilities. A CLI script runs all reproduction experiments into `outputs/heeg2015/`.

**Tech Stack:** Python 3.14, NumPy, SciPy, Matplotlib, pytest.

---

### Task 1: Paper Data and Forward Model

**Files:**
- Create: `src/thermal_history/__init__.py`
- Create: `src/thermal_history/paper_data.py`
- Create: `src/thermal_history/model.py`
- Create: `tests/test_model.py`

- [ ] Write failing tests for Table I sensor lookup, positive growth rates, observation transform round-trip, and noiseless five-temperature crystallinity generation.
- [ ] Run `python -m pytest tests/test_model.py -v` and verify the tests fail because modules are missing.
- [ ] Implement sensor data and model functions.
- [ ] Run `python -m pytest tests/test_model.py -v` and verify the tests pass.

### Task 2: Inverse Solvers

**Files:**
- Create: `src/thermal_history/solvers.py`
- Create: `tests/test_solvers.py`

- [ ] Write failing tests for noiseless LLS recovery of `(600, 400, 300, 200, 100)`, non-negative NNLS output, and constrained regularized output preserving total time.
- [ ] Run `python -m pytest tests/test_solvers.py -v` and verify the tests fail because solvers are missing.
- [ ] Implement LLS, Tikhonov, total-time constrained Tikhonov, NNLS, and clipped noisy observations.
- [ ] Run `python -m pytest tests/test_solvers.py -v` and verify the tests pass.

### Task 3: Reproduction Experiments

**Files:**
- Create: `src/thermal_history/experiments.py`
- Create: `tests/test_experiments.py`

- [ ] Write failing tests for fixed-seed main reconstruction summaries and the short-spike estimate near `10 s`.
- [ ] Run `python -m pytest tests/test_experiments.py -v` and verify the tests fail because experiment functions are missing.
- [ ] Implement Monte Carlo experiment functions for main reconstruction, regularization sweep, and spike sweep.
- [ ] Run `python -m pytest tests/test_experiments.py -v` and verify the tests pass.

### Task 4: Script, Plots, and Documentation

**Files:**
- Create: `scripts/reproduce_heeg2015.py`
- Create: `src/thermal_history/plotting.py`
- Create: `README.md`

- [ ] Write the CLI script and plotting helpers.
- [ ] Run `python scripts/reproduce_heeg2015.py --samples 500 --output outputs/heeg2015`.
- [ ] Verify the expected CSV, PNG, and output README files are produced.
- [ ] Run the full test suite with `python -m pytest -v`.
- [ ] Review `git diff` for accidental unrelated changes.
- [ ] Commit implementation with message `feat: reproduce heeg 2015 thermal history simulations`.
