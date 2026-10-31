# Probabilities of causation in nondeterministic causal models

Reproduction code for the synthetic and semi-synthetic experiments in the paper *Probabilities of Causation in Probabilistic Nondeterministic Structural Causal Models*. The repository generates the approved main-text and appendix figures from frozen, documented latent kernels, and can separately refit the Adult and JOBS II models from their public source data.

The calculations concern population-level sharp identification intervals. Our semi-synthetic counterfactual ground truths depend on explicitly fitted causal models and are not identified causal effects in the underlying observational studies.

## Quick start

Requires Python 3.10+ (tested on Python 3.13). From the repository root:

```bash
python -m pip install -e .
python reproduce_all.py
python -m pip install pytest
python -m pytest -q
```

The plotting command requires no raw-data downloads: frozen, fitted kernels and synthetic DGP parameters are provided in `data/derived/` and `data/simulations/`. Output files are written to the ignored `results/figures/` directory.

| Produced file | Contents |
|---|---|
| `results/figures/applications_adult_jobs.pdf` | Two-row Adult and JOBS II main-text figure: PN, PS, PNS |
| `results/figures/simulation_main.pdf` | Two anchor simulation examples, common margin / soft exogeneity |
| `results/figures/simulation_grid_appendix.pdf` | Final three-page, 18-DGP appendix grid (margin left, exogeneity right) |
| `results/figures/simulation_grid_page_{1,2,3}.pdf` | Individually embeddable appendix pages |
| Same stems with `.png` | Preview images |

The approved figure versions are included in `figures/reference/`. The code recreates their data, curves, structure and legends; minor rendering differences can arise from Matplotlib versions/fonts.

## Repository layout

```text
.
├── README.md
├── pyproject.toml
├── reproduce_all.py               # Generate the three final figure groups
├── src/pnscm/
│   ├── bounds.py                  # Analytic TP, margin, soft-exogeneity, s-monotone bounds
│   ├── lp.py                      # Generic independent 8-corner LP verifier
│   └── plot_utils.py              # Shared visual conventions
├── experiments/
│   ├── adult.py                   # Refit Adult two-state model from raw data
│   ├── jobs.py                    # Refit JOBS II ITT-calibrated model from raw data
│   ├── applications.py            # Plot Adult + JOBS II composite
│   └── simulations.py             # Plot 2 anchor examples + 18 appendix DGPs
├── data/
│   ├── derived/                   # Frozen Adult / JOBS II fitted kernels
│   ├── simulations/               # Exact DGP and grid specification CSVs
│   └── raw/                       # Optional source datasets (not distributed)
├── figures/reference/             # Previously approved figures
└── tests/test_regression.py       # Oracle coverage, numerical gains, independent LP
```

### Mathematical conventions

For a finite latent state space, we define the masses and Bernoulli kernels

\[
 w_u=p(u),\quad a_u=p(x\mid u),\quad
 b_u=p(y_x\mid u),\quad c_u=p(y_{x'}\mid u).
\]

The probability-of-causation ground truths use the PNSCM product counterfactual semantics, including

\[
 \mathrm{PNS}=\sum_u w_u b_u(1-c_u),\quad
 \mathrm{PN}=\frac{\sum_u w_u a_u b_u(1-c_u)}{p(x,y)},\quad
 \mathrm{PS}=\frac{\sum_u w_u(1-a_u)b_u(1-c_u)}{p(x',y')}.
\]

For the **common margin**, all three mechanisms are constrained to \([\varepsilon,1-\varepsilon]\). For **soft exogeneity**, \(|a_u-p(x)|\leq\delta\). `bounds.py` implements the analytic sharp endpoints and the compatibility thresholds \(\varepsilon_{\max}\) and \(\delta_{\min}\). The black dotted lines in the figures are **oracle** \(\varepsilon^*,\delta^*\), not estimated from \(\mathcal D\). Red lines are **data-compatibility thresholds**, which are not guarantees of structural validity. The green dashed strict-TP-exogeneity benchmarks can be invalid under the fitted latent model.

The `lp.py` solver independently optimizes the mass distribution over a fixed eight-point corner support for the selected restriction; it does **not** optimize the paper's experimental curves (which use the analytic formulas).

## Adult fitting protocol

Binary **treatment:** education at least bachelor's level (`education_num >= 13`). **Outcome:** annual income above $50k. **Latent factor:** age at least 45. Use `adult.data` and `adult.test` together (48,842 records), and estimate each conditional Bernoulli probability with **add-one Laplace smoothing**, \((k+1)/(n+2)\); estimate the two latent weights by empirical frequencies.

The assumed PNSCM is \(U\to X\to Y\) and \(U\to Y\). This treats age as sufficient for confounding within the constructed model; that assumption is not established by the Adult data.

Download the Adult data from the [UCI Adult repository](https://archive.ics.uci.edu/dataset/2/adult), then save as `data/raw/adult.data` and `data/raw/adult.test` and run:

```bash
python -m experiments.adult
```

The command writes `results/derived/adult_kernels_refit.csv` and prints the maximum absolute difference from the frozen reference. It never overwrites `data/derived/adult_kernels.csv`.

## JOBS II fitting protocol

The 899-row dataset supplied with R's [`mediation` package](https://cran.r-project.org/package=mediation) is an illustrative subset of a randomized encouragement study. The columns are:

- **Z:** randomized intervention offer (`treat`), rather than the causal treatment analyzed in our bounds.
- **X:** actual workshop participation (`comply`; one-sided participation in the subset).
- **U:** high baseline depressive symptoms (`depress1 >= 1.83`, the rounded baseline median).
- **Y:** fewer subsequent depressive symptoms (`depress2 <=` the **unrounded sample median**, approximately 1.60).

The distinction between the rounded displayed post-intervention cutoff and its stored floating-point representation matters: the unrounded median includes two observations that would be excluded by the literal number `1.60`. Our frozen model uses the **unrounded median**.

Let \(r=p(Z=1)\), \(q_u=p(X=1\mid Z=1,u)\), and \(m_{zu}=p(Y=1\mid Z=z,u)\). Using add-one smoothing for \(q_u,m_{0u},m_{1u}\), the ITT-calibrated model is

\[
 a_u=rq_u,\quad c_u=m_{0u},\quad b_u=c_u+\frac{m_{1u}-m_{0u}}{q_u}.
\]

This calibration assumes random assignment of the offer, one-sided noncompliance, an **exclusion restriction** (offer only affects symptoms through participation), and **conditional response homogeneity** within each baseline depression stratum. These assumptions permit a semi-synthetic completion of unknown potential outcomes; they do *not* follow automatically from randomization. The fitted counterfactuals are model-implied.

To export the public R subset:

```r
install.packages("mediation")
data("jobs", package = "mediation")
write.csv(jobs, "data/raw/jobs_pilot.csv", row.names = FALSE)
```

Then:

```bash
python -m experiments.jobs
```

The command writes `results/derived/jobs_kernels_refit.csv` and compares it to the frozen reference. We do not distribute the full restricted ICPSR study data.

For a complete rerun from the raw data (rather than the checked-in fitted kernels), refit both studies and then explicitly point the plotting command at the outputs:

```bash
python -m experiments.adult
python -m experiments.jobs
python -m experiments.applications \
  --adult-kernels results/derived/adult_kernels_refit.csv \
  --jobs-kernels results/derived/jobs_kernels_refit.csv
```

The synthetic figures are independent of raw data and regenerate directly from the fixed DGP tables.


## Synthetic simulations

`data/simulations/anchor_dgps.csv` contains the two five-state anchor DGPs used for the main synthetic figure. `design_latents.csv` contains the 18 frozen five-state sensitivity DGPs, M11–M33 (margin) and D11–D33 (soft exogeneity); `design_summary.csv` records their targeted and achieved parameters. These are **curated design points**, not independent draws from a DGP distribution. The appendix figures pair M11/D11, M12/D12, M13/D13 on page 1; M21/D21 etc. on page 2; and M31/D31 etc. on page 3. Columns repeat PN, PS, PNS on the left and right. The labels use \(\varepsilon^*,\delta^*\).

`python -m experiments.simulations` directly generates the main synthetic figure and three-page appendix PDF. The fixed DGP tables are **inputs**, not regenerated via search/optimization at reproduction time.

## Validation and limitations

`python -m pytest -q` verifies frozen oracle values and width reductions, oracle coverage for all 20 synthetic DGPs and both semi-synthetic experiments, and independent LP agreement with analytic endpoints. If optional raw CSV files are available in `data/raw/`, it also checks that their refitted kernels match the frozen ones.

Results are exact for the constructed finite-population models, not uncertainty-calibrated estimates of causal effects in the historical studies. In particular, the JOBS II model inherits assumptions about compliance and the outcome response mechanism. The original screening was exploratory; its alternative DGPs and discarded preliminary outcomes are intentionally not needed to reproduce the final paper figures.
