# Torsional Autocorrelation → Global Step Size Dataset (Windowed CSV)

This README documents the derived dataset produced from ATLAS MD trajectories for testing whether **recent changes in local torsional decorrelation** predict **forthcoming global conformational displacement**.

The dataset is **windowed**: for each protein × replicate, we slide non-overlapping windows in time using the design  
**prev [W] | curr [W] | gap [≥W] | outcome [W]**. Predictors are computed from `prev` and `curr`; the outcome is computed on the future `outcome` window. One row in the CSV corresponds to one such windowed observation.

---

## 1) Source & Scope

- **MD source**: ATLAS database (https://www.dsimb.inserm.fr/ATLAS) trajectories; aligned, PBC-removed protein-only `*_fit.xtc` with associated `*.pdb` topology.  
- **Proteins**: 5 monomeric proteins × 3 replicates × 100 ns each (≈10,000 frames per replicate at 10 ps/frame).  
- **Focus**: Associational population trends under a fixed MD protocol (not causal).

---

## 2) Statistical Design Summary

- **Goal**: Quantify whether **Δr** (the change in torsional decorrelation rate between two past windows) predicts **|Δs|** (the future global displacement) while adjusting for local amplitude (RMSF) and ϕ–ψ coupling, and probing heterogeneity by secondary structure (SS).  
- **Key estimand**: Population‑level adjusted association between standardized `Δr` and `|Δs|`.  
- **Outcome window** is **strictly in the future** of predictor windows with a **gap ≥ W** to curb simultaneity and short‑range leakage.  
- **Thinning**: Window anchors advance in steps of `thin_frames` (often = `W_frames`) to reduce dependence among rows.

---

## 3) Variable Dictionary (column groups)

### A. Identity & timing
- `pdb_chain` — PDB and chain, e.g., `1h02_B`.  
- `pdb_id` — Four‑letter PDB ID.  
- `replicate` — `R1`/`R2`/`R3`.  
- `n_frames_total` — Frames in this replicate.  
- `dt_ps` — Frame spacing (ps). ATLAS default is ~10 ps.  
- `W_frames`, `gap_frames`, `thin_frames` — Window length, gap, and thinning **in frames**.  
- `window_index`, `anchor_frame` — Index and starting frame of this observation.  
- `prev_start_ps`, `prev_end_ps`, `curr_start_ps`, `curr_end_ps`, `out_start_ps`, `out_end_ps` — Window boundaries in **ps**.  
- `N_ca` — Number of Cα atoms.  
- `K_rama` — Number of residues with well‑defined ϕ/ψ used in torsion analysis.

### B. Outcome (future global displacement)
- `Y_abs_ds` — \|Δs\| over the **outcome** window, where  
  **s(t)** = **PC1** of standardized `[Rg(t), Q_native(t)]`, computed **within protein × replicate**.  
  The PC1 sign is fixed so `corr(s, −z(Q_native)) ≥ 0` (i.e., larger `s` corresponds to more “unfolded‑like” changes).  
  **Use**: Primary dependent variable.

### C. Torsional decorrelation (predictors)
- Circular ACF: ρ(k) = mean\_t cos(θ(t+k) − θ(t)).  
- **τ** = first lag k with ρ(k) ≤ ρ\* (default 0.10); **r = 1/τ** per residue (units: per‑frame).  
- We aggregate r within each window across residues (and across ϕ/ψ where indicated) and take Δr = r\_curr − r\_prev.

Columns:
- `dr_phi_mean`, `dr_phi_median` — Δr for ϕ.  
- `dr_psi_mean`, `dr_psi_median` — Δr for ψ.  
- `dr_avg_mean`, `dr_avg_median` — Δr for (ϕ,ψ) averaged per residue.  
- `rprev_phi_mean`, `rprev_psi_mean`, `rprev_avg_mean` — r means in `prev` window.  
- `rcurr_phi_mean`, `rcurr_psi_mean`, `rcurr_avg_mean` — r means in `curr` window.

**Why used**: Δr summarizes **recent acceleration/deceleration of local angular dynamics**. A positive association with `Y_abs_ds` supports the hypothesis that **faster recent torsional decorrelation precedes larger forthcoming global motion**.

> Unit note: r is per‑frame. To think in per‑ns units, multiply by (1000 / `dt_ps`).

### D. Secondary structure (SS) features
SS is assigned per **window** by Ramachandran “boxes” over (ϕ,ψ):  
- **H** (helix): φ ∈ [−100, −20], ψ ∈ [−80, 10] (liberal α‑region)  
- **E** (sheet): φ ∈ [−180, −90], ψ ∈ [90, 180]  
- **L** (loop/other): otherwise  
Each residue’s label is the **majority** class across frames in the window.

Fractions and change:
- `ss_frac_H_prev`, `ss_frac_E_prev`, `ss_frac_L_prev` — Fraction of residues labeled H/E/L in **prev**.  
- `ss_frac_H_curr`, `ss_frac_E_curr`, `ss_frac_L_curr` — Fraction in **curr**.  
- `ss_switch_frac` — Fraction of residues whose label changed from prev→curr (a volatility indicator).

SS‑stratified r and Δr (using **current** labels for stratification):
- `rprev_avg_mean_H/E/L` — Mean r (avg of ϕ,ψ) in **prev** among residues that are H/E/L in **curr**.  
- `rcurr_avg_mean_H/E/L` — Mean r in **curr** for H/E/L.  
- `dr_avg_mean_H/E/L` — Δr for H/E/L.  
- `n_res_H_curr`, `n_res_E_curr`, `n_res_L_curr` — Counts of residues in each SS class for **curr**.

**Why used**: Helices/sheets are typically **more stable**; loops **fluctuate** and often drive local‑to‑global coupling. These features lets us ask if **loop‑specific Δr** carries the predictive signal for `|Δs|` even when whole‑protein averages dilute it.

### E. Coupling & amplitude controls
- `C_curr` — ϕ–ψ **coupling** in `curr`: mean cos(ϕ − ψ) over residues & frames (−1 to 1).  
  **Why**: Controls for internal angular dependence that might mimic or mediate Δr’s effect.
- `rmsf_mean`, `rmsf_median`, `rmsf_q25`, `rmsf_q75` — **Cα RMSF** summaries within `curr`.  
  **Why**: Controls for **amplitude** of positional fluctuations (a confounder for both Δr and `|Δs|`).

### F. Global diagnostics (window means)
- `rg_prev_mean`, `rg_curr_mean`, `rg_out_mean` — Radius of gyration (Å).  
- `q_prev_mean`,  `q_curr_mean`,  `q_out_mean`  — Fraction of native contacts (0–1).

**Why used**: Help interpret s(t) and visualize trajectory evolution per window.

### G. Convenience standardizations (within protein × replicate)
For common modeling, the script appends `*_z_within_rep` for these columns (subtract mean, divide by SD **within prot×rep**):
- `Y_abs_ds`, `dr_avg_mean`, `dr_phi_mean`, `dr_psi_mean`, `rmsf_mean`, `C_curr`,  
  `ss_frac_H_curr`, `ss_frac_E_curr`, `ss_frac_L_curr`, `dr_avg_mean_H`, `dr_avg_mean_E`, `dr_avg_mean_L`.

**Why used**: Align scales for regression and mixed‑effects models; absorb level shifts across replicates.

---

## 4) Modeling Guidance (typical uses)

### Primary confirmatory model (linear mixed‑effects)
- **Outcome**: `Y_abs_ds_z_within_rep`  
- **Predictor**: `dr_avg_mean_z_within_rep`  
- **Controls**: `rmsf_mean_z_within_rep`, `C_curr_z_within_rep`  
- **Random effects**: protein (and optionally replicate) random intercepts

**Hypothesis**: slope(Δr) > 0 → faster recent torsional decorrelation predicts larger forthcoming global displacement.

### SS‑aware analysis
- Add `dr_avg_mean_L_z_within_rep` (loop‑specific Δr) alongside the overall Δr; or
- Stratify by `ss_frac_*` (e.g., inspect effects at high loop content).

### Nonparametric corroboration
- Partial Spearman between `Y_abs_ds` and `dr_avg_mean` after residualizing out `rmsf_mean` and `C_curr` within prot×rep.

---

## 5) Units & Scaling

- Time columns: **ps** (see `dt_ps` to convert frames→ps).  
- Rg, RMSF: **Å**.  
- Q_native: **fraction** (0–1).  
- Coupling `C_curr`: **unitless** in [−1, 1].  
- Rates `r*`: **per‑frame**; Δr is the difference of two per‑frame rates.  
- Outcome `Y_abs_ds`: **unitless** (PC1 distance).

---

## 6) Assumptions & Validity Considerations

- **Temporal ordering**: Outcome window is strictly after predictor windows with a non‑overlap gap.  
- **Dependence**: Thinning step reduces serial dependence between rows; use block‑aware SEs/permutation in inference.  
- **Stationarity (local)**: Moments are roughly stable within analyzed segments; inspect drift via QC plots.  
- **SS labeling**: Simple ϕ/ψ boxes; consider sensitivity to thresholds or external DSSP if available.  
- **Protocol specificity**: Results reflect the ATLAS CHARMM36m protocol; generalization to other force fields should be treated cautiously.

---

## 7) QC 

- Plot residual ACFs from your fitted model within prot×rep to check thinning adequacy.  
- Tertile plots of `dr_avg_mean` vs. `Y_abs_ds` (with CIs).  
- SS‑stratified distributions of `dr_avg_mean` (expect L > H/E).  
- Scatter of `rg_out_mean` vs `q_out_mean` to sanity‑check s(t) orientation.

---

## 8) Reproducibility (parameters to record)

- `window_ps`, `gap_mult`, `thin_frac`, `rho_star`, `contact_cutoff`, script version/commit, library versions (MDAnalysis, scikit‑learn).  
- Exact ATLAS subdirectories used and selection of PDB chains.

---

## 9) Expected Dataset Size

With 10,000 frames/replicate, `window_ps=2000 ps` (~200 frames), `gap_mult=1`, `thin_frac=1`:
- ≈ 47 rows/replicate × 3 replicates × 5 proteins ≈ **~705 rows total**.

---

## 10) Glossary

- **Circular ACF**: Autocorrelation on angles via `cos(Δθ)`, robust to wrap‑around at ±π.  
- **τ (decorrelation lag)**: Smallest lag where circular ACF drops below a threshold ρ\*.  
- **r (decorrelation rate)**: 1/τ; larger r = faster loss of angular memory.  
- **Δr**: Change in r between two past windows; positive Δr = recent acceleration.  
- **RMSF (Cα)**: Root‑mean‑square fluctuation of Cα positions within a window (proxy for amplitude).  
- **Q_native (Cα)**: Fraction of native Cα–Cα contacts present in a frame (contacts defined in the reference PDB).  
- **PC1 s(t)**: First principal component of standardized [Rg, Q] within prot×rep; a scalar progress coordinate.  
- **|Δs|**: Magnitude of forward change in s across the outcome window (global step size).  
- **H/E/L**: Helix/Sheet/Loop secondary‑structure classes assigned by (ϕ,ψ) boxes per window.