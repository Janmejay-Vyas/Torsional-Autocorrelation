#!/usr/bin/env python3
"""
Build a windowed dataset for the "torsional autocorrelation → global step size" study,
including secondary-structure features (helix/sheet/loop) derived from Ramachandran angles.

This script will:
  1) Parse each protein/replicate.
  2) Per-frame globals: Rg (Cα), Q_native (Cα-based contacts).
  3) s(t) = PC1(z(Rg), z(Q)), sign-fixed so corr(s, −Q) ≥ 0.
  4) Window design: prev [W] | curr [W] | gap [≥W] | outcome [W].
  5) Within windows, compute predictors:
       - Torsional decorrelation rates r = 1/τ where τ is the first circular ACF lag below ρ*
       - Δr between curr and prev (phi/psi/avg; mean & median across residues)
       - ϕ–ψ coupling C = mean cos(ϕ − ψ) in curr window
       - RMSF (Cα) summaries in curr window (mean/median/quantiles)
       - Secondary structure (SS) from (ϕ,ψ) with simple Ramachandran boxes:
           H (helix), E (sheet), L (loop/other), per residue via majority across frames in the window
         SS features:
           - Fractions of residues in H/E/L for prev and curr windows
           - Fraction of residues whose SS label changed (prev→curr)
           - r-by-SS means and Δr-by-SS means (H/E/L) using r_avg per residue
  6) Outcome: Y = |Δs| over outcome window (future non-overlapping).

Outputs ONE tidy CSV row per (protein, replicate, window).

Dependencies:
  pip install MDAnalysis scikit-learn numpy pandas tqdm
"""

import argparse
import os
import re
import sys
from glob import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

# Third-party analysis libs
try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Ramachandran
except Exception:
    mda = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -------------------------- utils --------------------------

def fail(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def find_protein_dirs(root: str) -> List[str]:
    """Return immediate subdirs that look like '<pdb>_<chain>_protein' and contain a .pdb and *_fit.xtc."""
    if not os.path.isdir(root):
        fail(f"Root directory does not exist: {root}")
    subdirs = [os.path.join(root, d) for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))]
    kept = []
    for d in subdirs:
        pdbs = glob(os.path.join(d, "*.pdb"))
        xtcs = glob(os.path.join(d, "*_fit.xtc"))
        if pdbs and xtcs:
            kept.append(d)
    if not kept:
        fail(f"No protein directories found under {root}.")
    return sorted(kept)


def parse_pdb_chain_from_dir(d: str) -> Tuple[str, str]:
    """Extract '1h02_B' (pdb_chain) from folder like '.../1h02_B_protein'."""
    base = os.path.basename(d.rstrip("/"))
    m = re.match(r"([0-9A-Za-z]{4})_([A-Za-z0-9])_protein", base)
    if not m:
        all_pdb = glob(os.path.join(d, "*.pdb"))
        if not all_pdb:
            fail(f"Could not parse PDB/chain in directory name {base} and no PDB file present.")
        name = os.path.splitext(os.path.basename(all_pdb[0]))[0]
        parts = name.split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return name, "A"
    return m.group(1), m.group(2)


def load_universe_for_replicate(dirpath: str, pdb_path: str, rep_index: int):
    """Load Universe for replicate R{rep_index}. Prefer PDB for topology + aligned XTC for frames."""
    xtc = glob(os.path.join(dirpath, f"*prod_R{rep_index}_fit.xtc"))
    if not xtc:
        fail(f"No XTC for replicate R{rep_index} in {dirpath}")
    xtc = xtc[0]
    U = mda.Universe(pdb_path, xtc, in_memory=False)
    return U


def get_dt_ps(universe):
    """Frame spacing in ps."""
    ts = universe.trajectory
    if hasattr(ts, "dt") and ts.dt is not None:
        return float(ts.dt)
    t0 = ts[0].time
    if len(ts) > 1:
        t1 = ts[1].time
        return float(t1 - t0)
    return 10.0  # ATLAS default 10 ps


def rg_ca(coords: np.ndarray) -> float:
    """Radius of gyration (Cα only). coords: (N,3)"""
    c = coords.mean(axis=0)
    dif = coords - c
    return float(np.sqrt((dif * dif).sum(axis=1).mean()))


def build_native_contact_pairs(pdb_universe, cutoff: float = 8.0, min_seq_sep: int = 3) -> np.ndarray:
    """
    Build native contact list using Cα–Cα distances in the reference PDB.
    Returns array shape (M,2) with indices into the CA selection (0-based).
    """
    ca = pdb_universe.select_atoms("name CA")
    if len(ca) < 4:
        fail("Not enough CA atoms to build native contacts.")
    ca_pos = ca.positions
    N = len(ca)
    pairs = []
    for i in range(N):
        res_i = ca[i].residue.resid
        xyz_i = ca_pos[i]
        dif = ca_pos[i+1:] - xyz_i
        d2 = np.sum(dif * dif, axis=1)
        d = np.sqrt(d2)
        js = np.where(d <= cutoff)[0] + (i+1)
        for j in js:
            res_j = ca[j].residue.resid
            if abs(res_j - res_i) >= min_seq_sep:
                pairs.append((i, j))
    if not pairs:
        fail("No native contacts found; increase cutoff.")
    return np.array(pairs, dtype=int)


def fraction_native_contacts(ca_coords: np.ndarray, native_pairs: np.ndarray, cutoff: float = 8.0) -> float:
    """Fraction of native contacts for a frame, using CA-CA native_pairs and cutoff."""
    a = ca_coords[native_pairs[:, 0]]
    b = ca_coords[native_pairs[:, 1]]
    dif = a - b
    d2 = np.sum(dif * dif, axis=1)
    d = np.sqrt(d2)
    return float(np.mean(d <= cutoff))


def compute_global_series(universe, native_pairs: np.ndarray, ca_sel=None):
    """Iterate frames to compute times (ps), Rg_CA, Q_native."""
    if ca_sel is None:
        ca_sel = universe.select_atoms("name CA")
    times = []
    rg = []
    qn = []
    for _ in universe.trajectory:
        times.append(universe.trajectory.time)
        ca_coords = ca_sel.positions
        rg.append(rg_ca(ca_coords))
        qn.append(fraction_native_contacts(ca_coords, native_pairs))
    return np.asarray(times), np.asarray(rg), np.asarray(qn)


def compute_ramachandran(universe):
    """
    Compute φ/ψ per residue for all frames using MDAnalysis Ramachandran.
    Returns:
      phi: (T, K) radians
      psi: (T, K) radians
      residx: [0..K-1] abstract indices for residues with defined ϕ/ψ
    """
    prot = universe.select_atoms("protein")
    rama = Ramachandran(prot).run()
    angles_deg = rama.angles  # list length T; each is (K,2) in degrees
    T = len(angles_deg)
    if T == 0:
        fail("No frames found while computing Ramachandran angles.")
    K_min = min(arr.shape[0] for arr in angles_deg)
    phi = np.empty((T, K_min), dtype=np.float32)
    psi = np.empty((T, K_min), dtype=np.float32)
    for t, arr in enumerate(angles_deg):
        phi[t] = np.deg2rad(arr[:K_min, 0])
        psi[t] = np.deg2rad(arr[:K_min, 1])
    residx = list(range(K_min))
    return phi, psi, residx


def circular_acf(angle_series: np.ndarray, max_lag: int) -> np.ndarray:
    """Circular ACF ρ(k) = mean_t cos(θ(t+k) − θ(t)), for k=1..max_lag."""
    L = len(angle_series)
    if L <= 1:
        return np.zeros(max_lag, dtype=np.float32)
    acf = np.zeros(max_lag, dtype=np.float32)
    for k in range(1, max_lag + 1):
        x = angle_series[k:] - angle_series[:-k]
        acf[k - 1] = np.mean(np.cos(x))
    return acf


def window_decorrelation_rate(phi_win: np.ndarray, psi_win: np.ndarray, rho_star: float,
                              return_arrays: bool = False):
    """
    Compute r=1/τ per residue (via circular ACF threshold ρ*), then summarize.
    - phi_win, psi_win: (L, K)
    Returns dict with mean/median summaries and (optionally) arrays r_phi, r_psi, r_avg of length K.
    """
    L, K = phi_win.shape
    max_lag = max(1, L - 1)
    tau_phi = np.empty(K, dtype=np.float32)
    tau_psi = np.empty(K, dtype=np.float32)
    for i in range(K):
        acf_phi = circular_acf(phi_win[:, i], max_lag)
        acf_psi = circular_acf(psi_win[:, i], max_lag)
        tphi = np.where(acf_phi <= rho_star)[0]
        tpsi = np.where(acf_psi <= rho_star)[0]
        tau_phi[i] = (tphi[0] + 1) if len(tphi) else float(L)  # censor at L if not crossed
        tau_psi[i] = (tpsi[0] + 1) if len(tpsi) else float(L)
    r_phi = 1.0 / tau_phi
    r_psi = 1.0 / tau_psi
    r_avg = 0.5 * (r_phi + r_psi)
    out = {
        "r_phi_mean": float(np.nanmean(r_phi)),
        "r_phi_median": float(np.nanmedian(r_phi)),
        "r_psi_mean": float(np.nanmean(r_psi)),
        "r_psi_median": float(np.nanmedian(r_psi)),
        "r_avg_mean": float(np.nanmean(r_avg)),
        "r_avg_median": float(np.nanmedian(r_avg)),
    }
    if return_arrays:
        out["r_phi_arr"] = r_phi
        out["r_psi_arr"] = r_psi
        out["r_avg_arr"] = r_avg
    return out


# -------------------------- secondary structure (SS) via (ϕ,ψ) boxes --------------------------

def rad2deg_signed(x: np.ndarray) -> np.ndarray:
    """Convert radians to degrees in (-180, 180]."""
    d = np.rad2deg(x)
    d = (d + 180.0) % 360.0 - 180.0
    return d


def ss_masks_from_phipsi_deg(phi_deg: np.ndarray, psi_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple Ramachandran box classifier per-frame/per-residue.
    Inputs: phi_deg, psi_deg with shape (L, K) in degrees in (-180, 180].
    Returns boolean masks (L,K): is_helix, is_sheet, is_loop (loop is "else").
    Boxes (liberal):
      Helix (α):   φ ∈ [-100,-20], ψ ∈ [-80, 10]
      Sheet (β):   φ ∈ [-180,-90], ψ ∈ [ 90, 180]  (includes wrap at 180 already handled)
      Loop:        otherwise
    """
    helix = (phi_deg >= -100) & (phi_deg <= -20) & (psi_deg >= -80) & (psi_deg <= 10)
    sheet = (phi_deg >= -180) & (phi_deg <= -90) & (psi_deg >= 90) & (psi_deg <= 180)
    loop = ~(helix | sheet)
    return helix, sheet, loop


def ss_labels_window(phi_win: np.ndarray, psi_win: np.ndarray, thr_major: float = 0.5):
    """
    From window (L,K) radians, compute per-residue SS label via majority of frames.
    Returns:
      labels: np.array shape (K,), values 'H','E','L'
      frac_H, frac_E, frac_L: fractions of residues labeled as H/E/L (modal across frames)
    """
    phi_d = rad2deg_signed(phi_win)
    psi_d = rad2deg_signed(psi_win)
    helix, sheet, loop = ss_masks_from_phipsi_deg(phi_d, psi_d)  # (L,K)
    # Majority per residue:
    H_frac_res = helix.mean(axis=0)  # (K,)
    E_frac_res = sheet.mean(axis=0)
    # Loop frac is residual
    L_frac_res = 1.0 - np.maximum(H_frac_res, 0.0) - np.maximum(E_frac_res, 0.0)
    labels = np.full(H_frac_res.shape, 'L', dtype='<U1')
    labels[(H_frac_res >= thr_major) & (H_frac_res >= E_frac_res)] = 'H'
    labels[(E_frac_res >= thr_major) & (E_frac_res >  H_frac_res)] = 'E'
    frac_H = float(np.mean(labels == 'H'))
    frac_E = float(np.mean(labels == 'E'))
    frac_L = float(np.mean(labels == 'L'))
    return labels, frac_H, frac_E, frac_L


# -------------------------- globals / outcomes --------------------------

def build_s_series(rg: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Build s(t) = PC1(z(Rg), z(Q)), sign fixed so corr(s, -Q) >= 0."""
    if PCA is None or StandardScaler is None:
        fail("scikit-learn not available; install scikit-learn.")
    X = np.vstack([rg, q]).T
    Xz = StandardScaler().fit_transform(X)
    pca = PCA(n_components=1)
    s = pca.fit_transform(Xz).ravel()
    corr = np.corrcoef(s, -Xz[:, 1])[0, 1]
    if corr < 0:
        s = -s
    return s


def sliding_windows(n_frames: int, W: int, gap: int, thin: int):
    """
    Generate anchors for prev|curr|gap|out design.
    prev = [a, a+W)
    curr = [a+W, a+2W)
    gap  = [a+2W, a+2W+gap)
    out  = [a+2W+gap, a+3W+gap)
    """
    total_len = 3*W + gap
    anchors = []
    a = 0
    while a + total_len <= n_frames:
        anchors.append((a, a, a + W, a + 2*W + gap))
        a += max(thin, 1)
    return anchors


def frames_to_time_ps(idx: int, times: np.ndarray) -> float:
    if idx < 0: idx = 0
    if idx >= len(times): idx = len(times) - 1
    return float(times[idx])


def rmsf_ca(coords_win: np.ndarray) -> Dict[str, float]:
    """RMSF of Cα positions within a window; returns mean/median/quantiles across residues."""
    mu = coords_win.mean(axis=0, keepdims=True)
    dif = coords_win - mu
    msd = np.mean(np.sum(dif * dif, axis=2), axis=0)  # (N,)
    rmsf = np.sqrt(msd)
    return {
        "rmsf_mean": float(np.mean(rmsf)),
        "rmsf_median": float(np.median(rmsf)),
        "rmsf_q25": float(np.quantile(rmsf, 0.25)),
        "rmsf_q75": float(np.quantile(rmsf, 0.75)),
    }


# -------------------------- main per-replicate pipeline --------------------------

def process_replicate(prot_dir: str, out_rows: List[dict],
                      window_ps: float, gap_mult: float, thin_frac: float,
                      rho_star: float, contact_cutoff: float, verbose: bool):
    pdbs = glob(os.path.join(prot_dir, "*.pdb"))
    if not pdbs:
        fail(f"No PDB found in {prot_dir}")
    pdb_path = pdbs[0]
    pdb_chain = os.path.splitext(os.path.basename(pdb_path))[0]  # e.g., 1h02_B
    pdb_id = pdb_chain.split("_")[0]

    # Native contacts from PDB
    Uref = mda.Universe(pdb_path)
    native_pairs = build_native_contact_pairs(Uref, cutoff=contact_cutoff, min_seq_sep=3)
    ca_ref = Uref.select_atoms("name CA")
    N_ca = len(ca_ref)

    for rep in [1, 2, 3]:
        U = load_universe_for_replicate(prot_dir, pdb_path, rep)
        times_ps, rg, q = compute_global_series(U, native_pairs)
        dt_ps = get_dt_ps(U)
        s = build_s_series(rg, q)

        # Cache Cα coordinates per frame for RMSF windows
        ca = U.select_atoms("name CA")
        T = len(U.trajectory)
        ca_coords = np.empty((T, N_ca, 3), dtype=np.float32)
        for ti, _ in enumerate(U.trajectory):
            ca_coords[ti] = ca.positions

        # ϕ/ψ time series (T, K) for residues with defined angles
        phi, psi, residx = compute_ramachandran(U)
        T2, K = phi.shape
        if T2 != T:
            T = min(T, T2)
            phi = phi[:T]
            psi = psi[:T]
            times_ps = times_ps[:T]
            rg = rg[:T]
            q = q[:T]
            s = s[:T]
            ca_coords = ca_coords[:T]

        # Window sizes in frames
        W = max(2, int(round(window_ps / dt_ps)))
        gap = int(round(gap_mult * W))
        thin = max(1, int(round(thin_frac * W)))
        anchors = sliding_windows(T, W, gap, thin)
        if verbose:
            print(f"[{pdb_chain} R{rep}] T={T} frames, dt={dt_ps:.3f} ps, W={W} (~{W*dt_ps:.1f} ps), gap={gap}, thin={thin}, windows={len(anchors)}")

        for w_idx, (a, prev_s, curr_s, out_s) in enumerate(anchors):
            prev_e = prev_s + W
            curr_e = curr_s + W
            out_e  = out_s  + W
            if out_e > T:
                break

            phi_prev = phi[prev_s:prev_e]  # (W,K)
            psi_prev = psi[prev_s:prev_e]
            phi_curr = phi[curr_s:curr_e]
            psi_curr = psi[curr_s:curr_e]

            # Decorrel rates (also get per-residue arrays for SS-stratified summaries)
            r_prev = window_decorrelation_rate(phi_prev, psi_prev, rho_star=rho_star, return_arrays=True)
            r_curr = window_decorrelation_rate(phi_curr, psi_curr, rho_star=rho_star, return_arrays=True)

            # Δr summaries (overall)
            d = {
                "dr_phi_mean": r_curr["r_phi_mean"] - r_prev["r_phi_mean"],
                "dr_phi_median": r_curr["r_phi_median"] - r_prev["r_phi_median"],
                "dr_psi_mean": r_curr["r_psi_mean"] - r_prev["r_psi_mean"],
                "dr_psi_median": r_curr["r_psi_median"] - r_prev["r_psi_median"],
                "dr_avg_mean": r_curr["r_avg_mean"] - r_prev["r_avg_mean"],
                "dr_avg_median": r_curr["r_avg_median"] - r_prev["r_avg_median"],
                # raw r means for reference
                "rprev_phi_mean": r_prev["r_phi_mean"], "rcurr_phi_mean": r_curr["r_phi_mean"],
                "rprev_psi_mean": r_prev["r_psi_mean"], "rcurr_psi_mean": r_curr["r_psi_mean"],
                "rprev_avg_mean": r_prev["r_avg_mean"], "rcurr_avg_mean": r_curr["r_avg_mean"],
            }

            # Secondary structure labels via Ramachandran windows (modal per residue)
            labels_prev, ssH_prev, ssE_prev, ssL_prev = ss_labels_window(phi_prev, psi_prev, thr_major=0.5)
            labels_curr, ssH_curr, ssE_curr, ssL_curr = ss_labels_window(phi_curr, psi_curr, thr_major=0.5)
            ss_switch_frac = float(np.mean(labels_prev != labels_curr))

            # SS-stratified r summaries and deltas (based on CURRENT-window labels)
            rprev_avg = r_prev["r_avg_arr"]
            rcurr_avg = r_curr["r_avg_arr"]
            mask_H = (labels_curr == 'H')
            mask_E = (labels_curr == 'E')
            mask_L = (labels_curr == 'L')

            def mean_or_nan(x, m):
                if np.any(m):
                    return float(np.nanmean(x[m]))
                return float('nan')

            rprev_H = mean_or_nan(rprev_avg, mask_H)
            rprev_E = mean_or_nan(rprev_avg, mask_E)
            rprev_L = mean_or_nan(rprev_avg, mask_L)
            rcurr_H = mean_or_nan(rcurr_avg, mask_H)
            rcurr_E = mean_or_nan(rcurr_avg, mask_E)
            rcurr_L = mean_or_nan(rcurr_avg, mask_L)

            dr_H = rcurr_H - rprev_H if np.isfinite(rprev_H) and np.isfinite(rcurr_H) else float('nan')
            dr_E = rcurr_E - rprev_E if np.isfinite(rprev_E) and np.isfinite(rcurr_E) else float('nan')
            dr_L = rcurr_L - rprev_L if np.isfinite(rprev_L) and np.isfinite(rcurr_L) else float('nan')

            ss_feats = {
                "ss_frac_H_prev": ssH_prev, "ss_frac_E_prev": ssE_prev, "ss_frac_L_prev": ssL_prev,
                "ss_frac_H_curr": ssH_curr, "ss_frac_E_curr": ssE_curr, "ss_frac_L_curr": ssL_curr,
                "ss_switch_frac": ss_switch_frac,
                "rprev_avg_mean_H": rprev_H, "rprev_avg_mean_E": rprev_E, "rprev_avg_mean_L": rprev_L,
                "rcurr_avg_mean_H": rcurr_H, "rcurr_avg_mean_E": rcurr_E, "rcurr_avg_mean_L": rcurr_L,
                "dr_avg_mean_H": dr_H, "dr_avg_mean_E": dr_E, "dr_avg_mean_L": dr_L,
                "n_res_H_curr": int(mask_H.sum()), "n_res_E_curr": int(mask_E.sum()), "n_res_L_curr": int(mask_L.sum()),
            }

            # Coupling and RMSF in CURRENT predictor window
            dif = phi_curr - psi_curr
            C_curr = float(np.mean(np.cos(dif)))
            coords_curr = ca_coords[curr_s:curr_e]
            rmsf_stats = rmsf_ca(coords_curr)

            # Outcome on s(t): |Δs| in outcome window (end - start, abs)
            Y = abs(s[out_e - 1] - s[out_s])

            # Global means in windows (diagnostics)
            rg_prev_mean = float(np.mean(rg[prev_s:prev_e]))
            rg_curr_mean = float(np.mean(rg[curr_s:curr_e]))
            rg_out_mean  = float(np.mean(rg[out_s:out_e]))
            q_prev_mean  = float(np.mean(q[prev_s:prev_e]))
            q_curr_mean  = float(np.mean(q[curr_s:curr_e]))
            q_out_mean   = float(np.mean(q[out_s:out_e]))

            row = {
                "pdb_chain": pdb_chain,
                "pdb_id": pdb_id,
                "replicate": f"R{rep}",
                "n_frames_total": T,
                "dt_ps": dt_ps,
                "W_frames": W,
                "gap_frames": gap,
                "thin_frames": thin,
                "window_index": w_idx,
                "anchor_frame": a,
                "prev_start_ps": frames_to_time_ps(prev_s, times_ps),
                "curr_start_ps": frames_to_time_ps(curr_s, times_ps),
                "out_start_ps": frames_to_time_ps(out_s, times_ps),
                "prev_end_ps": frames_to_time_ps(prev_e - 1, times_ps),
                "curr_end_ps": frames_to_time_ps(curr_e - 1, times_ps),
                "out_end_ps": frames_to_time_ps(out_e - 1, times_ps),
                "N_ca": N_ca,
                "K_rama": K,
                "Y_abs_ds": float(Y),
                "C_curr": float(C_curr),
                "rg_prev_mean": rg_prev_mean, "rg_curr_mean": rg_curr_mean, "rg_out_mean": rg_out_mean,
                "q_prev_mean": q_prev_mean,   "q_curr_mean": q_curr_mean,   "q_out_mean": q_out_mean,
            }
            row.update(d)
            row.update(rmsf_stats)
            row.update(ss_feats)
            out_rows.append(row)


# -------------------------- entrypoint --------------------------

def main():
    ap = argparse.ArgumentParser(description="Build torsional-autocorr dataset (with SS features) from ATLAS MD.")
    ap.add_argument("--root", required=True, help="Path to ATLAS dataset root (folders like 1h02_B_protein, ...)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--window-ps", type=float, default=2000.0, help="Window length W in ps (default: 2000 ps = 2 ns)")
    ap.add_argument("--gap-mult", type=float, default=1.0, help="Gap as a multiple of W (default: 1.0)")
    ap.add_argument("--thin-frac", type=float, default=1.0, help="Thinning step as a fraction of W (default: 1.0 → step=W)")
    ap.add_argument("--rho-star", type=float, default=0.10, help="Circular ACF threshold ρ* to define τ (default: 0.10)")
    ap.add_argument("--contact-cutoff", type=float, default=8.0, help="Native contact cutoff in Å for Cα–Cα (default: 8.0)")
    ap.add_argument("--verbose", action="store_true", help="Print per-replicate info")
    args = ap.parse_args()

    if mda is None:
        fail("MDAnalysis is not installed. Please `pip install MDAnalysis` and re-run.")
    if PCA is None or StandardScaler is None:
        fail("scikit-learn is not installed. Please `pip install scikit-learn` and re-run.")

    prot_dirs = find_protein_dirs(args.root)
    rows = []
    for d in tqdm(prot_dirs, desc="Proteins"):
        try:
            process_replicate(
                prot_dir=d,
                out_rows=rows,
                window_ps=args.window_ps,
                gap_mult=args.gap_mult,
                thin_frac=args.thin_frac,
                rho_star=args.rho_star,
                contact_cutoff=args.contact_cutoff,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(d)} due to error: {e}", file=sys.stderr)

    if not rows:
        fail("No rows were produced; check inputs/parameters.")
    df = pd.DataFrame(rows)

    # Convenience standardized columns (within protein×replicate)
    df["prot_rep"] = df["pdb_chain"] + "_" + df["replicate"]
    for col in [
        "Y_abs_ds", "dr_avg_mean", "dr_phi_mean", "dr_psi_mean",
        "rmsf_mean", "C_curr",
        "ss_frac_H_curr", "ss_frac_E_curr", "ss_frac_L_curr",
        "dr_avg_mean_H", "dr_avg_mean_E", "dr_avg_mean_L"
    ]:
        if col in df.columns:
            df[col + "_z_within_rep"] = df.groupby("prot_rep")[col].transform(
                lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-12)
            )

    df.drop(columns=["prot_rep"], inplace=True)
    outdir = os.path.dirname(args.out)
    if outdir and not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[OK] Wrote {len(df):,} rows to {args.out}")


if __name__ == "__main__":
    main()
