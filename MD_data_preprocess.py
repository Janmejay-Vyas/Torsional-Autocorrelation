#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MD_data_preprocess_v3.py

Adds:
  • abs_delta_r_mean
  • step_prev_* RMSDs (prev_end → curr_end)
  • curr_pathlen_w* intrawindow drift
  • seg_ss_major (H/E/L) and seg_rmsf_tertile (low/mid/high, per protein)
Keeps:
  • Robust Δr via circular IAT (r = 1/τ_int), relaxed valid-fraction, dynamic ACF maxlag, etc.
"""

from __future__ import annotations
import os, sys, math, json, argparse, warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.dihedrals import Dihedral
    from MDAnalysis.analysis import rms, contacts
except Exception as e:
    raise RuntimeError("Requires MDAnalysis (pip install MDAnalysis).") from e

# ===== circular stats / corr (unchanged core) =====

def circ_mean(theta: np.ndarray) -> float:
    x = np.asarray(theta, float); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    return float(np.arctan2(np.nansum(np.sin(x)), np.nansum(np.cos(x))))

def circ_corr_matrix(A: np.ndarray) -> np.ndarray:
    A = np.asarray(A, float)
    A = A[np.any(np.isfinite(A), axis=1)]
    if A.ndim != 2 or A.shape[0] < 3 or A.shape[1] < 2:
        n = A.shape[1] if (A.ndim == 2) else 0
        return np.full((n, n), np.nan)
    mu = np.array([circ_mean(A[:, j]) for j in range(A.shape[1])])
    S = np.sin(A - mu[None, :]); S = np.where(np.isfinite(S), S, 0.0)
    C = S.T @ S
    d = np.sqrt(np.diag(C) + 1e-15)
    D = np.outer(d, d)
    R = np.where(D > 0, C / D, np.nan)
    R = 0.5 * (R + R.T); np.clip(R, -1.0, 1.0, out=R)
    return R

def composite_R(phi_win: np.ndarray, psi_win: np.ndarray) -> np.ndarray:
    return 0.5 * (circ_corr_matrix(phi_win) + circ_corr_matrix(psi_win))

def fisher_z(R: np.ndarray) -> np.ndarray:
    Z = np.array(R, dtype=float, copy=True)
    np.fill_diagonal(Z, 0.999)
    Z = np.clip(Z, -0.999, 0.999)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = 0.5 * np.log((1.0 + Z) / (1.0 - Z))
    np.fill_diagonal(Z, 0.0)
    return Z

def upper_tri_stats(M: np.ndarray) -> Tuple[float, float]:
    if M.ndim != 2 or M.shape[0] != M.shape[1]: return np.nan, np.nan
    iu = np.triu_indices(M.shape[0], 1)
    vals = M[iu]; vals = vals[np.isfinite(vals)]
    if vals.size == 0: return np.nan, np.nan
    return float(vals.mean()), float(vals.var(ddof=0))

def leading_eigval(M: np.ndarray) -> float:
    if not np.all(np.isfinite(M)): return np.nan
    try: return float(np.linalg.eigvalsh(M).max())
    except Exception: return np.nan

def fro_offdiag_norm(M: np.ndarray) -> float:
    if M.ndim != 2 or M.shape[0] != M.shape[1]: return np.nan
    iu = np.triu_indices(M.shape[0], 1)
    vals = M[iu]; vals = vals[np.isfinite(vals)]
    if vals.size == 0: return np.nan
    return float(np.sqrt(2.0 * np.sum(vals * vals)))

def spectral_norm_symmetric(M: np.ndarray) -> float:
    if not np.all(np.isfinite(M)): return np.nan
    try: return float(np.max(np.abs(np.linalg.eigvalsh(M))))
    except Exception: return np.nan

def ridge_shrink(R: np.ndarray, lam: float) -> np.ndarray:
    if not np.all(np.isfinite(R)) or lam <= 0: return R
    I = np.eye(R.shape[0], dtype=R.dtype)
    return (1.0 - lam) * R + lam * I

def lag1_circ_acf(theta: np.ndarray) -> float:
    x = np.asarray(theta, float); x = x[np.isfinite(x)]
    if x.size < 3: return np.nan
    return float(np.mean(np.cos(x[1:] - x[:-1])))

# ===== IAT-based rate (same as v2) =====

def circular_acf(theta: np.ndarray, maxlag: int) -> np.ndarray:
    x = np.asarray(theta, float); x = x[np.isfinite(x)]
    n = x.size
    if n < 3: return np.full(maxlag + 1, np.nan)
    rho = np.empty(maxlag + 1, float); rho[0] = 1.0
    for k in range(1, maxlag + 1):
        if k >= n: rho[k] = np.nan
        else: rho[k] = float(np.mean(np.cos(x[k:] - x[:-k])))
    return rho

def iat_rate(theta: np.ndarray, maxlag: int) -> float:
    if maxlag < 1: return np.nan
    rho = circular_acf(theta, maxlag=maxlag)
    if not np.any(np.isfinite(rho)): return np.nan
    s = 0.0
    for k in range(1, len(rho)):
        val = rho[k]
        if not np.isfinite(val): break
        if val <= 0: break
        s += val
    tau = 1.0 + 2.0 * max(0.0, s)
    if not np.isfinite(tau) or tau <= 0: return np.nan
    return float(1.0 / tau)

# ===== geometry / weights =====

from MDAnalysis.analysis import rms

def rmsd_superposed(A: np.ndarray, B: np.ndarray) -> float:
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3: return np.nan
    return float(rms.rmsd(A, B, center=True, superposition=True))

def radius_of_gyration(X: np.ndarray) -> float:
    if X.ndim != 2 or X.shape[1] != 3: return np.nan
    cm = X.mean(axis=0, keepdims=True)
    dif = X - cm
    return float(np.sqrt(np.mean(np.sum(dif * dif, axis=1))))

def weighted_kabsch_rmsd(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> float:
    if A.shape != B.shape or A.ndim != 2 or A.shape[1] != 3: return np.nan
    w = np.asarray(w, float).reshape(-1)
    if w.size != A.shape[0]: return np.nan
    w = np.clip(w, 0.0, None)
    if np.all(w == 0): return np.nan
    ws = w / (w.sum())
    Ac = A - np.sum(ws[:, None] * A, axis=0, keepdims=True)
    Bc = B - np.sum(ws[:, None] * B, axis=0, keepdims=True)
    H = (Ac * ws[:, None]).T @ Bc
    U, S, Vt = np.linalg.svd(H)
    Rm = Vt.T @ U.T
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1; Rm = Vt.T @ U.T
    A_rot = Ac @ Rm
    diff = A_rot - Bc
    mse = float(np.sum(ws * np.sum(diff * diff, axis=1)))
    return float(np.sqrt(mse))

def gaussian_weights_from_segment(CA_curr_end: np.ndarray, seg_ca_indices: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: sigma = 6.0
    seg_ca = CA_curr_end[seg_ca_indices] if len(seg_ca_indices) > 0 else CA_curr_end[:0]
    if seg_ca.shape[0] == 0:
        return np.ones(CA_curr_end.shape[0], dtype=float) / CA_curr_end.shape[0]
    centroid = seg_ca.mean(axis=0, keepdims=True)
    d2 = np.sum((CA_curr_end - centroid)**2, axis=1)
    w = np.exp(-d2 / (2.0 * sigma * sigma))
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)

# ===== torsions & SS =====

def build_phi_psi_all(u: mda.Universe) -> Tuple[np.ndarray, np.ndarray]:
    prot = u.select_atoms("protein"); residues = prot.residues; n_res = len(residues)
    phi_groups, psi_groups = [], []; phi_map, psi_map = {}, {}

    def atom_or_none(res, name):
        sel = res.atoms.select_atoms(f"name {name}")
        return sel[0] if sel.n_atoms > 0 else None

    for i, res in enumerate(residues):
        if i > 0:
            prev_res = residues[i - 1]
            C_im1 = atom_or_none(prev_res, "C")
            N_i   = atom_or_none(res, "N")
            CA_i  = atom_or_none(res, "CA")
            C_i   = atom_or_none(res, "C")
            if all(a is not None for a in (C_im1, N_i, CA_i, C_i)):
                phi_groups.append(mda.AtomGroup([C_im1, N_i, CA_i, C_i]))
                phi_map[i] = len(phi_groups) - 1
        if i < n_res - 1:
            next_res = residues[i + 1]
            N_i   = atom_or_none(res, "N")
            CA_i  = atom_or_none(res, "CA")
            C_i   = atom_or_none(res, "C")
            N_ip1 = atom_or_none(next_res, "N")
            if all(a is not None for a in (N_i, CA_i, C_i, N_ip1)):
                psi_groups.append(mda.AtomGroup([N_i, CA_i, C_i, N_ip1]))
                psi_map[i] = len(psi_groups) - 1

    phi_all = np.full((len(u.trajectory), n_res), np.nan, dtype=float)
    psi_all = np.full((len(u.trajectory), n_res), np.nan, dtype=float)

    if len(phi_groups) > 0:
        Dphi = Dihedral(phi_groups).run()
        phi_rad = np.radians(Dphi.results.angles)
        for res_idx, col_idx in phi_map.items():
            phi_all[:, res_idx] = phi_rad[:, col_idx]

    if len(psi_groups) > 0:
        Dpsi = Dihedral(psi_groups).run()
        psi_rad = np.radians(Dpsi.results.angles)
        for res_idx, col_idx in psi_map.items():
            psi_all[:, res_idx] = psi_rad[:, col_idx]

    return phi_all, psi_all

def ca_coords_for_frames(u: mda.Universe, frames: List[int]) -> np.ndarray:
    CA = u.select_atoms("protein and name CA")
    out = np.empty((len(frames), CA.n_atoms, 3), dtype=float)
    for i, f in enumerate(frames):
        u.trajectory[f]; out[i] = CA.positions
    return out

def ss_from_phipsi(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    phid = np.degrees(phi); psid = np.degrees(psi)
    H = ((phid >= -100) & (phid <= -30) & (psid >= -80) & (psid <= -15))
    E = ((phid >= -180) & (phid <= -80) & (psid >= 90) & (psid <= 180))
    out = np.full(phid.shape, 'L', dtype='<U1'); out[E] = 'E'; out[H] = 'H'
    return out

def seg_ss_fractions(phi_win: np.ndarray, psi_win: np.ndarray) -> Tuple[float, float, float]:
    lab = ss_from_phipsi(phi_win, psi_win)
    total = lab.size
    if total == 0: return np.nan, np.nan, np.nan
    return float(np.mean(lab == 'H')), float(np.mean(lab == 'E')), float(np.mean(lab == 'L'))

def protein_ss_fractions(phi_win_all: np.ndarray, psi_win_all: np.ndarray) -> Tuple[float, float, float]:
    lab = ss_from_phipsi(phi_win_all, psi_win_all)
    if lab.size == 0: return np.nan, np.nan, np.nan
    return float(np.mean(lab == 'H')), float(np.mean(lab == 'E')), float(np.mean(lab == 'L'))

def segment_rmsf_curr(coords_curr: np.ndarray) -> float:
    if coords_curr.ndim != 3 or coords_curr.shape[2] != 3 or coords_curr.shape[0] < 2: return np.nan
    mean_pos = coords_curr.mean(axis=0, keepdims=True)
    diffs = coords_curr - mean_pos
    rmsf_per_res = np.sqrt((diffs**2).sum(axis=2).mean(axis=0))
    return float(np.mean(rmsf_per_res))

# ===== params / helpers =====

@dataclass
class Params:
    ridge_lambda: float = 0.05
    min_L_eff: int = 4
    smooth_width: int = 3
    acf_maxlag: int = 300
    valid_frac: float = 0.70
    compute_s: bool = False
    native_mode: str = "pdb"
    contacts_cutoff: float = 8.0
    debug: bool = False

def valid_cols_frac(prevA: np.ndarray, currA: np.ndarray, frac: float) -> np.ndarray:
    return (np.isfinite(prevA).mean(axis=0) >= frac) & (np.isfinite(currA).mean(axis=0) >= frac)

def smooth_within_protein(df: pd.DataFrame, col: str, newcol: str, width: int) -> pd.DataFrame:
    df = df.copy(); df[newcol] = np.nan
    for _, g in df.groupby(["protein_id"], sort=False):
        g = g.sort_values(["segment_id","replicate","window_index"])
        idx, vals = [], []
        for _, gg in g.groupby(["replicate","segment_id"], sort=False):
            sm = gg[col].astype(float).rolling(window=width, center=True, min_periods=1).median()
            idx.append(gg.index.values); vals.append(sm.values)
        if idx:
            df.loc[np.concatenate(idx), newcol] = np.concatenate(vals)
    return df

def compute_group_acfs(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["protein_id","replicate","segment_id"]
    df = df.sort_values(keys + ["window_index"]).copy()
    acf_map = {
        "acf1_step_by_seg": "step_global_abs",
        "acf1_dcorr_mean_by_seg": "d_seg_corr_mean",
        "acf1_dcorr_fro_by_seg": "d_seg_corr_fro",
        "acf1_dcorr_fro_z_by_seg": "d_seg_corr_fro_z",
        "acf1_dcorr_spec_by_seg": "d_seg_corr_spec",
    }
    def acf_1d(x, maxlag=2):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        n = x.size
        if n < 4: return np.array([np.nan]*(maxlag+1))
        x = x - x.mean()
        den = np.dot(x, x) + 1e-15
        return np.array([1.0 if k==0 else np.dot(x[:-k], x[k:]) / den for k in range(maxlag+1)])
    for key, g in df.groupby(keys, sort=False):
        for outcol, src in acf_map.items():
            ac = acf_1d(g[src].to_numpy()) if src in g else np.array([np.nan, np.nan, np.nan])
            df.loc[g.index, outcol] = ac[1] if ac.size > 1 else np.nan
    return df

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Preprocess raw MD CSV → final feature dataset (v3).")
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ridge-lambda", type=float, default=0.05)
    ap.add_argument("--min-L-eff", type=int, default=4)
    ap.add_argument("--smooth-width", type=int, default=3)
    ap.add_argument("--acf-maxlag", type=int, default=300)
    ap.add_argument("--valid-frac", type=float, default=0.70)
    ap.add_argument("--compute-s", action="store_true")
    ap.add_argument("--native-mode", choices=["pdb","first_frame"], default="pdb")
    ap.add_argument("--contacts-cutoff", type=float, default=8.0)
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    P = Params(args.ridge_lambda, args.min_L_eff, args.smooth_width, args.acf_maxlag,
               args.valid_frac, args.compute_s, args.native_mode, args.contacts_cutoff, args.debug)

    raw = pd.read_csv(args.raw)
    rows_out: List[Dict[str, float]] = []
    dropped_torsions = dropped_CA = 0

    rep_keys = ["pdb_chain","replicate","xtc_path","pdb_path","tpr_path","protein_id"]
    for rep_vals, g in raw.groupby(rep_keys):
        pdb_chain, replicate, xtc_path, pdb_path, tpr_path, protein_id = rep_vals
        print(f"[PREP] {pdb_chain} {replicate}", file=sys.stderr)

        u = mda.Universe(pdb_path if os.path.exists(pdb_path) else tpr_path, xtc_path)
        residues = u.select_atoms("protein").residues; n_res = len(residues)
        CA = u.select_atoms("protein and name CA")

        # torsions (once per replicate)
        phi_all, psi_all = build_phi_psi_all(u)

        # optional s(t)
        native_pairs = np.empty((0,2), dtype=int); ref_CA = None
        if P.compute_s:
            if P.native_mode == "pdb" and os.path.exists(pdb_path):
                ref = mda.Universe(pdb_path); ref_CA = ref.select_atoms("protein and name CA")
            else:
                u.trajectory[0]; ref_CA = CA.copy()
            ref_dmat = contacts.distance_array(ref_CA.positions, ref_CA.positions)
            iu = np.triu_indices(ref_CA.n_atoms, 1)
            mask = (ref_dmat < P.contacts_cutoff)
            native_pairs = np.argwhere(mask)

        for _, rr in g.iterrows():
            prev_frames = list(range(int(rr["prev_start_f"]), int(rr["prev_end_f"])))
            curr_frames = list(range(int(rr["curr_start_f"]), int(rr["curr_end_f"])))
            out_frames  = list(range(int(rr["out_start_f"]),  int(rr["out_end_f"])))
            prev_last = prev_frames[-1]; curr_last = curr_frames[-1]; out_last = out_frames[-1]
            out2_last = int(rr["out2_end_f"])
            if out2_last >= len(u.trajectory): out2_last = None

            s0 = int(rr["segment_start_index"]); s1 = int(rr["segment_end_index"])
            seg_res_idx = np.arange(s0, s1 + 1)
            ca_idx_all = np.array(json.loads(rr["segment_ca_indices_json"]), dtype=int)

            # torsion windows
            phi_prev = phi_all[prev_frames][:, seg_res_idx]; psi_prev = psi_all[prev_frames][:, seg_res_idx]
            phi_curr = phi_all[curr_frames][:, seg_res_idx]; psi_curr = psi_all[curr_frames][:, seg_res_idx]

            both_valid = valid_cols_frac(phi_prev, phi_curr, P.valid_frac) & valid_cols_frac(psi_prev, psi_curr, P.valid_frac)
            if both_valid.sum() < P.min_L_eff:
                dropped_torsions += 1; continue

            idx_both = np.flatnonzero(both_valid)
            phi_prev_v, psi_prev_v = phi_prev[:, idx_both], psi_prev[:, idx_both]
            phi_curr_v, psi_curr_v = phi_curr[:, idx_both], psi_curr[:, idx_both]
            L_eff = phi_curr_v.shape[1]

            mask_reduced = (ca_idx_all[idx_both] >= 0)
            if mask_reduced.sum() < P.min_L_eff:
                dropped_CA += 1; continue
            seg_ca_indices = ca_idx_all[idx_both][mask_reduced]

            # IAT-based rate (median across residues) with lag-1 fallback
            def agg_rate_IAT(mat_prev, mat_curr):
                T_prev, T_curr = mat_prev.shape[0], mat_curr.shape[0]
                Kp = max(1, min(P.acf_maxlag, T_prev - 2))
                Kc = max(1, min(P.acf_maxlag, T_curr - 2))
                rps, rcs = [], []
                for j in range(mat_prev.shape[1]):
                    rp = iat_rate(mat_prev[:, j], Kp); rc = iat_rate(mat_curr[:, j], Kc)
                    if np.isfinite(rp): rps.append(rp)
                    if np.isfinite(rc): rcs.append(rc)
                def med(v):
                    v = np.asarray(v, float); v = v[np.isfinite(v)]
                    return float(np.median(v)) if v.size else np.nan
                r_prev_m, r_curr_m = med(rps), med(rcs)
                if not np.isfinite(r_prev_m):
                    r1s = [lag1_circ_acf(mat_prev[:, j]) for j in range(mat_prev.shape[1])]
                    r_prev_m = float(np.nanmedian(np.maximum(0.0, 1.0 - np.asarray(r1s))))
                if not np.isfinite(r_curr_m):
                    r1s = [lag1_circ_acf(mat_curr[:, j]) for j in range(mat_curr.shape[1])]
                    r_curr_m = float(np.nanmedian(np.maximum(0.0, 1.0 - np.asarray(r1s))))
                return r_prev_m, r_curr_m, (r_curr_m - r_prev_m)

            r_phi_prev, r_phi_curr, delta_r_phi = agg_rate_IAT(phi_prev_v, phi_curr_v)
            r_psi_prev, r_psi_curr, delta_r_psi = agg_rate_IAT(psi_prev_v, psi_curr_v)
            delta_r_mean = np.nanmean([delta_r_phi, delta_r_psi])
            abs_delta_r_mean = float(abs(delta_r_mean)) if np.isfinite(delta_r_mean) else np.nan

            # correlations / deltas
            R_prev = composite_R(phi_prev_v, psi_prev_v)
            R_curr = composite_R(phi_curr_v, psi_curr_v)
            if P.ridge_lambda > 0:
                R_prev = ridge_shrink(R_prev, P.ridge_lambda)
                R_curr = ridge_shrink(R_curr, P.ridge_lambda)
            dR = R_curr - R_prev
            d_seg_corr_fro  = fro_offdiag_norm(dR)
            d_seg_corr_spec = spectral_norm_symmetric(dR)
            R_prev_z, R_curr_z = fisher_z(R_prev), fisher_z(R_curr)
            dR_z = R_curr_z - R_prev_z
            d_seg_corr_fro_z  = fro_offdiag_norm(dR_z)
            d_seg_corr_spec_z = spectral_norm_symmetric(dR_z)
            mu_prev, var_prev = upper_tri_stats(R_prev); mu_curr, var_curr = upper_tri_stats(R_curr)
            leig_prev = leading_eigval(R_prev); leig_curr = leading_eigval(R_curr)

            nn_prev = [R_prev[i, i+1] for i in range(L_eff-1) if np.isfinite(R_prev[i, i+1])]
            nn_curr = [R_curr[i, i+1] for i in range(L_eff-1) if np.isfinite(R_curr[i, i+1])]
            seg_corr_nn_prev = float(np.mean(nn_prev)) if nn_prev else np.nan
            seg_corr_nn_curr = float(np.mean(nn_curr)) if nn_curr else np.nan

            d_seg_corr_mean = float(mu_curr - mu_prev)
            d_seg_corr_nn   = float(seg_corr_nn_curr - seg_corr_nn_prev)
            d_seg_corr_leig = float(leig_curr - leig_prev)
            d_seg_corr_var  = float(var_curr - var_prev)

            # φ↔ψ coupling averages
            def avg_coupling(phiW, psiW):
                vals = []
                for j in range(phiW.shape[1]):
                    s1 = np.sin(phiW[:, j] - circ_mean(phiW[:, j]))
                    s2 = np.sin(psiW[:, j] - circ_mean(psiW[:, j]))
                    if np.isfinite(s1).sum() > 2 and np.isfinite(s2).sum() > 2:
                        num = float(np.nansum(s1 * s2))
                        den = math.sqrt(float(np.nansum(s1*s1)) * float(np.nansum(s2*s2))) + 1e-15
                        rho = num / den
                        if np.isfinite(rho): vals.append(rho)
                return float(np.nanmean(vals)) if vals else np.nan
            seg_phi_psi_coupling_prev = avg_coupling(phi_prev_v, psi_prev_v)
            seg_phi_psi_coupling_curr = avg_coupling(phi_curr_v, psi_curr_v)

            # coordinates (endpoints)
            u.trajectory[prev_last]; CA_prev_end = CA.positions.copy()
            u.trajectory[curr_last]; CA_curr_end = CA.positions.copy()
            u.trajectory[out_last];  CA_out_end  = CA.positions.copy()
            CA_out2_end = None
            if out2_last is not None:
                u.trajectory[out2_last]; CA_out2_end = CA.positions.copy()

            # segment RMSF in current window
            coords_curr_seg = ca_coords_for_frames(u, curr_frames)[:, seg_ca_indices, :]
            seg_rmsf_mean = segment_rmsf_curr(coords_curr_seg)

            # SS fractions & majority for seg/protein
            phi_curr_valid = phi_curr_v[:, :]
            psi_curr_valid = psi_curr_v[:, :]
            seg_frac_H, seg_frac_E, seg_frac_L = seg_ss_fractions(phi_curr_valid, psi_curr_valid)

            phi_curr_all = phi_all[curr_frames, :]
            psi_curr_all = psi_all[curr_frames, :]
            prot_frac_H, prot_frac_E, prot_frac_L = protein_ss_fractions(phi_curr_all, psi_curr_all)
            p_ss = np.array([seg_frac_H, seg_frac_E, seg_frac_L], float)
            p_ss = np.clip(p_ss, 1e-12, 1.0); p_ss /= p_ss.sum()
            seg_ss_entropy = float(-(p_ss * np.log(p_ss)).sum())
            ss_labels = np.array(['H','E','L'])
            seg_ss_major = ss_labels[int(np.argmax([seg_frac_H, seg_frac_E, seg_frac_L]))]

            # outcomes & new steps
            step_global_abs = rmsd_superposed(CA_out_end, CA_curr_end)
            # previous→current (new)
            step_prev_abs = rmsd_superposed(CA_curr_end, CA_prev_end)

            rg_curr_end = radius_of_gyration(CA_curr_end)
            rg_out_end  = radius_of_gyration(CA_out_end)
            d_rmsd_init = float(rmsd_superposed(CA_out_end, ca_coords_for_frames(u, [0])[0])
                                - rmsd_superposed(CA_curr_end, ca_coords_for_frames(u, [0])[0]))
            abs_d_rg    = float(abs(rg_out_end - rg_curr_end))

            # weighted RMSDs
            w4  = gaussian_weights_from_segment(CA_curr_end, seg_ca_indices, sigma=4.0)
            w6  = gaussian_weights_from_segment(CA_curr_end, seg_ca_indices, sigma=6.0)
            w10 = gaussian_weights_from_segment(CA_curr_end, seg_ca_indices, sigma=10.0)

            step_w4_abs  = weighted_kabsch_rmsd(CA_out_end,  CA_curr_end, w4)
            step_w6_abs  = weighted_kabsch_rmsd(CA_out_end,  CA_curr_end, w6)
            step_w10_abs = weighted_kabsch_rmsd(CA_out_end,  CA_curr_end, w10)

            # previous→current weighted (new)
            step_prev_w4_abs  = weighted_kabsch_rmsd(CA_curr_end, CA_prev_end, w4)
            step_prev_w6_abs  = weighted_kabsch_rmsd(CA_curr_end, CA_prev_end, w6)
            step_prev_w10_abs = weighted_kabsch_rmsd(CA_curr_end, CA_prev_end, w10)

            # 12Å mask
            seg_centroid = CA_curr_end[seg_ca_indices].mean(axis=0, keepdims=True)
            d2 = np.sum((CA_curr_end - seg_centroid)**2, axis=1)
            mask12 = (np.sqrt(d2) <= 12.0)
            step_mask12_abs      = rmsd_superposed(CA_out_end[mask12],  CA_curr_end[mask12])
            step_prev_mask12_abs = rmsd_superposed(CA_curr_end[mask12], CA_prev_end[mask12])

            # out path length (as before)
            CA_out_traj = ca_coords_for_frames(u, out_frames) if len(out_frames) > 1 else None
            out_pathlen_w4 = out_pathlen_w6 = np.nan
            if CA_out_traj is not None and CA_out_traj.shape[0] >= 2:
                steps4, steps6 = [], []
                for t in range(1, CA_out_traj.shape[0]):
                    steps4.append(weighted_kabsch_rmsd(CA_out_traj[t], CA_out_traj[t-1], w4))
                    steps6.append(weighted_kabsch_rmsd(CA_out_traj[t], CA_out_traj[t-1], w6))
                out_pathlen_w4 = float(np.nansum(steps4)); out_pathlen_w6 = float(np.nansum(steps6))

            # NEW: curr window drift (path length inside current window)
            CA_curr_traj = ca_coords_for_frames(u, curr_frames) if len(curr_frames) > 1 else None
            curr_pathlen_w4 = curr_pathlen_w6 = np.nan
            if CA_curr_traj is not None and CA_curr_traj.shape[0] >= 2:
                steps4c, steps6c = [], []
                for t in range(1, CA_curr_traj.shape[0]):
                    steps4c.append(weighted_kabsch_rmsd(CA_curr_traj[t], CA_curr_traj[t-1], w4))
                    steps6c.append(weighted_kabsch_rmsd(CA_curr_traj[t], CA_curr_traj[t-1], w6))
                curr_pathlen_w4 = float(np.nansum(steps4c)); curr_pathlen_w6 = float(np.nansum(steps6c))

            # horizon-2 (unchanged)
            step_global_abs_h2 = d_rmsd_init_h2 = abs_d_rg_h2 = np.nan
            step_w4_abs_h2 = step_w6_abs_h2 = step_w10_abs_h2 = step_mask12_abs_h2 = np.nan
            if CA_out2_end is not None:
                step_global_abs_h2 = rmsd_superposed(CA_out2_end, CA_curr_end)
                CA_init = ca_coords_for_frames(u, [0])[0]
                d_rmsd_init_h2 = float(rmsd_superposed(CA_out2_end, CA_init) - rmsd_superposed(CA_curr_end, CA_init))
                rg_out2_end = radius_of_gyration(CA_out2_end)
                abs_d_rg_h2 = float(abs(rg_out2_end - rg_curr_end))
                step_w4_abs_h2  = weighted_kabsch_rmsd(CA_out2_end, CA_curr_end, w4)
                step_w6_abs_h2  = weighted_kabsch_rmsd(CA_out2_end, CA_curr_end, w6)
                step_w10_abs_h2 = weighted_kabsch_rmsd(CA_out2_end, CA_curr_end, w10)
                step_mask12_abs_h2 = rmsd_superposed(CA_out2_end[mask12], CA_curr_end[mask12])

            row = {
                "protein_id": protein_id, "pdb_chain": pdb_chain, "replicate": replicate,
                "segment_id": rr["segment_id"], "segment_start_resid": int(rr["segment_start_resid"]),
                "segment_end_resid": int(rr["segment_end_resid"]), "segment_len": int(rr["segment_len"]),
                "window_index": int(rr["window_index"]), "dt_ps": float(rr["dt_ps"]), "W_ps": float(rr["W_ps"]),
                "W_frames": int(rr["W_frames"]), "gap_ps": float(rr["gap_ps"]), "gap_frames": int(rr["gap_frames"]),
                "prev_start_ps": float(rr["prev_start_ps"]), "prev_end_ps": float(rr["prev_end_ps"]),
                "curr_start_ps": float(rr["curr_start_ps"]), "curr_end_ps": float(rr["curr_end_ps"]),
                "out_start_ps": float(rr["out_start_ps"]), "out_end_ps": float(rr["out_end_ps"]),
                "out2_end_ps": float(rr["out2_end_ps"]),
                "r_phi_prev": float(r_phi_prev), "r_phi_curr": float(r_phi_curr),
                "r_psi_prev": float(r_psi_prev), "r_psi_curr": float(r_psi_curr),
                "delta_r_phi": float(delta_r_phi), "delta_r_psi": float(delta_r_psi),
                "delta_r_mean": float(delta_r_mean), "abs_delta_r_mean": float(abs_delta_r_mean),
                "d_seg_corr_mean": float(d_seg_corr_mean), "d_seg_corr_nn": float(d_seg_corr_nn),
                "d_seg_corr_leig": float(d_seg_corr_leig), "d_seg_corr_var": float(d_seg_corr_var),
                "d_seg_corr_fro": float(d_seg_corr_fro), "d_seg_corr_spec": float(d_seg_corr_spec),
                "d_seg_corr_fro_z": float(d_seg_corr_fro_z), "d_seg_corr_spec_z": float(d_seg_corr_spec_z),
                "seg_corr_mean_prev": float(mu_prev), "seg_corr_mean_curr": float(mu_curr),
                "seg_corr_leig_prev": float(leig_prev), "seg_corr_leig_curr": float(leig_curr),
                "seg_phi_psi_coupling_prev": float(seg_phi_psi_coupling_prev),
                "seg_phi_psi_coupling_curr": float(seg_phi_psi_coupling_curr),
                "seg_L_eff_curr": float(L_eff),
                "seg_rmsf_mean": float(seg_rmsf_mean),
                "seg_frac_H": float(seg_frac_H), "seg_frac_E": float(seg_frac_E), "seg_frac_L": float(seg_frac_L),
                "seg_ss_entropy": float(seg_ss_entropy), "seg_ss_major": seg_ss_major,
                "prot_frac_H": float(prot_frac_H), "prot_frac_E": float(prot_frac_E), "prot_frac_L": float(prot_frac_L),
                "protein_length": int(n_res),
                "step_global_abs": float(step_global_abs),
                "step_prev_abs": float(step_prev_abs),
                "rmsd_init_curr": float(rmsd_superposed(CA_curr_end, ca_coords_for_frames(u, [0])[0])),
                "rmsd_init_out": float(rmsd_superposed(CA_out_end,  ca_coords_for_frames(u, [0])[0])),
                "d_rmsd_init": float(d_rmsd_init),
                "rg_curr_end": float(rg_curr_end), "rg_out_end": float(rg_out_end), "abs_d_rg": float(abs_d_rg),
                "step_w4_abs": float(step_w4_abs), "step_w6_abs": float(step_w6_abs), "step_w10_abs": float(step_w10_abs),
                "step_prev_w4_abs": float(step_prev_w4_abs), "step_prev_w6_abs": float(step_prev_w6_abs), "step_prev_w10_abs": float(step_prev_w10_abs),
                "step_mask12_abs": float(step_mask12_abs), "step_prev_mask12_abs": float(step_prev_mask12_abs),
                "curr_pathlen_w4": float(curr_pathlen_w4), "curr_pathlen_w6": float(curr_pathlen_w6),
                "out_pathlen_w4": float(out_pathlen_w4), "out_pathlen_w6": float(out_pathlen_w6),
                "step_global_abs_h2": float(step_global_abs_h2),
                "d_rmsd_init_h2": float(d_rmsd_init_h2), "abs_d_rg_h2": float(abs_d_rg_h2),
                "step_w4_abs_h2": float(step_w4_abs_h2), "step_w6_abs_h2": float(step_w6_abs_h2),
                "step_w10_abs_h2": float(step_w10_abs_h2), "step_mask12_abs_h2": float(step_mask12_abs_h2),
                "time_ok": True, "gap_ok": True,
            }
            rows_out.append(row)

    if not rows_out:
        raise RuntimeError("No rows produced; relax --valid-frac / --min-L-eff, or check raw windows.")

    df = pd.DataFrame(rows_out)

    # ACF diagnostics
    try:
        df = compute_group_acfs(df)
    except Exception as e:
        print(f"[WARN] ACF compute failed: {e}", file=sys.stderr)

    # Smooth ΔR Fro norms (unchanged)
    if "d_seg_corr_fro" in df.columns:
        df = smooth_within_protein(df, "d_seg_corr_fro", "d_seg_corr_fro_sm3", P.smooth_width)
        df["abs_d_seg_corr_fro_sm3"] = df["d_seg_corr_fro_sm3"].abs()
    if "d_seg_corr_fro_z" in df.columns:
        df = smooth_within_protein(df, "d_seg_corr_fro_z", "d_seg_corr_fro_z_sm3", P.smooth_width)
        df["abs_d_seg_corr_fro_z_sm3"] = df["d_seg_corr_fro_z_sm3"].abs()

    # RMSF tertiles per protein
    df["seg_rmsf_tertile"] = np.nan
    for pid, g in df.groupby("protein_id", sort=False):
        q = np.nanquantile(g["seg_rmsf_mean"], [1/3, 2/3])
        lab = pd.cut(g["seg_rmsf_mean"], bins=[-np.inf, q[0], q[1], np.inf], labels=["low","mid","high"], include_lowest=True)
        df.loc[g.index, "seg_rmsf_tertile"] = lab.astype(str)

    # order
    prefer = [
        "protein_id","pdb_chain","replicate","segment_id","segment_start_resid","segment_end_resid","segment_len",
        "window_index","dt_ps","W_ps","W_frames","gap_ps","gap_frames",
        "prev_start_ps","prev_end_ps","curr_start_ps","curr_end_ps","out_start_ps","out_end_ps","out2_end_ps",
        "r_phi_prev","r_phi_curr","r_psi_prev","r_psi_curr","delta_r_phi","delta_r_psi","delta_r_mean","abs_delta_r_mean",
        "d_seg_corr_mean","d_seg_corr_nn","d_seg_corr_leig","d_seg_corr_var","d_seg_corr_fro","d_seg_corr_spec",
        "d_seg_corr_fro_z","d_seg_corr_spec_z","d_seg_corr_fro_sm3","abs_d_seg_corr_fro_sm3","d_seg_corr_fro_z_sm3","abs_d_seg_corr_fro_z_sm3",
        "seg_corr_mean_prev","seg_corr_mean_curr","seg_corr_leig_prev","seg_corr_leig_curr",
        "seg_phi_psi_coupling_prev","seg_phi_psi_coupling_curr","seg_L_eff_curr",
        "seg_rmsf_mean","seg_rmsf_tertile","seg_frac_H","seg_frac_E","seg_frac_L","seg_ss_major","seg_ss_entropy",
        "prot_frac_H","prot_frac_E","prot_frac_L","protein_length",
        "step_prev_abs","step_prev_w4_abs","step_prev_w6_abs","step_prev_w10_abs","step_prev_mask12_abs",
        "curr_pathlen_w4","curr_pathlen_w6",
        "step_global_abs","rmsd_init_curr","rmsd_init_out","d_rmsd_init","rg_curr_end","rg_out_end","abs_d_rg",
        "step_w4_abs","step_w6_abs","step_w10_abs","step_mask12_abs","out_pathlen_w4","out_pathlen_w6",
        "step_global_abs_h2","d_rmsd_init_h2","abs_d_rg_h2","step_w4_abs_h2","step_w6_abs_h2","step_w10_abs_h2","step_mask12_abs_h2",
        "time_ok","gap_ok",
        "acf1_step_by_seg","acf1_dcorr_mean_by_seg","acf1_dcorr_fro_by_seg","acf1_dcorr_fro_z_by_seg","acf1_dcorr_spec_by_seg",
    ]
    cols = [c for c in prefer if c in df.columns] + [c for c in df.columns if c not in prefer]
    df = df[cols]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    if P.debug:
        print(f"[OK] wrote {args.out} rows={len(df)} | dropped_torsions={dropped_torsions} | dropped_CA={dropped_CA}", file=sys.stderr)
    else:
        print(f"[OK] wrote {args.out} rows={len(df)}", file=sys.stderr)

if __name__ == "__main__":
    main()
