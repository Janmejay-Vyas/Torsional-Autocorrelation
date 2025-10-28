#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_MD_data.py
This script parses molecular dynamics simulation output files and extracts relevant data such as conformational windows, RMSD, and torsional angles for each frame.
"""

# Importing the required libraries
from __future__ import annotations

import os 
import re
import sys
import json
import argparse
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda

@dataclass
class ReplicateFiles:
    """
    Simple container for a replicate's file paths, and IDs. 
    """
    pdb: str
    tpr: str
    xtc: str
    protein_id: str
    pdb_chain: str
    replicate: str


def find_replicates(root: str) -> List[ReplicateFiles]:
    """
    Scans the root directory for protein replicates and returns a list of ReplicateFile objects
    """
    reps: List[ReplicateFiles] = []
    for d in sorted(os.listdir(root)):
        dpath = os.path.join(root, d)
        if not os.path.isdir(dpath) or not d.endswith("_protein"):
            continue

        pdb_chain = d.replace("_protein", "")
        pdb = os.path.join(dpath, f"{pdb_chain}.pdb")
        if not os.path.exists(pdb):
            raise FileNotFoundError(f"Expected PDB not found for {pdb_chain} under {dpath}")

        for f in sorted(os.listdir(dpath)):
            m = re.match(rf"^{re.escape(pdb_chain)}_prod_(R\d+)_fit\.xtc$", f)
            if not m:
                continue
            rep = m.group(1)
            xtc = os.path.join(dpath, f)

            # TPR fallback naming conventions seen in ATLAS
            tpr1 = os.path.join(dpath, f"{pdb_chain}_prod_{rep}.tpr")
            tpr2 = os.path.join(dpath, f"{pdb_chain}_{rep}.tpr")
            tpr = tpr1 if os.path.exists(tpr1) else (tpr2 if os.path.exists(tpr2) else "")

            protein_id = pdb_chain.split("_")[0]
            reps.append(ReplicateFiles(
                pdb=pdb, tpr=tpr, xtc=xtc,
                protein_id=protein_id, pdb_chain=pdb_chain, replicate=rep
            ))

    if not reps:
        raise ValueError(f"No replicates found under: {root}")
    return reps


def estimate_dt_ps(u: mda.Universe) -> float:
    """
    Estimate frame spacing (in picoseconds) from the first two frames 
    """
    try:
        t0 = None
        for i, ts in enumerate(u.trajectory):
            if i == 0:
                t0 = float(ts.time)
            else:
                dt = float(ts.time) - float(t0)
                if dt > 0:
                    return dt
    except Exception as e:
        print(f"Warning: Could not estimate frame spacing: {e}", file=sys.stderr)
        pass
    return 10.0 # Default to 10 ps if unable to estimate


def segment_indices(n_res: int, L: int, stride: int) -> List[Tuple[int, int]]:
    """
    Compute [start, end] (inclusive) residue-index windows of length L with a given stride
    """
    return [(s, s + L - 1) for s in range(0, max(0, n_res - L + 1), stride)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Extract raw window/segment geometry from ATLAS MD (no transforms).")
    ap.add_argument("--root", required=True, help="Directory containing *_protein/ folders.")
    ap.add_argument("--out", required=True, help="Output CSV path; script appends _raw_dataset.csv if missing.")
    # Time geometry
    ap.add_argument("--window-ps", type=float, default=2000.0, help="Window length in ps (default: 2000).")
    ap.add_argument("--gap-mult", type=float, default=1.0, help="Gap as multiple of window length (default: 1.0).")
    ap.add_argument("--step-mult", type=float, default=1.0, help="Step as multiple of window length (default: 1.0).")
    # Segment geometry
    ap.add_argument("--seg-len", type=int, default=11, help="Residues per segment (default: 11).")
    ap.add_argument("--seg-stride", type=int, default=3, help="Residue stride (default: 3).")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    reps = find_replicates(args.root)

    rows = []
    for r in reps:
        print(f"[RAW] Scanning {r.pdb_chain} {r.replicate}", file=sys.stderr)

        # Load once per replicate to probe dt, residue count, and CA indices.
        u = mda.Universe(r.pdb, r.xtc) if os.path.exists(r.pdb) else mda.Universe(r.tpr, r.xtc)
        dt_ps = estimate_dt_ps(u)
        n_frames = len(u.trajectory)

        residues = u.select_atoms("protein").residues
        n_res = len(residues)
        segs = segment_indices(n_res, args.seg_len, args.seg_stride)

        # Map absolute atom index → CA index in the “protein CA” selection.
        CA = u.select_atoms("protein and name CA")
        abs_to_capos = {atom.index: i for i, atom in enumerate(CA.atoms)}

        # Per-residue CA position (or -1 if missing). This is saved as JSON to avoid recompute.
        ca_pos_per_res = []
        for res in residues:
            sel = res.atoms.select_atoms("name CA")
            ca_pos_per_res.append(abs_to_capos.get(sel[0].index, -1) if sel.n_atoms == 1 else -1)
        ca_pos_per_res = np.array(ca_pos_per_res, dtype=int)

        # Convert time geometry to frames. Using integers ensures exact slicing later.
        Wf = max(10, int(round(args.window_ps / dt_ps)))
        gapf = int(round(args.gap_mult * Wf))
        stepf= int(round(args.step_mult * Wf))

        win = 0
        i = 0
        while True:
            prev_start = i
            prev_end = i + Wf
            curr_start = prev_end + gapf
            curr_end = curr_start + Wf
            out_start = curr_end + gapf
            out_end = out_start + Wf
            out2_end = out_end + Wf  # may exceed trajectory; that’s fine at raw stage

            # Stop when we can’t form curr/out windows
            if curr_end > n_frames or out_end > n_frames:
                break

            for s0, s1 in segs:
                seg_start_resid = int(residues[s0].resid)
                seg_end_resid = int(residues[s1].resid)
                seg_ca_idx = ca_pos_per_res[slice(s0, s1+1)].tolist()  # may include -1 entries

                rows.append({
                    # Identity
                    "protein_id": r.protein_id,
                    "pdb_chain": r.pdb_chain,
                    "replicate": r.replicate,

                    # File paths (for deterministic reopen later)
                    "pdb_path": os.path.abspath(r.pdb) if r.pdb else "",
                    "tpr_path": os.path.abspath(r.tpr) if r.tpr else "",
                    "xtc_path": os.path.abspath(r.xtc),

                    # Segment geometry (both index-space and resid labels)
                    "segment_id": f"{seg_start_resid}-{seg_end_resid}",
                    "segment_start_index": s0,              # 0-based index in residues array
                    "segment_end_index": s1,
                    "segment_start_resid": seg_start_resid, # human-friendly label
                    "segment_end_resid": seg_end_resid,
                    "segment_len": int(args.seg_len),
                    "segment_ca_indices_json": json.dumps(seg_ca_idx),

                    # Time geometry in frames (authoritative for slicing)
                    "window_index": win,
                    "W_frames": int(Wf),
                    "gap_frames": int(gapf),
                    "step_mult": float(args.step_mult),

                    "prev_start_f": int(prev_start),
                    "prev_end_f": int(prev_end),
                    "curr_start_f": int(curr_start),
                    "curr_end_f": int(curr_end),
                    "out_start_f": int(out_start),
                    "out_end_f": int(out_end),
                    "out2_end_f": int(out2_end),

                    # Redundant (nice-to-have) times in ps (for QA/plots)
                    "dt_ps": float(dt_ps),
                    "W_ps": float(args.window_ps),
                    "gap_ps": float(args.gap_mult * args.window_ps),
                    "prev_start_ps": float(prev_start * dt_ps),
                    "prev_end_ps": float(prev_end * dt_ps),
                    "curr_start_ps": float(curr_start * dt_ps),
                    "curr_end_ps": float(curr_end * dt_ps),
                    "out_start_ps": float(out_start * dt_ps),
                    "out_end_ps": float(out_end * dt_ps),
                    "out2_end_ps": float(out2_end * dt_ps),
                })

            win += 1
            i += stepf

    df = pd.DataFrame(rows)
    out_csv = args.out if args.out.endswith("_raw_dataset.csv") else f"{args.out}_raw_dataset.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote raw dataset: {out_csv}  (rows={len(df)})", file=sys.stderr)


if __name__ == "__main__":
    main()
