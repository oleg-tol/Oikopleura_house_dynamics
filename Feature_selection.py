#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection and normalisation for Gaussian HMM / PCA input.

Outputs saved to OUTPUT_DIR/feature_selection/ and OUTPUT_DIR/matrices/:
    feature_selection_report.csv   – CV, correlation, effect size per feature
    recommended_features.txt       – features surviving CV + corr filters
    ctrl_vs_treatment_stats.csv    – MWU effect sizes, all features × conditions
    feature_stats_summary.csv      – MWU stats for recommended features
    matrix_*_norm.csv              – normalised feature matrices (ref / all,
                                     all-windows / active-only)
    matrix_*_raw.csv               – un-normalised reference condition matrix
"""
import os, re, warnings
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# I/O CONFIG
# =============================================================================
# Set LOAD_FROM_CSV = True to run standalone from saved CSVs.
# Set to False when appended to / run after locomotion_metrics.py.
LOAD_FROM_CSV = False

_ROOT      = "YOUR FOLDER"
OUTPUT_DIR = os.path.join(_ROOT, "Data_hmm")

CSV_DIR = os.path.join(OUTPUT_DIR, "wide_tables")
FS_DIR  = os.path.join(OUTPUT_DIR, "feature_selection")
MTX_DIR = os.path.join(OUTPUT_DIR, "matrices")
for _d in (FS_DIR, MTX_DIR, CSV_DIR):
    os.makedirs(_d, exist_ok=True)

# =============================================================================
# TIMING / WINDOWING  (inherit from main script globals, or set defaults here)
# =============================================================================
FS_RS         = globals().get("FS_RS",         6.0)
WIN_RS_FRAMES = globals().get("WIN_RS_FRAMES", int(round(3.0 * FS_RS)))
STRIDE_RS_FR  = globals().get("STRIDE_RS_FR",  int(round(1.0 * FS_RS)))
DT_RS         = globals().get("DT_RS",         1.0 / FS_RS)

# =============================================================================
# IDENTIFIER COLUMNS
# =============================================================================
KCOLS = ["Experiment_ID", "Condition", "Individual"]

# =============================================================================
# CONDITIONS
# =============================================================================
CONTROL_NAME  = globals().get("CONTROL_NAME",  "control")
CONTROL_COLOR = globals().get("CONTROL_COLOR", "#AED6F1")
OTHER_COLORS  = globals().get("OTHER_COLORS",  ["#F1948A", "#82E0AA"])

REFERENCE_COND = CONTROL_NAME

PALETTE: Dict[str, str] = globals().get("PALETTE", {
    CONTROL_NAME: CONTROL_COLOR,
    "6-OHDA":     "#F1948A",
})

# =============================================================================
# FEATURE SELECTION THRESHOLDS
# =============================================================================
CORR_THRESHOLD = 0.70    # Spearman |r| above this → one of the pair is redundant
CV_MIN         = 0.05    # coefficient of variation below this → near-zero variance, drop
CLIP_Q         = (0.005, 0.995)   # robust clip quantiles before normalisation


# =============================================================================
FEATURE_DEFS = [
    # ── Already windowed (3 s / 1 s stride) ─────────────────────────────────
    ("vel_p95",          "window_wide", "vel_p95",           None,   None, "Velocity p95 (µm/s)"),
    ("vel_max",          "window_wide", "vel_max",           None,   None, "Velocity max (µm/s)"),
    ("vel_frac_active",  "window_wide", "vel_frac_active",   None,   None, "Frac Active"),
    ("acc_p95",          "window_wide", "acc_p95",           None,   None, "Acceleration p95 (µm/s²)"),
    ("tortuosity",       "window_wide", "tortuosity",        None,   None, "Tortuosity"),
    ("path_complexity",  "window_wide", "path_complexity",   None,   None, "Path Complexity (corr-H)"),
    ("msd_slope",        "window_wide", "msd_slope",         None,   None, "MSD slope (px²/s²)"),
    # ── SVD Takens complexity (windowed, same stride) ─────────────────────
    ("svd_complexity",   "svd",         None,                None,   None, "SVD Complexity (H)"),
    # ── Native-frame wides → windowed on-the-fly ─────────────────────────
    ("omega_mean",       "native_wide", "omega_abs_wide",      "mean", None, "Omega mean (deg/s)"),
    ("omega_p95",        "native_wide", "omega_abs_wide",      "p",    95.0, "Omega p95 (deg/s)"),
    ("tbf_mean",         "native_wide", "tailbeat_freq_wide",  "mean", None, "Tail-beat freq (Hz)"),
    ("tbf_amp_p95",      "native_wide", "tailbeat_amp_wide",   "p",    95.0, "Tail-beat amp p95 (px)"),
    ("curv_mean",        "native_wide", "avg_curvature_wide",  "mean", None, "Curvature mean"),
    ("quirkiness_mean",  "native_wide", "quirkiness_wide",     "mean", None, "Quirkiness mean"),
]

FEATURE_LABELS = {fd[0]: fd[5] for fd in FEATURE_DEFS}

# Which features to log-transform before normalisation (positive-skewed)
LOG_FEATS = {
    "vel_p95", "vel_max", "acc_p95", "omega_mean", "omega_p95",
    "tbf_amp_p95", "msd_slope", "curv_mean", "quirkiness_mean",
}


# =============================================================================
# HELPERS
# =============================================================================
def _tcols(df: pd.DataFrame) -> List[str]:
    return sorted(
        [c for c in df.columns if isinstance(c, str) and c.startswith("Time_")],
        key=lambda c: int(re.search(r"\d+$", c).group()) if re.search(r"\d+$", c) else 0)


def _rank_biserial_r(x: np.ndarray, y: np.ndarray) -> float:
    """Effect size for Mann-Whitney U (ranges −1 to +1)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return np.nan
    U, _ = mannwhitneyu(x, y, alternative="two-sided")
    return float(1.0 - 2.0 * U / (x.size * y.size))


def _holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj   = np.empty(m, float)
    prev  = 0.0
    for rank, i in enumerate(order):
        a      = min(1.0, pvals[i] * (m - rank))
        adj[i] = max(prev, a)
        prev   = adj[i]
    return adj.tolist()


# =============================================================================
# STEP 0: Wide-table helpers
# =============================================================================
def _wide_table_to_long(wdf: pd.DataFrame, feat_name: str) -> pd.DataFrame:
    """
    Convert a windowed-wide table (Time_XXX cols keyed by window-start frame)
    into a long DataFrame: one row per (animal × window).
    """
    if wdf is None or wdf.empty:
        return pd.DataFrame()
    k  = [c for c in KCOLS if c in wdf.columns]
    tc = _tcols(wdf)
    if not tc:
        return pd.DataFrame()
    rows = []
    for _, row in wdf.iterrows():
        for col in tc:
            v = float(row[col])
            if not np.isfinite(v):
                continue
            start = int(re.search(r"\d+$", col).group())
            meta  = {c: row[c] for c in k}
            meta["WindowStart"] = start
            meta[feat_name]     = v
            rows.append(meta)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def wide_to_window_summary(df: pd.DataFrame,
                            feat_name: str,
                            stat: str = "mean",
                            q: float = 95.0,
                            win_frames: int = WIN_RS_FRAMES,
                            stride_frames: int = STRIDE_RS_FR) -> pd.DataFrame:
    """
    Slide a window over a native-frame wide table and compute a per-window scalar.
    Returns long DataFrame: one row per valid (animal × window).
    """
    if df is None or df.empty:
        return pd.DataFrame()
    k      = [c for c in KCOLS if c in df.columns]
    tc     = _tcols(df)
    if not tc:
        return pd.DataFrame()
    T      = len(tc)
    starts = np.arange(0, T - win_frames + 1, stride_frames)
    if starts.size == 0:
        return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        vals = row[tc].to_numpy(float)
        for s in starts:
            w   = vals[s: s + win_frames]
            fin = w[np.isfinite(w)]
            if fin.size == 0:
                continue
            v = float(np.nanmean(fin)) if stat == "mean" else float(np.nanpercentile(fin, q))
            if not np.isfinite(v):
                continue
            meta = {c: row[c] for c in k}
            meta["WindowStart"] = int(s)
            meta[feat_name]     = v
            rows.append(meta)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# STEP 1: Assemble the master window feature table
# =============================================================================
def _load(name: str) -> Optional[pd.DataFrame]:
    """
    Retrieve a DataFrame: from globals() if LOAD_FROM_CSV=False, else from CSV_DIR.
    """
    if not LOAD_FROM_CSV:
        obj = globals().get(name)
        if obj is not None:
            return obj if isinstance(obj, pd.DataFrame) else None
        wwt = globals().get("window_wide_tables")
        if isinstance(wwt, dict) and name in wwt:
            return wwt[name]
        return None
    else:
        candidates = [
            os.path.join(CSV_DIR, f"{name}.csv"),
            os.path.join(CSV_DIR, f"window_{name}_wide.csv"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return pd.read_csv(p)
        return None


def assemble_window_feature_table() -> pd.DataFrame:
    """
    Build the master long-format feature table:
    one row per (animal × 3 s window), columns = KCOLS + WindowStart + features.
    """
    print("\n[assemble] Building window feature table…")
    parts: Dict[str, pd.DataFrame] = {}

    wwt = (globals().get("window_wide_tables") if not LOAD_FROM_CSV else None)

    for feat, src_type, src_name, stat, q, _ in FEATURE_DEFS:

        # ── A. Already-windowed metrics ──────────────────────────────────
        if src_type == "window_wide":
            wdf = None
            if isinstance(wwt, dict):
                wdf = wwt.get(src_name)
            if wdf is None:
                wdf = _load(src_name)
            if wdf is None:
                csv_path = os.path.join(CSV_DIR, f"window_{src_name}_wide.csv")
                if os.path.isfile(csv_path):
                    wdf = pd.read_csv(csv_path)
            if wdf is None or (hasattr(wdf, "empty") and wdf.empty):
                print(f"  [skip] {feat}: window_wide '{src_name}' not found")
                continue
            long = _wide_table_to_long(wdf, feat)

        # ── B. SVD complexity ─────────────────────────────────────────────
        elif src_type == "svd":
            cdf = (globals().get("complexity_win_wide") if not LOAD_FROM_CSV else None)
            if cdf is None or (hasattr(cdf, "empty") and cdf.empty):
                csv_path = os.path.join(CSV_DIR, "svd_complexity_wide.csv")
                if os.path.isfile(csv_path):
                    cdf = pd.read_csv(csv_path)
            if cdf is None or (hasattr(cdf, "empty") and cdf.empty):
                print(f"  [skip] {feat}: svd_complexity_wide not found")
                continue
            long = _wide_table_to_long(cdf, feat)

        # ── C. Native-frame wides → windowed on-the-fly ──────────────────
        elif src_type == "native_wide":
            ndf = _load(src_name)
            if ndf is None or (hasattr(ndf, "empty") and ndf.empty):
                csv_path = os.path.join(CSV_DIR, f"{src_name}.csv")
                if os.path.isfile(csv_path):
                    ndf = pd.read_csv(csv_path)
            if ndf is None or (hasattr(ndf, "empty") and ndf.empty):
                print(f"  [skip] {feat}: {src_name} not found")
                continue
            long = wide_to_window_summary(ndf, feat,
                                          stat=stat or "mean",
                                          q=q or 95.0)
        else:
            continue

        if long is None or (hasattr(long, "empty") and long.empty):
            print(f"  [skip] {feat}: produced empty table")
            continue

        print(f"  {feat}: {len(long)} window rows")
        parts[feat] = long

    if not parts:
        raise RuntimeError(
            "No features assembled. Ensure locomotion_metrics.py ran first, "
            "or set LOAD_FROM_CSV=True and point CSV_DIR at Data_hmm/wide_tables.")

    key_cols = [c for c in KCOLS + ["WindowStart"]
                if all(c in df.columns for df in parts.values())]
    out = list(parts.values())[0]
    for feat, df in list(parts.items())[1:]:
        merge_on = [c for c in key_cols if c in out.columns and c in df.columns]
        out = pd.merge(out, df[merge_on + [feat]], on=merge_on, how="outer")

    feat_cols = [fd[0] for fd in FEATURE_DEFS if fd[0] in out.columns]
    n_complete = out[feat_cols].notna().all(axis=1).sum()
    print(f"\n[assemble] Done: {len(out)} windows × {len(feat_cols)} features")
    print(f"  Features present  : {feat_cols}")
    print(f"  Windows all-finite: {n_complete} / {len(out)}")
    print(f"  Windows by cond   : {out.groupby('Condition', dropna=False).size().to_dict()}")
    return out


# =============================================================================
# STEP 2: Feature selection
# =============================================================================
def run_feature_selection(win_df: pd.DataFrame,
                           feat_cols: List[str]
                           ) -> Tuple[List[str], pd.DataFrame]:
    """
    Three-step feature selection computed on REFERENCE_COND (control) animals:
      A. CV filter   — near-zero variance features
      B. Spearman correlation filter — remove redundant features
         Tie-breaking: keep the feature with higher discriminability
         (|rank-biserial r|, control vs treatment).
      C. Effect-size diagnostic — saved to CSV but NOT used for hard filtering

    Returns
    -------
    recommended : List[str]    features surviving A + B
    report_df   : pd.DataFrame one row per feature with all diagnostics
    """
    print(f"\n[feature_selection] Computing on '{REFERENCE_COND}' animals…")

    ref = win_df[win_df["Condition"].str.lower() == REFERENCE_COND.lower()].copy()
    if ref.empty:
        print(f"  [warn] '{REFERENCE_COND}' not found in data. Using all conditions.")
        ref = win_df.copy()

    feat_cols = [c for c in feat_cols if c in ref.columns]
    ref[feat_cols] = ref[feat_cols].apply(pd.to_numeric, errors="coerce")

    # ── A. Coefficient of Variation ──────────────────────────────────────
    cv: Dict[str, float] = {}
    for c in feat_cols:
        v  = ref[c].dropna().to_numpy(float)
        v  = v[np.isfinite(v)]
        mu = float(np.nanmean(v))
        sd = float(np.nanstd(v, ddof=1)) if v.size > 1 else 0.0
        cv[c] = (sd / abs(mu)) if abs(mu) > 1e-12 else 0.0

    low_var = [c for c, v in cv.items() if v < CV_MIN]
    print(f"  CV filter drops: {low_var}")

    # ── B. Spearman correlation ───────────────────────────────────────────
    valid    = ref[feat_cols].apply(pd.to_numeric, errors="coerce").dropna()
    corr_mat = valid.corr(method="spearman")

    treatments = [c for c in win_df["Condition"].dropna().unique()
                  if c.lower() != REFERENCE_COND.lower()]
    ctrl_data  = win_df[win_df["Condition"].str.lower() == REFERENCE_COND.lower()]

    eff: Dict[str, float] = {}
    for c in feat_cols:
        xa = ctrl_data[c].dropna().to_numpy(float)
        rs = []
        for trt in treatments:
            xb = win_df[win_df["Condition"] == trt][c].dropna().to_numpy(float)
            rs.append(abs(_rank_biserial_r(xa, xb)))
        eff[c] = float(np.nanmean(rs)) if rs else 0.0

    corr_pairs = [(a, b, corr_mat.loc[a, b])
                  for a, b in combinations(feat_cols, 2)
                  if abs(corr_mat.loc[a, b]) >= CORR_THRESHOLD]

    to_drop_corr: set = set()
    for a, b, r in sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        if a in to_drop_corr or b in to_drop_corr:
            continue
        drop = b if eff.get(a, 0.0) >= eff.get(b, 0.0) else a
        keep = a if drop == b else b
        to_drop_corr.add(drop)
        print(f"  Corr drop: {drop:20s}  (|r|={abs(r):.2f} with {keep};  "
              f"eff_{keep}={eff.get(keep,0):.2f} > eff_{drop}={eff.get(drop,0):.2f})")

    to_drop     = set(low_var) | to_drop_corr
    recommended = [c for c in feat_cols if c not in to_drop]
    print(f"\n  Recommended ({len(recommended)}): {recommended}")

    # ── Build report ─────────────────────────────────────────────────────
    rows = []
    for c in feat_cols:
        v = ref[c].dropna().to_numpy(float)
        v = v[np.isfinite(v)]
        rows.append({
            "Feature":       c,
            "Label":         FEATURE_LABELS.get(c, c),
            "CV":            cv.get(c, np.nan),
            "N_windows":     len(v),
            "Median":        float(np.nanmedian(v))       if v.size else np.nan,
            "IQR":           float(np.nanpercentile(v, 75) -
                                   np.nanpercentile(v, 25)) if v.size else np.nan,
            "Effect_size":   eff.get(c, np.nan),
            "Dropped_CV":    c in set(low_var),
            "Dropped_Corr":  c in to_drop_corr,
            "Kept":          c in recommended,
        })
    report_df = pd.DataFrame(rows)
    report_df.to_csv(os.path.join(FS_DIR, "feature_selection_report.csv"), index=False)
    with open(os.path.join(FS_DIR, "recommended_features.txt"), "w") as fh:
        fh.write("\n".join(recommended) + "\n")
    print(f"  Report → {FS_DIR}/feature_selection_report.csv")

    return recommended, report_df


# =============================================================================
# STEP 3: Stats — each treatment vs control (Holm-corrected MWU)
# =============================================================================
def compute_feature_stats(win_df: pd.DataFrame,
                           feat_cols: List[str],
                           ref_cond: str = REFERENCE_COND) -> pd.DataFrame:
    """
    Per-animal mean first (avoids window pseudo-replication), then MWU.
    Returns one row per (feature × treatment) with:
      N_ctrl, N_trt, Mean/Median/Std for each,
      U, p_raw, p_holm, effect_size (|rank-biserial r|), sig
    """
    k   = [c for c in KCOLS if c in win_df.columns]
    agg = win_df.groupby(k, dropna=False)[feat_cols].mean().reset_index()

    ctrl_mask  = agg["Condition"].str.lower() == ref_cond.lower()
    ctrl_data  = agg[ctrl_mask]
    treatments = sorted([c for c in agg["Condition"].dropna().unique()
                         if c.lower() != ref_cond.lower()])

    rows = []
    for feat in feat_cols:
        xa     = ctrl_data[feat].dropna().to_numpy(float)
        xa     = xa[np.isfinite(xa)]
        raw_ps, infos = [], []
        for trt in treatments:
            xb = agg[agg["Condition"] == trt][feat].dropna().to_numpy(float)
            xb = xb[np.isfinite(xb)]

            def _desc(v):
                return (len(v),
                        float(np.mean(v))        if v.size else np.nan,
                        float(np.median(v))      if v.size else np.nan,
                        float(np.std(v, ddof=1)) if v.size > 1 else np.nan)

            n_c, m_c, med_c, sd_c = _desc(xa)
            n_t, m_t, med_t, sd_t = _desc(xb)

            if xa.size < 2 or xb.size < 2:
                raw_ps.append(1.0)
                infos.append(dict(
                    Feature=feat, Label=FEATURE_LABELS.get(feat, feat),
                    Control=ref_cond, Treatment=trt,
                    N_ctrl=n_c, Mean_ctrl=m_c, Median_ctrl=med_c, Std_ctrl=sd_c,
                    N_trt=n_t,  Mean_trt=m_t,  Median_trt=med_t,  Std_trt=sd_t,
                    U=np.nan, p_raw=1.0, effect_size=np.nan))
                continue

            U, p = mannwhitneyu(xa, xb, alternative="two-sided")
            r    = _rank_biserial_r(xa, xb)
            raw_ps.append(float(p))
            infos.append(dict(
                Feature=feat, Label=FEATURE_LABELS.get(feat, feat),
                Control=ref_cond, Treatment=trt,
                N_ctrl=n_c, Mean_ctrl=m_c, Median_ctrl=med_c, Std_ctrl=sd_c,
                N_trt=n_t,  Mean_trt=m_t,  Median_trt=med_t,  Std_trt=sd_t,
                U=float(U), p_raw=float(p), effect_size=float(r)))

        adj = _holm_adjust(raw_ps)
        for info, p_adj in zip(infos, adj):
            info["p_holm"] = p_adj
            info["sig"]    = p_adj < 0.05
            rows.append(info)

    df_out = pd.DataFrame(rows)
    for col in ["Mean_ctrl","Median_ctrl","Std_ctrl","Mean_trt","Median_trt","Std_trt",
                "U","p_raw","p_holm","effect_size"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round(4)
    return df_out


# =============================================================================
# STEP 4: Normalisation helpers
# =============================================================================
def _robust_clip(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    lo_q, hi_q = CLIP_Q
    for c in cols:
        if c not in df.columns: continue
        lo, hi = df[c].quantile([lo_q, hi_q])
        df[c]  = df[c].clip(lo, hi)
    return df


def _log_transform(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in LOG_FEATS and c in df.columns:
            df[c] = np.log1p(df[c].astype(float).clip(lower=0))
    return df


def _normalise_within_individual(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Per-individual robust z-score (median / IQR).
    Removes between-animal baseline differences so PCA captures *changes*
    in behaviour across windows.
    Animals with IQR == 0 for a feature get that feature set to 0.
    """
    df = df.copy()

    def _rz(series: pd.Series) -> pd.Series:
        med = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        if iqr == 0 or not np.isfinite(iqr):
            return pd.Series(0.0, index=series.index)
        return (series - med) / iqr

    if "Individual" not in df.columns:
        for c in cols:
            mu, sd = df[c].mean(), df[c].std(ddof=0)
            df[c]  = (df[c] - mu) / (sd if sd > 0 else 1.0)
        return df

    df[cols] = (df.groupby("Individual", group_keys=False)[cols]
                  .transform(_rz))
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# =============================================================================
# STEP 5: Case-insensitive condition fix
# =============================================================================
def _normalise_condition_case(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    """
    If the capitalisation of the reference condition in the data doesn't match
    REFERENCE_COND, remap to lowercase so all comparisons work.
    """
    if "Condition" not in df.columns:
        return df
    unique     = df["Condition"].dropna().unique()
    needs_fix  = any(c.lower() == ref.lower() and c != ref for c in unique)
    if not needs_fix:
        return df
    mapping = {c: c.lower() if c.lower() == ref.lower() else c for c in unique}
    print(f"  [case fix] Condition remapping: { {k: v for k, v in mapping.items() if k != v} }")
    out = df.copy()
    out["Condition"] = out["Condition"].map(mapping).fillna(out["Condition"])
    return out


# =============================================================================
# STEP 6: Build analysis matrices
# =============================================================================
def build_analysis_matrix(
    win_df: pd.DataFrame,
    recommended: List[str],
    active_only: bool = False,
    pca_ref_conds: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Build normalised feature matrices ready for PCA → Gaussian HMM.

    Fit normalisation on reference (control) animals;
    apply the SAME parameters to all conditions.
    Here, per-individual robust z-scores are used so all conditions
    are processed identically.

    Parameters
    ----------
    win_df          : output of assemble_window_feature_table()
    recommended     : feature columns to include
    active_only     : if True, drop windows where vel_frac_active == 0
    pca_ref_conds   : conditions that form the reference population
                      (default: [REFERENCE_COND])

    Returns
    -------
    meta_ref : metadata DataFrame for reference-condition rows
    X_ref    : normalised feature matrix for reference conditions
    meta_all : metadata DataFrame for all conditions
    X_all    : normalised feature matrix for all conditions
    """
    if pca_ref_conds is None:
        pca_ref_conds = [REFERENCE_COND]

    feat_cols = [c for c in recommended if c in win_df.columns]
    if not feat_cols:
        raise RuntimeError("No recommended features found in win_df.")

    df = win_df.copy()

    if active_only and "vel_frac_active" in df.columns:
        n_before = len(df)
        df = df[df["vel_frac_active"] > 0].copy()
        print(f"  [active_only] dropped {n_before - len(df)} inactive windows, "
              f"{len(df)} remain")

    df = df.dropna(subset=feat_cols).copy()
    print(f"  After NaN drop: {len(df)} windows")

    df = _robust_clip(df, feat_cols)
    df = _log_transform(df, feat_cols)
    df = _normalise_within_individual(df, feat_cols)

    k        = [c for c in KCOLS + ["WindowStart"] if c in df.columns]
    meta_all = df[k].copy().reset_index(drop=True)
    X_all    = df[feat_cols].to_numpy(float)

    ref_mask = df["Condition"].isin(pca_ref_conds)
    meta_ref = df.loc[ref_mask, k].copy().reset_index(drop=True)
    X_ref    = df.loc[ref_mask, feat_cols].to_numpy(float)

    print(f"  Reference matrix ({pca_ref_conds}): {X_ref.shape}")
    print(f"  Full matrix (all conditions):       {X_all.shape}")
    print(f"  Windows by condition: {df.groupby('Condition', dropna=False).size().to_dict()}")
    print(f"  Features: {feat_cols}")

    return meta_ref, X_ref, meta_all, X_all


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # ── 1. Assemble ────────────────────────────────────────────────────────
    win_df        = assemble_window_feature_table()
    win_df        = _normalise_condition_case(win_df, REFERENCE_COND)
    PALETTE       = {(k.lower() if k.lower() == REFERENCE_COND.lower() else k): v
                     for k, v in PALETTE.items()}
    all_feat_cols = [fd[0] for fd in FEATURE_DEFS if fd[0] in win_df.columns]
    print(f"\nAll features available: {all_feat_cols}")

    # ── 2. Feature selection ────────────────────────────────────────────────
    recommended, report_df = run_feature_selection(win_df, all_feat_cols)

    # ── 3. Stats: each recommended feature, ctrl vs treatment ───────────────
    stats_df = compute_feature_stats(win_df, recommended)
    stats_csv = os.path.join(FS_DIR, "feature_stats_summary.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"\n[stats] Saved → {stats_csv}")
    print(stats_df[["Feature","Treatment","N_ctrl","N_trt",
                    "Mean_ctrl","Mean_trt","effect_size","p_raw","p_holm","sig"]]
          .to_string(index=False))

    # Also save condition-level effect sizes for all features (not just recommended)
    all_stats_df = compute_feature_stats(win_df, all_feat_cols)
    all_stats_df.to_csv(os.path.join(FS_DIR, "ctrl_vs_treatment_stats.csv"), index=False)
    print(f"[stats] All-feature stats → {FS_DIR}/ctrl_vs_treatment_stats.csv")

    # ── 4a. Matrices — ALL windows ─────────────────────────────────────────
    print("\n[build_matrix] ALL windows…")
    meta_ref_all, X_ref_all, meta_all_all, X_all_all = build_analysis_matrix(
        win_df, recommended, active_only=False)

    # ── 4b. Matrices — ACTIVE windows only ─────────────────────────────────
    print("\n[build_matrix] ACTIVE windows only…")
    meta_ref_act, X_ref_act, meta_all_act, X_all_act = build_analysis_matrix(
        win_df, recommended, active_only=True)

    # ── 5. Save matrices ────────────────────────────────────────────────────
    feat_kept = [c for c in recommended if c in win_df.columns]

    def _save(meta: pd.DataFrame, X: np.ndarray, fname: str):
        out = meta.copy()
        for i, c in enumerate(feat_kept):
            out[c] = X[:, i]
        path = os.path.join(MTX_DIR, fname)
        out.to_csv(path, index=False, float_format="%.4f")
        print(path)

    print("\n[save] Writing matrices…")
    _save(meta_ref_all, X_ref_all, f"matrix_{REFERENCE_COND}_all_windows_norm.csv")
    _save(meta_all_all, X_all_all,  "matrix_all_conditions_all_windows_norm.csv")
    _save(meta_ref_act, X_ref_act, f"matrix_{REFERENCE_COND}_active_windows_norm.csv")
    _save(meta_all_act, X_all_act,  "matrix_all_conditions_active_windows_norm.csv")

    raw = win_df[win_df["Condition"].str.lower() == REFERENCE_COND.lower()].copy()
    raw[[c for c in KCOLS + ["WindowStart"] + feat_kept if c in raw.columns]].to_csv(
        os.path.join(MTX_DIR, f"matrix_{REFERENCE_COND}_raw.csv"), index=False)
    print(os.path.join(MTX_DIR, f"matrix_{REFERENCE_COND}_raw.csv"))

    # ── 6. Summary ──────────────────────────────────────────────────────────
    print(f"""
{'='*65}
Feature selection complete
  All features assessed : {len(all_feat_cols)}
  Recommended (kept)    : {len(recommended)}
    → {recommended}
  Dropped (CV)          : {report_df.loc[report_df.Dropped_CV,  'Feature'].tolist()}
  Dropped (corr≥{CORR_THRESHOLD})  : {report_df.loc[report_df.Dropped_Corr,'Feature'].tolist()}

Output directories
  Feature selection  →  {FS_DIR}
  Normalised matrices →  {MTX_DIR}

Ready for PCA → Gaussian HMM:
  Reference ({REFERENCE_COND}) — all windows : {X_ref_all.shape}
  Reference ({REFERENCE_COND}) — active only : {X_ref_act.shape}
  All conditions — all windows              : {X_all_all.shape}
  All conditions — active only              : {X_all_act.shape}
  Features in matrix: {feat_kept}
{'='*65}
""")