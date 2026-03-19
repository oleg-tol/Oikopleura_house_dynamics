#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PCA on locomotion window features.

Outputs saved to OUTPUT_DIR/pca/:
    pca_coordinates_all.csv      – all windows with PC1…PCn scores
    pca_summary_per_animal.csv   – per-animal mean ± SD of PC scores
    pca_loadings.csv             – loading matrix + variance explained
    coverage_report.csv          – per-feature window coverage
    excluded_features_pc_corr.csv – Spearman r of low-coverage features vs PCs
"""
import os, re, warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


LOAD_FROM_CSV = False

_ROOT      = "YOUR FOLDER"
OUTPUT_DIR = os.path.join(_ROOT, "Data_hmm")
CSV_DIR    = os.path.join(OUTPUT_DIR, "wide_tables")
MTX_DIR    = os.path.join(OUTPUT_DIR, "matrices")
PCA_DIR    = os.path.join(OUTPUT_DIR, "pca")
os.makedirs(PCA_DIR, exist_ok=True)


KCOLS          = ["Experiment_ID", "Condition", "Individual"]
REFERENCE_COND = globals().get("REFERENCE_COND", "control")
PALETTE: Dict[str, str] = globals().get("PALETTE", {
    "control": "#AED6F1",
    "6-OHDA":  "#F1948A",
})

# =============================================================================
# COMPARISON GROUPS
# =============================================================================
COMPARISONS: Dict[str, Optional[List[str]]] = {
    "all_conditions": None,
}

# =============================================================================
# FEATURE CONFIG
# =============================================================================
FEATURE_LABELS = {
    "vel_p95":          "Velocity p95 (µm/s)",
    "vel_max":          "Velocity max (µm/s)",
    "vel_frac_active":  "Frac Active",
    "acc_p95":          "Accel p95 (µm/s²)",
    "tortuosity":       "Tortuosity",
    "path_complexity":  "Path Complexity (corr-H)",
    "msd_slope":        "MSD slope (px²/s²)",
    "svd_complexity":   "SVD Complexity (H)",
    "omega_mean":       "Omega mean (deg/s)",
    "omega_p95":        "Omega p95 (deg/s)",
    "tbf_mean":         "Tail-beat freq (Hz)",
    "tbf_amp_p95":      "Tail-beat amp p95 (px)",
    "curv_mean":        "Curvature mean",
    "quirkiness_mean":  "Quirkiness mean",
}

LOG_FEATS = {
    "vel_p95", "vel_max", "acc_p95", "omega_mean", "omega_p95",
    "tbf_amp_p95", "msd_slope", "curv_mean", "quirkiness_mean",
}

COVERAGE_MIN = 0.60   # minimum fraction of finite windows for a feature to enter PCA
ELLIPSE_CI   = 0.95   # confidence ellipse level (kept for downstream use)
CLIP_Q       = (0.005, 0.995)

# =============================================================================
# HELPERS
# =============================================================================
def _fl(c: str) -> str:
    return FEATURE_LABELS.get(c, c)

def _normalise_condition_case(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    """Lower-case any condition values that match ref case-insensitively."""
    if "Condition" not in df.columns:
        return df
    mapping = {c: c.lower() if c.lower() == ref.lower() and c != ref else c
               for c in df["Condition"].dropna().unique()}
    changed = {k: v for k, v in mapping.items() if k != v}
    if changed:
        print(f"  [case fix] Condition remapping: {changed}")
        df = df.copy()
        df["Condition"] = df["Condition"].map(mapping).fillna(df["Condition"])
    return df


# =============================================================================
# 1. Data loading
# =============================================================================
def load_win_df() -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the window feature table and recommended feature list.
    Uses in-memory globals when LOAD_FROM_CSV = False (normal pipeline use).
    Falls back to the normalised matrix CSV when running standalone.
    """
    if LOAD_FROM_CSV:
        p   = os.path.join(MTX_DIR, "matrix_all_conditions_all_windows_norm.csv")
        df  = pd.read_csv(p)
        fcs = [c for c in df.columns if c not in KCOLS + ["WindowStart"]]
        print(f"[load] {p}  →  {df.shape}")
        return df, fcs

    win_df_      = globals().get("win_df")
    recommended_ = globals().get("recommended", [])
    if win_df_ is None or (hasattr(win_df_, "empty") and win_df_.empty):
        raise RuntimeError(
            "win_df not in memory. "
            "Run the feature-selection script first, or set LOAD_FROM_CSV=True.")
    return win_df_, recommended_


# =============================================================================
# 2. Feature coverage check
# =============================================================================
def coverage_report(win_df: pd.DataFrame,
                    feat_cols: List[str]) -> Tuple[List[str], pd.DataFrame]:
    """
    Per-feature coverage = fraction of finite rows.
    Features below COVERAGE_MIN are excluded from PCA but correlated with PCs
    afterwards so their information is not lost.
    Saves coverage_report.csv.
    """
    n    = len(win_df)
    rows = []
    for c in feat_cols:
        if c not in win_df.columns:
            rows.append({"Feature": c, "Label": _fl(c), "Coverage": 0.0,
                         "N_finite": 0, "N_total": n,
                         "Included": False, "Note": "missing"})
            continue
        n_fin = int(win_df[c].notna().sum())
        cov   = n_fin / n if n > 0 else 0.0
        rows.append({"Feature": c, "Label": _fl(c), "Coverage": round(cov, 4),
                     "N_finite": n_fin, "N_total": n,
                     "Included": cov >= COVERAGE_MIN, "Note": ""})

    df       = pd.DataFrame(rows).sort_values("Coverage", ascending=False)
    included = df.loc[df["Included"],  "Feature"].tolist()
    excluded = df.loc[~df["Included"], "Feature"].tolist()

    path = os.path.join(PCA_DIR, "coverage_report.csv")
    df.to_csv(path, index=False)
    print(path)

    if excluded:
        print(f"\n[coverage] Excluding from PCA (coverage < {COVERAGE_MIN:.0%}): {excluded}")
        print(f"  These will be correlated with PCs in a supplementary CSV.")
    print(f"[coverage] PCA features ({len(included)}): {included}")

    return included, df


# =============================================================================
# 3. Preprocessing
# =============================================================================
def preprocess(win_df: pd.DataFrame,
               feat_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Pipeline:
      1. Robust clip  (0.5 % – 99.5 % across all windows)
      2. log1p        (right-skewed locomotion features)
      3. Per-individual robust z-score  (median / IQR)
         → removes between-animal baseline; PCA captures relative variation
      4. Drop rows still NaN in any included feature
      5. Global StandardScaler — FIT ON CONTROL ONLY, apply to all
         → equalises feature scales so no feature dominates PC1
    """
    df = win_df.copy()
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce")

    # 1. Clip
    for c in feat_cols:
        lo, hi = df[c].quantile(list(CLIP_Q))
        df[c]  = df[c].clip(lo, hi)

    # 2. log1p
    log_applied = [c for c in feat_cols if c in LOG_FEATS]
    for c in log_applied:
        df[c] = np.log1p(df[c].clip(lower=0))
    if log_applied:
        print(f"  log1p: {log_applied}")

    # 3. Per-individual robust z-score
    def _rz(s: pd.Series) -> pd.Series:
        med = s.median()
        iqr = s.quantile(0.75) - s.quantile(0.25)
        return (s - med) / iqr if (iqr > 0 and np.isfinite(iqr)) \
               else pd.Series(0.0, index=s.index)

    if "Individual" in df.columns:
        df[feat_cols] = (df.groupby("Individual", group_keys=False)[feat_cols]
                           .transform(_rz))
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    else:
        for c in feat_cols:
            mu, sd = df[c].mean(), df[c].std(ddof=0)
            df[c]  = (df[c] - mu) / (sd if sd > 0 else 1.0)

    # 4. Drop NaN rows
    n_before = len(df)
    df       = df.dropna(subset=feat_cols).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} NaN rows → {len(df)} remain")

    # 5. GlobalStandardScaler — fit on CONTROL animals only
    ref_mask = df["Condition"].str.lower() == REFERENCE_COND.lower()
    if ref_mask.sum() == 0:
        print(f"  [warn] No '{REFERENCE_COND}' windows found for scaler fit. "
              f"Using all windows.")
        ref_mask = pd.Series(True, index=df.index)

    scaler = StandardScaler()
    scaler.fit(df.loc[ref_mask, feat_cols].to_numpy(float))
    df[feat_cols] = scaler.transform(df[feat_cols].to_numpy(float))

    stds = df.loc[ref_mask, feat_cols].std().round(3).to_dict()
    print(f"  StandardScaler fit on {ref_mask.sum()} '{REFERENCE_COND}' windows")
    print(f"  Feature std in reference (should be ≈1): {stds}")

    return df, feat_cols


# =============================================================================
# 4. PCA — fit on control, project all
# =============================================================================
def fit_project_pca(df: pd.DataFrame,
                    feat_cols: List[str],
                    n_components: int = 10
                    ) -> Tuple[PCA, pd.DataFrame]:
    """
    Fit PCA on REFERENCE_COND (control) windows only.
    Project all windows onto that behavioural space.
    Adds PC1…PCn columns to df.
    """
    ref_mask = df["Condition"].str.lower() == REFERENCE_COND.lower()
    X_fit    = df.loc[ref_mask, feat_cols].to_numpy(float)
    X_all    = df[feat_cols].to_numpy(float)

    n_comp = min(n_components, X_fit.shape[1], X_fit.shape[0])
    pca    = PCA(n_components=n_comp, random_state=42)
    pca.fit(X_fit)

    scores = pca.transform(X_all)
    for i in range(n_comp):
        df[f"PC{i+1}"] = scores[:, i]

    evr = pca.explained_variance_ratio_
    print(f"\n[PCA] Fit on {ref_mask.sum()} '{REFERENCE_COND}' windows, "
          f"projected {len(df)} total windows")
    print("  Variance explained: " +
          "  ".join(f"PC{i+1}: {v:.1%}" for i, v in enumerate(evr)))
    return pca, df


# =============================================================================
# 5. Correlation of excluded (low-coverage) features with PCs
# =============================================================================
def compute_excluded_feature_pc_correlation(win_df_raw: pd.DataFrame,
                                             df_scored: pd.DataFrame,
                                             excluded_feats: List[str],
                                             pca: PCA,
                                             n_pcs: int = 4) -> pd.DataFrame:
    """
    Spearman r between each excluded (low-coverage) feature and each PC.
    Saves excluded_features_pc_corr.csv.
    Returns the correlation DataFrame.
    """
    if not excluded_feats:
        return pd.DataFrame()

    print(f"\n[excluded features] Correlating with PCs…")
    merge_keys = [c for c in KCOLS + ["WindowStart"]
                  if c in win_df_raw.columns and c in df_scored.columns]
    pc_cols = [f"PC{i+1}" for i in range(min(n_pcs, pca.n_components_))]
    merged  = pd.merge(
        df_scored[merge_keys + pc_cols],
        win_df_raw[merge_keys + [f for f in excluded_feats if f in win_df_raw.columns]],
        on=merge_keys, how="inner")

    rows = []
    for feat in excluded_feats:
        if feat not in merged.columns:
            continue
        for pc in pc_cols:
            sub = merged[[pc, feat]].dropna()
            if len(sub) < 10:
                rows.append({"Feature": feat, "Label": _fl(feat),
                             "PC": pc, "Spearman_r": np.nan, "N": len(sub)})
                continue
            r, p = spearmanr(sub[feat], sub[pc])
            rows.append({"Feature": feat, "Label": _fl(feat),
                         "PC": pc, "Spearman_r": round(float(r), 4),
                         "p": round(float(p), 4), "N": len(sub)})

    if not rows:
        return pd.DataFrame()

    rho_df = pd.DataFrame(rows)
    path   = os.path.join(PCA_DIR, "excluded_features_pc_corr.csv")
    rho_df.to_csv(path, index=False)
    print(path)
    return rho_df


# =============================================================================
# 6. Save outputs
# =============================================================================
def save_pca_outputs(df_scored: pd.DataFrame,
                     pca: PCA,
                     feat_cols: List[str]):
    pc_cols   = [c for c in df_scored.columns if re.match(r"^PC\d+$", c)]
    meta_cols = [c for c in KCOLS + ["WindowStart"] if c in df_scored.columns]

    # Full window-level PC coordinates
    out = df_scored[meta_cols + pc_cols].copy()
    path = os.path.join(PCA_DIR, "pca_coordinates_all.csv")
    out.to_csv(path, index=False, float_format="%.4f")
    print(path)

    # Per-animal centroid (mean ± SD of PC scores across windows)
    k        = [c for c in KCOLS if c in df_scored.columns]
    centroid = (df_scored.groupby(k, dropna=False)[pc_cols]
                          .agg(["mean", "std", "count"])
                          .reset_index())
    centroid.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in centroid.columns]
    path = os.path.join(PCA_DIR, "pca_summary_per_animal.csv")
    centroid.to_csv(path, index=False, float_format="%.4f")
    print(path)

    # Loadings table
    load_df = pd.DataFrame(
        pca.components_.T,
        index=feat_cols,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)])
    load_df.index.name = "Feature"
    load_df.insert(0, "Label", [_fl(c) for c in feat_cols])
    evr_row = pd.DataFrame(
        [["variance_explained", ""] + list(pca.explained_variance_ratio_)],
        columns=["Feature", "Label"] +
                [f"PC{i+1}" for i in range(pca.n_components_)])
    path = os.path.join(PCA_DIR, "pca_loadings.csv")
    pd.concat([load_df.reset_index(), evr_row]).to_csv(path, index=False,
                                                        float_format="%.4f")
    print(path)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # ── 1. Load ───────────────────────────────────────────────────────────
    win_df_raw, recommended_ = load_win_df()
    win_df_raw = _normalise_condition_case(win_df_raw, REFERENCE_COND)
    if not recommended_:
        recommended_ = [c for c in win_df_raw.columns
                        if c not in KCOLS + ["WindowStart"]]
    print(f"Loaded: {win_df_raw.shape}  |  features: {recommended_}")

    # ── 2. Coverage ───────────────────────────────────────────────────────
    pca_feats, cov_df = coverage_report(win_df_raw, recommended_)
    excluded_feats    = cov_df.loc[~cov_df["Included"], "Feature"].tolist()

    # ── 3. Preprocess ─────────────────────────────────────────────────────
    print("\n[preprocess]")
    df_clean, pca_feats = preprocess(win_df_raw.copy(), pca_feats)
    print(f"  Clean: {df_clean.shape[0]} windows × {len(pca_feats)} features")
    print(f"  Windows by condition: "
          f"{df_clean.groupby('Condition', dropna=False).size().to_dict()}")

    # ── 4. PCA ────────────────────────────────────────────────────────────
    pca_model, df_scored = fit_project_pca(df_clean, pca_feats, n_components=10)

    # ── 5. Excluded-feature correlation ───────────────────────────────────
    excl_corr_df = compute_excluded_feature_pc_correlation(
        win_df_raw, df_scored, excluded_feats, pca_model, n_pcs=4)

    # ── 6. Save ───────────────────────────────────────────────────────────
    print("\n[save]")
    save_pca_outputs(df_scored, pca_model, pca_feats)

    # ── 7. Summary ────────────────────────────────────────────────────────
    evr    = pca_model.explained_variance_ratio_
    cumevr = np.cumsum(evr)
    n80    = int(np.searchsorted(cumevr, 0.80)) + 1
    n90    = int(np.searchsorted(cumevr, 0.90)) + 1

    print(f"""
{'='*65}
PCA complete
  Reference for PCA fit  : '{REFERENCE_COND}'
  Features in PCA        : {len(pca_feats)}  →  {pca_feats}
  Excluded (low coverage): {excluded_feats}
  Windows analysed       : {len(df_scored)}
  Windows by condition   : {df_scored.groupby("Condition", dropna=False).size().to_dict()}
  Animals by condition   : {df_scored.groupby("Condition", dropna=False)["Individual"].nunique().to_dict()}

  Variance explained:
    {'  '.join(f"PC{i+1}: {v:.1%}" for i, v in enumerate(evr[:6]))}
    PCs for 80 %: {n80}   |   PCs for 90 %: {n90}

Outputs → {PCA_DIR}/
  coverage_report.csv
  pca_coordinates_all.csv
  pca_summary_per_animal.csv
  pca_loadings.csv
  excluded_features_pc_corr.csv   (if any excluded)

Next step — Gaussian HMM:
  Input  : pca_coordinates_all.csv  (columns PC1…PC{n80} cover 80 % variance)
  Group by Individual, sort by WindowStart → ordered time-series per animal
{'='*65}
""")

    # ── Expose to session for HMM script ──────────────────────────────────
    pca_features  = pca_feats
    pca_object    = pca_model
    df_pca_scored = df_scored