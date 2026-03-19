#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMM analysis pipeline for house-expansion dataset.

Sections:
  1. BIC scan   — fit on REFERENCE_COND only, scan K = MIN_STATES…MAX_STATES
  2. Final fit  — fit on ALL animals with K_FINAL, Viterbi decode
  3. Plots      — full ethograms, HE transient panels, chord diagrams

Outputs saved to OUTDIR (Data_hmm/hmm/):
    bic_scan_results.csv
    hmm_restart_diagnostics.csv
    hmm_decoded_all_windows.csv
    hmm_state_occupancy_per_animal.csv
    hmm_transition_rates_per_animal.csv
    hmm_state_means.csv
    hmm_transition_matrix.csv
    transition_matrix_events_<cond>.csv
    05_ethogram_<cond>.png/svg
    05b_he_transients_<cond>.png/svg
    chord_diagrams/chord_<cond>.png/svg
    chord_diagrams/chord_all_conditions.png/svg

Run after pca_analysis.py.
"""

import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from hmmlearn.hmm import GaussianHMM
from pycirclize import Circos

# =============================================================================
# PATHS
# =============================================================================
_ROOT   = "YOUR FOLDER"
PCA_CSV = os.path.join(_ROOT, "Data_hmm", "pca", "pca_coordinates_all.csv")
OUTDIR  = os.path.join(_ROOT, "Data_hmm", "hmm")
CHORD_DIR = os.path.join(OUTDIR, "chord_diagrams")
os.makedirs(OUTDIR,    exist_ok=True)
os.makedirs(CHORD_DIR, exist_ok=True)

# =============================================================================
# KNOBS — BIC SCAN
# =============================================================================
N_PCS          = 4       # PCs to use (check pca_loadings.csv for 80 % variance)
REFERENCE_COND = "control"
N_PER_ANIMAL   = None    # None → all windows; positive int to subsample
MIN_WIN        = 5       # minimum windows per animal to include a sequence

MIN_STATES     = 2
MAX_STATES     = 10
N_INIT_SCAN    = 5       # random restarts per K
N_ITER_SCAN    = 150     # EM iterations per restart

# =============================================================================
# KNOBS — FINAL FIT
# =============================================================================
K_FINAL       = 6        # ← set from bic_scan_results.csv elbow
N_INIT_FINAL  = 10
N_ITER_FINAL  = 300
COV_TYPE      = "full"   # "full" | "diag"
MIN_COVAR     = 0.01
RARE_FRAC     = 0.01     # states below this fraction of all windows are merged
MIN_DWELL     = 3        # median-filter dwell (windows ≈ seconds at 1-s stride)
RANDOM_STATE  = 42

# =============================================================================
# KNOBS — PLOTS
# =============================================================================
ETHOGRAM_N      = 20
PRE_SEC         = 60     # seconds of context before HE onset
POST_SEC        = 60     # seconds of context after HE offset
MIN_DWELL_SEC   = 10     # debounce for event-based transitions
CHORD_THRESHOLD = 0.10   # off-diagonal links below this are not drawn

# =============================================================================
# IDENTIFIERS + PALETTE
# =============================================================================
KCOLS   = ["Experiment_ID", "Condition", "Individual"]
PALETTE = {"control": "#AED6F1", "6-OHDA": "#F1948A"}

FPS_NATIVE   = 30.0
FS_RS        = 6.0    # resampled Hz (WindowStart is in resampled frames)
STRIDE_RS_FR = 6

PUB_STYLE = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi":      150,
    "savefig.dpi":     300,
}

# =============================================================================
# HOUSE EXPANSION FRAME RANGES  (native frames @ 30 fps)
# =============================================================================
HOUSE_EXPANSION_FRAMES = {
    # --- Control ---
    "20250527_100240_1_15m0s_None_None_None": [(1740, 2820)],
    "20250527_103744_1_15m0s_None_None_None": [(2550, 3150)],
    "20250527_111628_1_15m0s_None_None_None": [(1770, 2190)],
    "20250602_141106_1_15m0s_None_None_None": [(2070, 2880)],
    "20250623_102938_1_15m0s_None_None_None": [(270,   900)],
    "20250623_150448_1_15m0s_None_None_None": [(2880, 3810)],
    "20250623_152301_1_15m0s_None_None_None": [(240,   660)],
    "20250818_101132_1_15m0s_None_None_None": [(870,  1290)],
    "20250818_103120_1_15m0s_None_None_None": [(1860, 2400)],
    "20250818_103121_1_15m0s_None_None_None": [(240,   750)],
    "20250818_103123_1_15m0s_None_None_None": [(1200, 1680)],
    "20250818_152236_1_15m0s_None_None_None": [(17760, 18330)],
    "20250819_150135_1_15m0s_None_None_None": [(3300, 4110)],
    "20250902_095505_1_15m0s_None_None_None": [(1830, 2460)],
    "20250902_095507_1_15m0s_None_None_None": [(1860, 2520)],
    # --- 6-OHDA ---
    "20250602_153855_1_15m0s_None_None_None": [(1320, 1890)],
    "20250623_131035_1_15m0s_None_None_None": [(1050, 2010)],
    "20250623_131036_1_15m0s_None_None_None": [(7980, 8820)],
    "20250623_133008_1_15m0s_None_None_None": [(450,  1620)],
    "20250623_133010_1_15m0s_None_None_None": [(17940, 18510)],
    "20250623_134915_1_15m0s_None_None_None": [(16470, 17100)],
    "20250818_135423_1_15m0s_None_None_None": [(1080, 1770)],
}

# =============================================================================
# SHARED HELPERS
# =============================================================================
def _normalise_condition_case(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    mapping = {c: c.lower() if c.lower() == ref.lower() and c != ref else c
               for c in df["Condition"].dropna().unique()}
    changed = {k: v for k, v in mapping.items() if k != v}
    if changed:
        print(f"  [case fix] {changed}")
        df = df.copy()
        df["Condition"] = df["Condition"].map(mapping).fillna(df["Condition"])
    return df


def _state_palette(K: int) -> list:
    cmap = mpl.colormaps.get_cmap("tab10")
    return [mpl.colors.to_hex(cmap(i % 10)) for i in range(K)]


def _he_spans_sec(exp_id: str) -> list:
    """Exact match then prefix fallback (first 15 chars = timestamp)."""
    if exp_id in HOUSE_EXPANSION_FRAMES:
        raw = HOUSE_EXPANSION_FRAMES[exp_id]
    else:
        raw = next((v for k, v in HOUSE_EXPANSION_FRAMES.items()
                    if k[:15] == str(exp_id)[:15]), [])
    return [(s / FPS_NATIVE, e / FPS_NATIVE) for s, e in raw]


def _kmeans_init(X: np.ndarray, K: int, cov_type: str,
                 seed: int, eps: float = 1e-4):
    rng = np.random.RandomState(seed)
    try:
        km     = KMeans(n_clusters=K, random_state=seed, n_init=10).fit(X)
        labels = km.labels_
        means  = km.cluster_centers_
    except Exception:
        labels = rng.randint(0, K, X.shape[0])
        means  = np.array([X[labels == k].mean(0) if (labels == k).any()
                           else X[rng.choice(len(X))] for k in range(K)])
    D = X.shape[1]
    if cov_type == "full":
        covars = np.zeros((K, D, D))
        for k in range(K):
            pts       = X[labels == k]
            c         = np.cov(pts, rowvar=False) if len(pts) > 1 else np.eye(D)
            covars[k] = c + np.eye(D) * eps
    else:
        covars = np.zeros((K, D))
        for k in range(K):
            pts       = X[labels == k]
            covars[k] = np.clip(
                np.var(pts, 0) if len(pts) > 1 else np.ones(D), eps, None)
    counts = np.bincount(labels, minlength=K).astype(float)
    counts[counts == 0] = 1e-8
    sp  = counts / counts.sum()
    off = (1 - 0.8) / (K - 1) if K > 1 else 0.0
    T   = np.full((K, K), off)
    np.fill_diagonal(T, 0.8 if K > 1 else 1.0)
    T  /= T.sum(1, keepdims=True)
    return means, covars, sp, T


def _fit_one(X: np.ndarray, lengths: list, K: int,
             cov_type: str, n_iter: int, seed: int):
    means, covars, sp, T = _kmeans_init(X, K, cov_type, seed)
    m = GaussianHMM(n_components=K, covariance_type=cov_type,
                    n_iter=n_iter, tol=1e-4, random_state=seed,
                    verbose=False, init_params="", min_covar=MIN_COVAR)
    m.startprob_ = sp
    m.transmat_  = T
    m.means_     = means
    m.covars_    = covars
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, lengths)
    return m, m.score(X, lengths)


def _n_params(K: int, D: int, cov_type: str) -> int:
    cov_p = {"full": K * D * (D + 1) // 2, "diag": K * D}[cov_type]
    return (K - 1) + K * (K - 1) + K * D + cov_p


def smooth_states(states: np.ndarray, min_dwell: int) -> np.ndarray:
    if min_dwell <= 1:
        return states
    return median_filter(states.astype(int), size=min_dwell, mode="nearest").astype(int)


def _savefig(fig: plt.Figure, name: str, folder: str = None):
    d = folder or OUTDIR
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300 if ext == "png" else None)
    plt.close(fig)


def _draw_raster(ax: plt.Axes, seq: np.ndarray,
                 t_sec: np.ndarray, s_pal: list):
    """State sequence as Rectangle patches — vector, lines always render on top."""
    i = 0
    while i < len(seq):
        state   = seq[i]
        run_end = i + 1
        while run_end < len(seq) and seq[run_end] == state:
            run_end += 1
        ax.add_patch(mpatches.Rectangle(
            (float(t_sec[i]), 0),
            float(t_sec[run_end - 1]) + 1.0 - float(t_sec[i]), 1.0,
            linewidth=0, facecolor=s_pal[state]))
        i = run_end


# =============================================================================
# SECTION 1: BIC SCAN
# =============================================================================
print("=" * 65)
print("SECTION 1: BIC scan")
print("=" * 65)

print("[load] Reading PCA coordinates…")
df_pca  = pd.read_csv(PCA_CSV)
df_pca  = _normalise_condition_case(df_pca, REFERENCE_COND)
pc_cols = [f"PC{i+1}" for i in range(N_PCS) if f"PC{i+1}" in df_pca.columns]

if not pc_cols:
    raise RuntimeError(f"No PC columns found in {PCA_CSV}.")

k_meta = [c for c in KCOLS if c in df_pca.columns]
print(f"  Shape : {df_pca.shape}")
print(f"  PCs   : {pc_cols}")
print(f"  Conds : {df_pca['Condition'].value_counts().to_dict()}")

# ── Filter to reference condition ────────────────────────────────────────────
df_ref = df_pca[df_pca["Condition"].str.lower() == REFERENCE_COND.lower()].copy()
if df_ref.empty:
    raise RuntimeError(
        f"No rows for '{REFERENCE_COND}' in {PCA_CSV}. "
        f"Available: {df_pca['Condition'].unique().tolist()}")

print(f"\n[reference] '{REFERENCE_COND}':  "
      f"{len(df_ref)} windows  |  "
      f"{df_ref['Individual'].nunique()} animals")

# ── Build ordered sequences ───────────────────────────────────────────────────
seqs_scan, lengths_scan = [], []

for grp_keys, grp in df_ref.groupby(k_meta, dropna=False, sort=True):
    grp = grp.sort_values("WindowStart") if "WindowStart" in grp.columns else grp
    if N_PER_ANIMAL is not None and len(grp) > N_PER_ANIMAL:
        grp = grp.sample(n=N_PER_ANIMAL, random_state=RANDOM_STATE)
        grp = grp.sort_values("WindowStart") if "WindowStart" in grp.columns else grp
    mat = grp[pc_cols].to_numpy(float)
    if len(mat) < MIN_WIN or not np.all(np.isfinite(mat)):
        print(f"  [skip] {grp_keys}  — {len(mat)} windows or non-finite values")
        continue
    seqs_scan.append(mat)
    lengths_scan.append(len(mat))

if not seqs_scan:
    raise RuntimeError("No valid sequences for BIC scan.")

X_scan = np.vstack(seqs_scan)
N_scan = X_scan.shape[0]
print(f"\n[sequences] {len(lengths_scan)} animals  |  "
      f"{N_scan} windows  |  D={X_scan.shape[1]}")
print(f"  Lengths: min={min(lengths_scan)}  "
      f"max={max(lengths_scan)}  median={int(np.median(lengths_scan))}")

# ── Scan ─────────────────────────────────────────────────────────────────────
print(f"\n[BIC scan] K={MIN_STATES}…{MAX_STATES}  "
      f"{N_INIT_SCAN} restarts  {N_ITER_SCAN} iters  cov={COV_TYPE}\n")

bic_rows = []
for K in range(MIN_STATES, MAX_STATES + 1):
    best_ll, best_m, all_lls = -np.inf, None, []
    for seed in range(N_INIT_SCAN):
        try:
            m, ll = _fit_one(X_scan, lengths_scan, K, COV_TYPE, N_ITER_SCAN, seed)
            if np.isfinite(ll):
                all_lls.append(ll)
                if ll > best_ll:
                    best_ll, best_m = ll, m
        except Exception:
            pass
    if best_m is None:
        print(f"  K={K:2d}  ALL restarts failed — skipping")
        continue
    npar    = _n_params(K, X_scan.shape[1], COV_TYPE)
    bic     = -2 * best_ll + npar * np.log(N_scan) / N_scan
    aic     = -2 * best_ll + 2 * npar / N_scan
    n_degen = int((np.diag(best_m.transmat_) > 0.999).sum())
    ll_std  = float(np.std(all_lls)) if len(all_lls) > 1 else np.nan
    bic_rows.append(dict(K=K, LL=best_ll, LL_std=ll_std,
                         N_converged=len(all_lls), BIC=bic, AIC=aic,
                         N_params=npar, N_degen=n_degen))
    stable = "stable" if (np.isfinite(ll_std) and ll_std < 500) else "UNSTABLE"
    print(f"  K={K:2d}  LL={best_ll:10.4f} ±{ll_std:7.1f}  "
          f"({len(all_lls)}/{N_INIT_SCAN} ok)  "
          f"BIC={bic:8.4f}  AIC={aic:8.4f}  "
          f"degen={n_degen}  [{stable}]")

if not bic_rows:
    raise RuntimeError("BIC scan produced no results.")

bic_df  = pd.DataFrame(bic_rows)
bic_path = os.path.join(OUTDIR, "bic_scan_results.csv")
bic_df.to_csv(bic_path, index=False, float_format="%.4f")
print(bic_path)

best_k_bic = int(bic_df.loc[bic_df["BIC"].idxmin(), "K"])
print(f"\n[BIC scan] BIC-minimum K = {best_k_bic}")
print(f"  → Review bic_scan_results.csv, then set K_FINAL above.")

# =============================================================================
# SECTION 2: FINAL FIT
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 2: Final fit")
print("=" * 65)

# ── Build sequences for ALL conditions ───────────────────────────────────────
seqs_all, lengths_all, valid_group_idx = [], [], []
n_imputed = 0

for grp_keys, grp in df_pca.groupby(k_meta, dropna=False, sort=True):
    grp_s = grp.sort_values("WindowStart") if "WindowStart" in grp.columns else grp
    mat   = grp_s[pc_cols].to_numpy(float)
    if len(mat) < MIN_WIN:
        continue
    n_nan = (~np.isfinite(mat)).sum()
    if n_nan > 0:
        n_imputed += n_nan
        mat = np.nan_to_num(mat, nan=0.0)
    seqs_all.append(mat)
    lengths_all.append(len(mat))
    valid_group_idx.append(grp_s.index[:len(mat)])

X_all = np.vstack(seqs_all)
print(f"\n[sequences] {len(lengths_all)} animals  |  "
      f"{X_all.shape[0]} windows  |  D={X_all.shape[1]}")
print(f"  Lengths: min={min(lengths_all)}  "
      f"max={max(lengths_all)}  median={int(np.median(lengths_all))}")
if n_imputed:
    print(f"  [warn] {n_imputed} NaN PC values imputed to 0.0")

# ── Multiple restarts + collapse guard ───────────────────────────────────────
print(f"\n[fit] K={K_FINAL}  {N_INIT_FINAL} restarts  {N_ITER_FINAL} iters")
all_fits = []
for seed in range(N_INIT_FINAL):
    try:
        m, ll = _fit_one(X_all, lengths_all, K_FINAL, COV_TYPE, N_ITER_FINAL, seed)
        all_fits.append({"seed": seed, "model": m, "ll": ll,
                         "converged": m.monitor_.converged})
        print(f"  seed={seed:2d}  LL={ll:12.4f}  converged={m.monitor_.converged}")
    except Exception as e:
        all_fits.append({"seed": seed, "model": None, "ll": np.nan,
                         "converged": False})
        print(f"  seed={seed:2d}  FAILED: {e}")

finite_lls = [r["ll"] for r in all_fits if np.isfinite(r["ll"])]
if not finite_lls:
    raise RuntimeError("All restarts failed.")

ll_med    = np.median(finite_lls)
ll_sd     = np.std(finite_lls)
ll_cutoff = ll_med + 2 * ll_sd

outliers = [r["seed"] for r in all_fits
            if np.isfinite(r["ll"]) and r["ll"] > ll_cutoff]
if outliers:
    print(f"\n  [collapse guard] median={ll_med:.1f}  SD={ll_sd:.1f}  "
          f"cutoff={ll_cutoff:.1f}  outlier seeds: {outliers}")
    clean = [r for r in all_fits
             if np.isfinite(r["ll"]) and r["ll"] <= ll_cutoff
             and r["model"] is not None]
    best = (max(clean, key=lambda r: r["ll"])
            if clean
            else max(all_fits, key=lambda r: r["ll"] if np.isfinite(r["ll"]) else -np.inf))
else:
    best = max(all_fits, key=lambda r: r["ll"] if np.isfinite(r["ll"]) else -np.inf)
    print(f"\n  [collapse guard] All seeds within 2 SD "
          f"(median={ll_med:.1f}  SD={ll_sd:.1f})")

best_m, best_ll = best["model"], best["ll"]
print(f"\n  Using seed={best['seed']}  LL={best_ll:.4f}  "
      f"converged: {sum(r['converged'] for r in all_fits)}/{N_INIT_FINAL}")

pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in all_fits]
             ).to_csv(os.path.join(OUTDIR, "hmm_restart_diagnostics.csv"), index=False)

K = best_m.n_components

# ── Viterbi decode ────────────────────────────────────────────────────────────
print("\n[decode] Viterbi…")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _, all_states = best_m.decode(X_all, lengths_all, algorithm="viterbi")

df_out = df_pca.copy()
df_out["HMM_state"] = pd.NA
cursor = 0
for idx, length in zip(valid_group_idx, lengths_all):
    df_out.loc[idx, "HMM_state"] = all_states[cursor: cursor + length]
    cursor += length

assert cursor == len(all_states)
df_out["HMM_state"] = df_out["HMM_state"].astype("Int64")
print(f"  {df_out['HMM_state'].notna().sum()} / {len(df_out)} windows decoded")

# ── Dwell smoothing ───────────────────────────────────────────────────────────
if MIN_DWELL > 1:
    valid_mask = df_out["HMM_state"].notna()

    def _smooth_group(g):
        return pd.Series(
            smooth_states(g.to_numpy(dtype=int), MIN_DWELL), index=g.index)

    smoothed = (df_out.loc[valid_mask]
                      .groupby("Individual", group_keys=False)["HMM_state"]
                      .apply(_smooth_group))
    df_out.loc[valid_mask, "HMM_state"] = smoothed
    df_out["HMM_state"] = df_out["HMM_state"].astype("Int64")
    print(f"[smooth] MIN_DWELL={MIN_DWELL} windows")

# ── Rare state pruning ────────────────────────────────────────────────────────
sub = df_out.dropna(subset=["HMM_state"]).copy()
sub["HMM_state"] = sub["HMM_state"].astype(int)
min_fr = max(1, int(RARE_FRAC * len(sub)))
vc     = sub["HMM_state"].value_counts()
rare   = sorted([int(s) for s in vc.index if vc[s] < min_fr])

if rare:
    keep = [s for s in range(K) if s not in rare]
    sm   = np.array([
        sub.loc[sub["HMM_state"] == s, pc_cols].median().to_numpy(float)
        if s in sub["HMM_state"].values else np.zeros(len(pc_cols))
        for s in range(K)])
    mapping_rare = {s: s for s in range(K)}
    for r in rare:
        d = np.linalg.norm(sm[keep] - sm[r], axis=1)
        mapping_rare[r] = keep[int(np.argmin(d))]
    uniq      = np.sort(np.unique(list(mapping_rare.values())))
    remap     = {old: new for new, old in enumerate(uniq)}
    final_map = {s: remap[mapping_rare[s]] for s in range(K)}
    K_new     = len(uniq)
    valid2    = df_out["HMM_state"].notna()
    df_out.loc[valid2, "HMM_state"] = (
        df_out.loc[valid2, "HMM_state"].astype(int).map(final_map))
    df_out["HMM_state"] = df_out["HMM_state"].astype("Int64")
    print(f"[prune] {len(rare)} rare state(s) {rare} merged  K: {K} → {K_new}")
    K = K_new
else:
    print("[prune] no rare states")

# ── Empirical means + transition matrix ───────────────────────────────────────
sub2 = df_out.dropna(subset=["HMM_state"]).copy()
sub2["HMM_state"] = sub2["HMM_state"].astype(int)

empirical_means = np.array([
    sub2.loc[sub2["HMM_state"] == s, pc_cols].mean().to_numpy(float)
    for s in range(K)])

T_emp = np.zeros((K, K))
for _, grp in sub2.groupby(k_meta, dropna=False, sort=True):
    grp = grp.sort_values("WindowStart") if "WindowStart" in grp.columns else grp
    seq = grp["HMM_state"].to_numpy(int)
    for a, b in zip(seq[:-1], seq[1:]):
        T_emp[a, b] += 1
row_sums = T_emp.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
T_emp /= row_sums
print(f"[empirical] Transition matrix + state means computed (K={K})")

# ── Occupancy + transition rates per animal ───────────────────────────────────
occ_rows, tr_rows = [], []
for grp_keys, grp in sub2.groupby(k_meta, dropna=False, sort=True):
    n   = len(grp)
    row = {c: (grp_keys[i] if isinstance(grp_keys, tuple) else grp_keys)
           for i, c in enumerate(k_meta)}
    for s in range(K):
        row[f"state_{s}_frac"] = (grp["HMM_state"] == s).sum() / n
    pv  = np.array([row[f"state_{s}_frac"] for s in range(K)])
    pp  = pv[pv > 0]
    row["state_entropy"] = float(-np.sum(pp * np.log2(pp))) if pp.size else 0.0
    occ_rows.append(row)

    grp2 = grp.sort_values("WindowStart") if "WindowStart" in grp.columns else grp
    seq  = grp2["HMM_state"].to_numpy(int)
    meta = {c: (grp_keys[i] if isinstance(grp_keys, tuple) else grp_keys)
            for i, c in enumerate(k_meta)}
    for s in range(K):
        for t in range(K):
            nt = int(np.sum((seq[:-1] == s) & (seq[1:] == t)))
            tr_rows.append({**meta, "from_state": s, "to_state": t,
                            "n_transitions": nt, "rate": nt / max(1, n - 1)})

occ_df   = pd.DataFrame(occ_rows)
trans_df = pd.DataFrame(tr_rows)

# ── Save fit outputs ──────────────────────────────────────────────────────────
print("\n[save - fit]")
s_names = [f"S{s}" for s in range(K)]
s_pal   = _state_palette(K)

for fname, dframe in [
    ("hmm_decoded_all_windows.csv",        df_out),
    ("hmm_state_occupancy_per_animal.csv", occ_df),
    ("hmm_transition_rates_per_animal.csv", trans_df),
]:
    path = os.path.join(OUTDIR, fname)
    dframe.to_csv(path, index=False, float_format="%.4f")
    print(path)

path = os.path.join(OUTDIR, "hmm_state_means.csv")
pd.DataFrame(empirical_means, columns=pc_cols,
             index=[f"state_{s}" for s in range(K)]
             ).to_csv(path, float_format="%.4f")
print(path)

path = os.path.join(OUTDIR, "hmm_transition_matrix.csv")
pd.DataFrame(T_emp, index=s_names, columns=s_names
             ).to_csv(path, float_format="%.4f")
print(path)

occ_means = occ_df[[f"state_{s}_frac" for s in range(K)]].mean()
print(f"\n  K={K}  LL={best_ll:.4f}  MIN_DWELL={MIN_DWELL}")
for s in range(K):
    print(f"    S{s}: mean occupancy {occ_means[f'state_{s}_frac']:.3f}")

# =============================================================================
# SECTION 3: PLOTS
# =============================================================================
print("\n" + "=" * 65)
print("SECTION 3: Plots")
print("=" * 65)

conditions = sorted(sub2["Condition"].dropna().unique(),
                    key=lambda c: (c.lower() != REFERENCE_COND.lower(), c))

ANIMAL_LABEL_COLS = [c for c in ["Experiment_ID", "Individual"] if c in sub2.columns]
sub2["_akey"]   = sub2[k_meta].astype(str).agg("|".join, axis=1)
sub2["_alabel"] = sub2[ANIMAL_LABEL_COLS].astype(str).agg(" | ".join, axis=1)
exp_id_idx      = k_meta.index("Experiment_ID") if "Experiment_ID" in k_meta else None


def _get_exp_id(akey: str) -> str:
    if exp_id_idx is None:
        return ""
    parts = akey.split("|")
    return parts[exp_id_idx] if len(parts) > exp_id_idx else ""


# ── 3a. Full ethograms ────────────────────────────────────────────────────────
print("\n[ethograms - full]")
rng_eth = random.Random(RANDOM_STATE)


def _sort_key_eth(k, cs):
    eid    = cs.loc[cs["_akey"] == k, "Experiment_ID"].iloc[0] \
             if "Experiment_ID" in cs.columns else ""
    has_he = bool(_he_spans_sec(eid))
    return (not has_he, -(cs["_akey"] == k).sum())


for cond in conditions:
    cs        = sub2[sub2["Condition"] == cond]
    key_label = (cs[["_akey", "_alabel"]]
                 .drop_duplicates("_akey")
                 .set_index("_akey")["_alabel"]
                 .to_dict())
    all_keys = list(key_label.keys())
    n_total  = len(all_keys)
    if not all_keys:
        continue
    chosen = rng_eth.sample(all_keys, min(ETHOGRAM_N, n_total))
    chosen = sorted(chosen, key=lambda k: _sort_key_eth(k, cs))
    n_anim = len(chosen)

    with mpl.rc_context(PUB_STYLE):
        fig, axes = plt.subplots(n_anim, 1,
                                 figsize=(14, max(3.5, n_anim * 0.75)),
                                 sharex=False)
        if n_anim == 1:
            axes = [axes]

        for ax, akey in zip(axes, chosen):
            adf = cs[cs["_akey"] == akey]
            if "WindowStart" in adf.columns:
                adf = adf.sort_values("WindowStart")
            seq   = adf["HMM_state"].to_numpy(int)
            ws    = (adf["WindowStart"].to_numpy(float)
                     if "WindowStart" in adf.columns
                     else np.arange(len(seq), dtype=float) * STRIDE_RS_FR)
            t_sec = ws / FS_RS
            t0, t1 = float(t_sec[0]), float(t_sec[-1]) + 1.0

            _draw_raster(ax, seq, t_sec, s_pal)
            ax.set_xlim(t0, t1)
            ax.set_ylim(0, 1)

            for he_s, he_e in _he_spans_sec(_get_exp_id(akey)):
                for x_val in (he_s, he_e):
                    if t0 <= x_val <= t1:
                        ax.axvline(x_val, color="white",   lw=4.0, zorder=3)
                        ax.axvline(x_val, color="#FF6600", lw=2.0,
                                   ls="--", zorder=4)

            ax.set_yticks([0.5])
            ax.set_yticklabels([key_label[akey][:38]], fontsize=6)
            if ax is axes[-1]:
                ax.set_xlabel("Time (s)", fontsize=8)
            else:
                ax.set_xticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        hs = [mpatches.Patch(fc=s_pal[s], label=s_names[s]) for s in range(K)]
        hs.append(mlines.Line2D([0], [0], color="#FF6600", lw=2.0,
                                 ls="--", label="HE boundary"))
        fig.legend(handles=hs, loc="lower center", fontsize=7,
                   ncol=K + 1, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(
            f"Ethograms — {cond.replace('_', ' ')}  (n={n_anim} of {n_total})\n"
            f"Orange dashed lines = house expansion boundaries",
            fontsize=10, y=1.005)
        try:
            fig.tight_layout(rect=[0, 0.04, 1.0, 0.97])
        except Exception:
            pass
        _savefig(fig, f"05_ethogram_{cond}")
        print(f"  {cond} → 05_ethogram_{cond}.png/svg")


# ── 3b. HE transient panels ───────────────────────────────────────────────────
print("\n[ethograms - HE transients]")

for cond in conditions:
    cs = sub2[sub2["Condition"] == cond]
    animals = []
    for exp_id, grp in cs.groupby("Experiment_ID", sort=True):
        spans = _he_spans_sec(exp_id)
        if not spans:
            continue
        grp_s = grp.sort_values("WindowStart") if "WindowStart" in grp.columns else grp
        seq   = grp_s["HMM_state"].to_numpy(int)
        ws    = (grp_s["WindowStart"].to_numpy(float)
                 if "WindowStart" in grp_s.columns
                 else np.arange(len(seq), dtype=float) * STRIDE_RS_FR)
        animals.append({"exp_id": exp_id, "seq": seq,
                        "t_sec": ws / FS_RS, "spans": spans})

    if not animals:
        print(f"  {cond}: no HE data — skipping")
        continue

    n_anim = len(animals)
    print(f"  {cond}: {n_anim} animals with HE data")

    with mpl.rc_context(PUB_STYLE):
        fig, axes = plt.subplots(
            n_anim, 2,
            figsize=(12, max(3.5, n_anim * 0.85)),
            gridspec_kw={"wspace": 0.06, "hspace": 0.10})
        if n_anim == 1:
            axes = axes[np.newaxis, :]

        for row_i, anim in enumerate(animals):
            seq   = anim["seq"]
            t_sec = anim["t_sec"]
            t_min = float(t_sec[0])
            t_max = float(t_sec[-1]) + 1.0
            he_s, he_e = anim["spans"][0]
            label = str(anim["exp_id"])[:32]

            ax_l   = axes[row_i, 0]
            win_l0 = max(t_min, he_s - PRE_SEC)
            mask_l = (t_sec >= win_l0) & (t_sec <= he_s + 1)
            if mask_l.any():
                _draw_raster(ax_l, seq[mask_l], t_sec[mask_l], s_pal)
            ax_l.axvline(he_s, color="white",   lw=4.0, zorder=3)
            ax_l.axvline(he_s, color="#FF6600", lw=2.5, ls="--", zorder=4)
            x_ticks = np.arange(np.ceil((win_l0 - he_s) / 20) * 20, 1, 20)
            ax_l.set_xticks(x_ticks + he_s)
            ax_l.set_xticklabels([f"{int(x)}" for x in x_ticks], fontsize=6)
            ax_l.set_xlim(win_l0, he_s + 1)
            ax_l.set_ylim(0, 1)
            ax_l.set_yticks([0.5])
            ax_l.set_yticklabels([label], fontsize=6)
            if row_i == n_anim - 1:
                ax_l.set_xlabel("Time relative to HE onset (s)", fontsize=7)
            else:
                ax_l.set_xticklabels([])
            for sp in ax_l.spines.values():
                sp.set_visible(False)
            if row_i == 0:
                ax_l.set_title(f"Onset  (−{PRE_SEC} s → 0)", fontsize=9,
                               fontweight="bold")

            ax_r   = axes[row_i, 1]
            win_r1 = min(t_max, he_e + POST_SEC)
            mask_r = (t_sec >= he_e - 1) & (t_sec <= win_r1)
            if mask_r.any():
                _draw_raster(ax_r, seq[mask_r], t_sec[mask_r], s_pal)
            ax_r.axvline(he_e, color="white",   lw=4.0, zorder=3)
            ax_r.axvline(he_e, color="#FF6600", lw=2.5, ls="--", zorder=4)
            x_ticks_r = np.arange(0, np.floor((win_r1 - he_e) / 20) * 20 + 1, 20)
            ax_r.set_xticks(x_ticks_r + he_e)
            ax_r.set_xticklabels([f"{int(x)}" for x in x_ticks_r], fontsize=6)
            ax_r.set_xlim(he_e - 1, win_r1)
            ax_r.set_ylim(0, 1)
            ax_r.set_yticks([])
            if row_i == n_anim - 1:
                ax_r.set_xlabel("Time relative to HE offset (s)", fontsize=7)
            else:
                ax_r.set_xticklabels([])
            for sp in ax_r.spines.values():
                sp.set_visible(False)
            if row_i == 0:
                ax_r.set_title(f"Offset  (0 → +{POST_SEC} s)", fontsize=9,
                               fontweight="bold")

        hs = [mpatches.Patch(fc=s_pal[s], label=s_names[s]) for s in range(K)]
        hs.append(mlines.Line2D([0], [0], color="#FF6600", lw=2.0,
                                 ls="--", label="HE boundary"))
        fig.legend(handles=hs, loc="lower center", fontsize=7,
                   ncol=K + 1, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(
            f"HE transients — {cond.replace('_', ' ')}  (n={n_anim})\n"
            f"Left: {PRE_SEC} s before onset  |  Right: {POST_SEC} s after offset",
            fontsize=10, y=1.01)
        try:
            fig.tight_layout(rect=[0, 0.05, 1.0, 0.97])
        except Exception:
            pass
        _savefig(fig, f"05b_he_transients_{cond}")
        print(f"  {cond} → 05b_he_transients_{cond}.png/svg")

sub2.drop(columns=["_akey", "_alabel"], errors="ignore", inplace=True)


# ── 3c. Event-based transition matrices → CSV + chord diagrams ───────────────
print(f"\n[transitions] Event-based  (min_dwell={MIN_DWELL_SEC} s)…")


def _extract_events(df_animal: pd.DataFrame, min_dwell: int) -> list:
    if df_animal.empty:
        return []
    seq = df_animal["HMM_state"].to_numpy(int)
    t   = (df_animal["WindowStart"].to_numpy(float) / FS_RS
           if "WindowStart" in df_animal.columns
           else np.arange(len(seq), dtype=float))
    runs = [(seq[0], 1)]
    for s in seq[1:]:
        if s == runs[-1][0]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((s, 1))
    merged = []
    for st, ln in runs:
        if ln < min_dwell and merged:
            pst, pln = merged.pop()
            merged.append((pst, pln + ln))
        else:
            merged.append((st, ln))
    cursor, events = 0, []
    for st, ln in merged:
        t_s = float(t[cursor])
        t_e = float(t[min(cursor + ln - 1, len(t) - 1)]) + 1.0
        events.append((st, t_s, t_e))
        cursor += ln
    return events


def _count_matrix(df_sub: pd.DataFrame, K: int, min_dwell: int) -> np.ndarray:
    T = np.zeros((K, K), dtype=float)
    for _, grp in df_sub.groupby(k_meta, dropna=False, sort=True):
        grp = grp.sort_values("WindowStart") if "WindowStart" in grp.columns else grp
        sts = [e[0] for e in _extract_events(grp, min_dwell)]
        for a, b in zip(sts[:-1], sts[1:]):
            T[a, b] += 1
    return T


def _prob_matrix(counts: np.ndarray) -> np.ndarray:
    rs = counts.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return counts / rs


all_counts: dict = {}
all_probs: dict  = {}

sub3 = df_out.dropna(subset=["HMM_state"]).copy()
sub3["HMM_state"] = sub3["HMM_state"].astype(int)

for cond in conditions:
    C = _count_matrix(sub3[sub3["Condition"] == cond], K, MIN_DWELL_SEC)
    P = _prob_matrix(C.copy())
    all_counts[cond] = C
    all_probs[cond]  = P
    print(f"  {cond}: {int(C.sum())} transitions")
    path = os.path.join(OUTDIR, f"transition_matrix_events_{cond}.csv")
    pd.DataFrame(
        np.hstack([C, P]),
        index   = s_names,
        columns = [f"count_{s}" for s in s_names] + [f"prob_{s}" for s in s_names]
    ).to_csv(path)
    print(f"  Saved: transition_matrix_events_{cond}.csv")

# Chord diagrams
print("\n[chord] Drawing…")
state_cmap = {s_names[i]: s_pal[i] for i in range(K)}


def _draw_chord(P: np.ndarray, fname: str):
    P_plot = P.copy()
    np.fill_diagonal(P_plot, 0.0)
    P_plot[P_plot < CHORD_THRESHOLD] = 0.0
    if P_plot.sum() == 0:
        print(f"  [skip] {fname}: no links above {CHORD_THRESHOLD:.0%}")
        return
    rs = P_plot.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P_df = pd.DataFrame(P_plot / rs, index=s_names, columns=s_names)
    try:
        circos = Circos.initialize_from_matrix(
            P_df,
            space     = 5,
            cmap      = state_cmap,
            label_kws = dict(size=12),
            link_kws  = dict(ec="black", lw=0.5, direction=1),
        )
        for ext in ("png", "svg"):
            circos.savefig(os.path.join(CHORD_DIR, f"{fname}.{ext}"))
        plt.close("all")
        print(f"  chord_diagrams/{fname}.png/svg")
    except Exception as e:
        print(f"  [warn] Chord failed for '{fname}': {e}")


for cond in conditions:
    _draw_chord(all_probs[cond], f"chord_{cond}")

T_pool = sum(all_counts.values())
_draw_chord(_prob_matrix(T_pool.copy()), "chord_all_conditions")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print(f"""
{'='*65}
HMM pipeline complete

  BIC scan  → {OUTDIR}/bic_scan_results.csv
              BIC-minimum K = {best_k_bic}  (K_FINAL used = {K_FINAL})

  Fit       → K={K}  LL={best_ll:.4f}  MIN_DWELL={MIN_DWELL}
  Mean state occupancy:
    {'  '.join(f"S{s}: {occ_means[f'state_{s}_frac']:.3f}" for s in range(K))}

  Outputs → {OUTDIR}/
    hmm_decoded_all_windows.csv
    hmm_state_occupancy_per_animal.csv
    hmm_transition_rates_per_animal.csv
    hmm_state_means.csv
    hmm_transition_matrix.csv
    hmm_restart_diagnostics.csv
    transition_matrix_events_<cond>.csv
    05_ethogram_<cond>.png/svg
    05b_he_transients_<cond>.png/svg
    chord_diagrams/chord_<cond>.png/svg
    chord_diagrams/chord_all_conditions.png/svg
{'='*65}
""")

# Expose to session for downstream use
hmm_model       = best_m
hmm_decoded     = df_out
hmm_occupancy   = occ_df
hmm_transitions = trans_df