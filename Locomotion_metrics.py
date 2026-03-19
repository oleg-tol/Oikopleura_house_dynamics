#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-animal locomotion metrics and centerline skeletons from DLC CSVs.

Metrics saved to OUTPUT_DIR/wide_tables/:
    com_x/y_wide            – centre-of-mass trajectories
    velocity_wide           – deadbanded speed (µm/s)
    avg_curvature_wide      – mean body curvature
    avg_tan_angles_wide     – mean tangent angles
    quirkiness_wide         – lateral deviation
    tailbeat_freq/amp_wide  – dominant frequency & amplitude
    omega_abs_wide          – absolute body rotation rate
    window_*_wide           – windowed velocity/acc/tortuosity/etc.
    svd_complexity_wide     – Takens-embedding SVD entropy
    state_ts_wide           – RDQ state (0=quiescent,1=dwelling,2=roaming)
    bout_summary            – scalar bout stats per animal
    per_animal_feature_matrix – all scalar means, one row per animal
"""
import os, re
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from itertools import combinations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter, detrend
from scipy.stats import mannwhitneyu, kruskal

try:
    from scipy.interpolate import PchipInterpolator
    _HAVE_PCHIP = True
except ImportError:
    _HAVE_PCHIP = False

# =============================================================================
# CONFIG
# =============================================================================
ROOT_DIR   = "YOUR FOLDER"
OUTPUT_DIR = os.path.join(ROOT_DIR, "Data_hmm")
os.makedirs(OUTPUT_DIR, exist_ok=True)

WIDE_DIR  = os.path.join(OUTPUT_DIR, "wide_tables")
STATE_DIR = os.path.join(OUTPUT_DIR, "state_rdq")
os.makedirs(WIDE_DIR,  exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

CONTROL_NAME  = "control"
CONTROL_COLOR = "#AED6F1"

OTHER_COLORS = ["#F1948A", "#82E0AA", "#F7DC6F", "#BB8FCE", "#F0B27A", "#A9CCE3"]

FPS_NATIVE = 30.0
DT_NATIVE  = 1.0 / FPS_NATIVE

APPLY_RESAMPLE = True
RESAMPLE_RATE  = 5
DT_RS          = RESAMPLE_RATE / FPS_NATIVE
FS_RS          = 1.0 / DT_RS

WIN_SEC_STD   = 3.0
STRIDE_SEC    = 1.0
WIN_RS_FRAMES = int(round(WIN_SEC_STD * FS_RS))
STRIDE_RS_FR  = int(round(STRIDE_SEC * FS_RS))
WIN_NV_FRAMES = int(round(WIN_SEC_STD * FPS_NATIVE))

MIN_CONTINUOUS_SEC    = 3.0
MIN_CONTINUOUS_FRAMES = int(round(MIN_CONTINUOUS_SEC * FPS_NATIVE))

LIKELIHOOD_MIN     = 0.99
IQR_THRESHOLD      = 2.0
DISTANCE_THRESHOLD = 100.0
SAVGOL_WL          = 15
SAVGOL_ORDER       = 2

DEADBAND_K_MAD  = 3.0
HYST_IN_SCALE   = 1.20
HYST_OUT_SCALE  = 0.80
PIXEL_TO_UM     = 11.56
VEL_CLIP_MAX    = 6000

MIN_VALID_FRAC_ANIMAL = 0.05

COMPLEXITY_WIN_EMBED      = 9
COMPLEXITY_MIN_VALID_FRAC = 0.70

QUIESCENCE_THRESHOLD_UM_S = 50.0
RDQ_THRESHOLD_COND = CONTROL_NAME

BOUT_MIN_FRAMES = 3
BOUT_MAX_GAP    = 2

HE_MAX_TRACES    = 500
HE_TRACE_ALPHA   = 0.07
HE_MAX_Y         = 0.65
HE_MAX_ANGLE_RAD = 2.0
HE_MIN_LEN_PX    = 8.0
HE_MAX_LEN_PX    = 250.0
HE_K             = 21

_KCOLS = ["Experiment_ID", "Condition", "Individual"]

SIGNIFICANCE_LEVELS = [
    (0.001, "***"),
    (0.01,  "**"),
    (0.05,  "*"),
    (1.0,   "ns"),
]

def _sig_label(p: float) -> str:
    for threshold, label in SIGNIFICANCE_LEVELS:
        if p <= threshold:
            return label
    return "ns"

# =============================================================================
# HOUSE-EXPANSION FRAME RANGES AFTER MANUAL SELECTION
# =============================================================================
HOUSE_EXPANSION_FRAMES: Dict[str, List[Tuple[int, int]]] = {
    # --- Control ---
    "20250527_100240_1_15m0s_None_None_None": [(1740, 2820)],
    "20250527_103744_1_15m0s_None_None_None": [(2550, 3150)],
    "20250527_111628_1_15m0s_None_None_None": [(1770, 2190)],
    "20250602_141106_1_15m0s_None_None_None": [(2070, 2880)],
    "20250623_102938_1_15m0s_None_None_None": [(270,  900)],
    "20250623_150448_1_15m0s_None_None_None": [(2880, 3810)],
    "20250623_152301_1_15m0s_None_None_None": [(240,  660)],
    "20250818_101132_1_15m0s_None_None_None": [(870,  1290)],
    "20250818_103120_1_15m0s_None_None_None": [(1860, 2400)],
    "20250818_103121_1_15m0s_None_None_None": [(240,  750)],
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
# CONDITION DETECTION
# =============================================================================
def _collect_csvs(root_dir: str) -> List[str]:
    paths = []
    if not os.path.isdir(root_dir):
        return paths
    for dp, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".csv"):
                paths.append(os.path.join(dp, fn))
    return sorted(set(paths))


def _condition_from_path(fp: str, root: str) -> Optional[str]:
    rel   = os.path.relpath(fp, root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) >= 2 else None


def _build_condition_order_and_palette(conditions: List[str]) -> Tuple[List[str], Dict[str, str]]:
    ctrl   = [c for c in conditions if c.lower() == CONTROL_NAME]
    others = sorted([c for c in conditions if c.lower() != CONTROL_NAME])
    ordered = ctrl + others
    palette: Dict[str, str] = {}
    idx = 0
    for c in ordered:
        if c.lower() == CONTROL_NAME:
            palette[c] = CONTROL_COLOR
        else:
            palette[c] = OTHER_COLORS[idx % len(OTHER_COLORS)]
            idx += 1
    return ordered, palette


# =============================================================================
# UTILITIES
# =============================================================================
def safe_to_numeric(df_like: pd.DataFrame) -> pd.DataFrame:
    return df_like.apply(pd.to_numeric, errors="coerce")

def build_time_cols(T: int) -> List[str]:
    return [f"Time_{i:03d}" for i in range(T)]

def time_cols_from(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if isinstance(c, str) and c.startswith("Time_")]

def read_filtered_csv(file_path: str):
    hdr         = pd.read_csv(file_path, nrows=3, header=None)
    individuals = hdr.iloc[1, 1::3].fillna("Unknown").astype(str).tolist()
    bodyparts   = hdr.iloc[2, 1::3].fillna("").astype(str).tolist()
    raw = pd.read_csv(file_path, header=None, skiprows=3, low_memory=False)
    cx  = safe_to_numeric(raw.iloc[:, 1::3]).values
    cy  = safe_to_numeric(raw.iloc[:, 2::3]).values
    lk  = safe_to_numeric(raw.iloc[:, 3::3]).values
    bodyparts = [(bp.strip() if isinstance(bp, str) and bp.strip() else f"BP_{i+1}")
                 for i, bp in enumerate(bodyparts)]
    return individuals, bodyparts, cx, cy, lk


def iqr_outlier_removal(arr: np.ndarray, thr: float) -> np.ndarray:
    if arr.size == 0:
        return arr
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    iqr = q75 - q25
    out = arr.copy()
    out[(out < q25 - thr*iqr) | (out > q75 + thr*iqr)] = np.nan
    return out


def filter_large_steps(cx: np.ndarray, cy: np.ndarray, thr: float):
    if cx.shape[0] <= 1:
        return cx, cy
    dist = np.sqrt(np.diff(cx, axis=0)**2 + np.diff(cy, axis=0)**2)
    keep = np.vstack([np.ones((1, dist.shape[1]), dtype=bool), dist <= thr])
    cx2, cy2 = cx.copy(), cy.copy()
    cx2[~keep] = np.nan
    cy2[~keep] = np.nan
    return cx2, cy2


def mask_short_runs_native(arr: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or arr.size == 0:
        return arr
    out = arr.copy()
    T, N = out.shape
    for j in range(N):
        v      = np.isfinite(out[:, j]).astype(int)
        padded = np.concatenate([[0], v, [0]])
        d      = np.diff(padded)
        for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            if (e - s) < min_len:
                out[s:e, j] = np.nan
    return out


def interpolate_bracketed(arr: np.ndarray) -> np.ndarray:
    out = np.empty_like(arr, dtype=float)
    for j in range(arr.shape[1]):
        s = pd.Series(arr[:, j], dtype="float64").interpolate(
            method="linear", limit_area="inside")
        out[:, j] = s.values
    return out


def resample_axis0(arr: np.ndarray, rate: int) -> np.ndarray:
    return arr if (rate is None or rate <= 1) else arr[::rate]


def row_nanmean(arr: np.ndarray) -> np.ndarray:
    return pd.DataFrame(arr).mean(axis=1, skipna=True).to_numpy()


def split_trunk_tail(n_cols: int) -> Tuple[np.ndarray, np.ndarray]:
    trunk = np.arange(min(4, n_cols))
    tail  = np.arange(4, n_cols) if n_cols > 4 else np.array([], dtype=int)
    return trunk, tail


def skeleton_indices(n_cols: int) -> np.ndarray:
    if n_cols == 0:
        return np.array([], dtype=int)
    _, tail = split_trunk_tail(n_cols)
    return np.array([0], dtype=int) if tail.size == 0 else np.concatenate(([0], tail))


# =============================================================================
# TAIL-BEAT HELPERS
# =============================================================================
def _angle_of(vec): return np.arctan2(vec[..., 1], vec[..., 0])
def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, np.nan, n)


def tail_trunk_signals(full_xy, trunk_idx, tail_idx):
    T, n = full_xy.shape[:2]
    trunk_angle  = np.full(T, np.nan)
    tail_lateral = np.full(T, np.nan)
    if len(trunk_idx) == 0 or n == 0:
        return trunk_angle, tail_lateral
    anchor    = 0 if 0 in trunk_idx else trunk_idx[0]
    base      = full_xy[:, anchor, :]
    others    = [j for j in trunk_idx if j != anchor] or [anchor]
    trunk_tip = np.nanmean(full_xy[:, others, :], axis=1)
    trunk_vec = trunk_tip - base
    trunk_angle = _angle_of(trunk_vec)
    u_trunk = _unit(trunk_vec)
    n_trunk = np.stack([-u_trunk[..., 1], u_trunk[..., 0]], axis=-1)
    if len(tail_idx) > 0:
        tail_tip     = full_xy[:, tail_idx[-1], :]
        tail_lateral = np.sum((tail_tip - base) * n_trunk, axis=-1)
    return trunk_angle, tail_lateral


def dominant_freq(win, fs):
    win = np.asarray(win, float)
    win = win[np.isfinite(win)]
    if win.size < 4 or np.nanstd(win) == 0:
        return np.nan
    x  = detrend(win, type='linear') if win.size >= 8 else (win - np.nanmean(win))
    xw = x * np.hanning(x.size)
    n  = xw.size
    nfft = 1
    while nfft < 8 * n: nfft <<= 1
    X = np.fft.rfft(xw, n=nfft)
    P = np.abs(X) ** 2
    f = np.fft.rfftfreq(nfft, d=1.0 / fs)
    P[0] = 0.0
    k = int(np.argmax(P))
    if k <= 0 or k >= P.size - 1:
        return float(f[k])
    a, b, c = P[k-1], P[k], P[k+1]
    denom   = a - 2*b + c
    if denom == 0 or not np.isfinite(denom):
        return float(f[k])
    return float((k + 0.5*(a-c)/denom) * fs / nfft)


def amp_pp_over2(win):
    win = np.asarray(win, float)[np.isfinite(np.asarray(win, float))]
    return 0.5 * (np.nanmax(win) - np.nanmin(win)) if win.size else np.nan


# =============================================================================
# VECTORISED WINDOW FEATURES
# =============================================================================
def _align_xy(comx, comy):
    kcols  = [k for k in _KCOLS if k in comx.columns and k in comy.columns]
    tx, ty = time_cols_from(comx), time_cols_from(comy)
    common = tx if tx == ty else sorted(set(tx) & set(ty), key=lambda s: int(s.split("_")[-1]))
    merged = comx[kcols + common].merge(comy[kcols + common], on=kcols, suffixes=("_x", "_y"))
    X = merged[[f"{c}_x" for c in common]].to_numpy(float)
    Y = merged[[f"{c}_y" for c in common]].to_numpy(float)
    return merged[kcols].reset_index(drop=True), common, X, Y


def _speed_um_s(X, Y, dt, pix2um):
    dx, dy = np.diff(X, axis=1), np.diff(Y, axis=1)
    v = (np.sqrt(dx*dx + dy*dy) / dt) * pix2um
    v = np.concatenate([np.full((v.shape[0], 1), np.nan), v], axis=1)
    v[v > VEL_CLIP_MAX] = np.nan
    return v


def _mad_1d(a, axis=1):
    med = np.nanmedian(a, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(a - med), axis=axis, keepdims=True)
    return med.squeeze(axis), mad.squeeze(axis)


def _windows(arr2d, W, S):
    T = arr2d.shape[1]
    if T < W: return None, None
    view   = sliding_window_view(arr2d, window_shape=W, axis=1)
    starts = np.arange(0, T - W + 1, S)
    return view[:, starts, :], starts


def _valid_windows(Xw, Yw, min_frac):
    return (np.isfinite(Xw) & np.isfinite(Yw)).sum(axis=2) >= (min_frac * Xw.shape[2])


def _tortuosity_windows(Xw, Yw, min_disp_px=5.0):
    dx, dy = np.diff(Xw, axis=2), np.diff(Yw, axis=2)
    path = np.nansum(np.hypot(dx, dy), axis=2)
    disp = np.hypot(Xw[..., -1] - Xw[..., 0], Yw[..., -1] - Yw[..., 0])
    bad  = (~np.isfinite(path)) | (~np.isfinite(disp)) | (disp < min_disp_px)
    with np.errstate(invalid='ignore', divide='ignore'):
        tau = path / disp
    tau[bad] = np.nan
    return tau


def _path_complexity_windows(Xw, Yw):
    """Correlation-entropy path complexity."""
    mu_x = np.nanmean(Xw, axis=2, keepdims=True)
    mu_y = np.nanmean(Yw, axis=2, keepdims=True)
    sx   = np.nanstd(Xw, axis=2, ddof=0, keepdims=True)
    sy   = np.nanstd(Yw, axis=2, ddof=0, keepdims=True)
    zx   = np.where(sx > 0, (Xw - mu_x) / sx, np.nan)
    zy   = np.where(sy > 0, (Yw - mu_y) / sy, np.nan)
    r    = np.nanmean(zx * zy, axis=2)
    p1   = np.clip((1.0 + r) * 0.5, 0, 1)
    p2   = np.clip((1.0 - r) * 0.5, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -(p1 * np.log2(p1) + p2 * np.log2(p2))
    H[~np.isfinite(H)] = np.nan
    return H


def _msd_slope_windows(Xw, Yw, dt, max_lag_frames=3):
    Lk       = max(1, min(max_lag_frames, Xw.shape[2] - 1))
    msd_list = []
    for k in range(1, Lk + 1):
        msd_k = np.nanmean((Xw[..., k:] - Xw[..., :-k])**2 +
                           (Yw[..., k:] - Yw[..., :-k])**2, axis=2)
        msd_list.append(msd_k)
    MSD = np.stack(msd_list, axis=2)
    t   = np.arange(1, Lk+1, dtype=float) * dt
    tc  = t - t.mean()
    M   = np.isfinite(MSD)
    num = np.nansum(np.where(M, MSD * tc, 0), axis=2)
    den = np.nansum(np.where(M, tc**2, 0), axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        slope = num / den
    slope[den <= 0] = np.nan
    return slope


def build_window_features(com_x_wide, com_y_wide,
                          pixel_to_um=PIXEL_TO_UM, dt=DT_RS,
                          win_frames=WIN_RS_FRAMES, stride_frames=STRIDE_RS_FR,
                          deadband_k_mad=DEADBAND_K_MAD, min_valid_frac=0.8):
    """
    Long-format window features: one row per (animal × window).
    Metrics: vel_p95, vel_max, vel_frac_active, acc_p95,
             tortuosity, path_complexity, msd_slope.
    """
    keys_df, tcols, X, Y = _align_xy(com_x_wide, com_y_wide)
    N, T = X.shape
    if T < win_frames:
        raise RuntimeError(f"Not enough frames ({T}) for window {win_frames}.")

    V    = _speed_um_s(X, Y, dt, pixel_to_um)
    dV   = np.diff(V, axis=1, prepend=np.nan) / dt
    abs_a = np.abs(dV)
    _, mad = _mad_1d(V, axis=1)
    thr    = (deadband_k_mad * mad).reshape(-1, 1)

    Xw, starts = _windows(X, win_frames, stride_frames)
    Yw, _      = _windows(Y, win_frames, stride_frames)
    Vw, _      = _windows(V, win_frames, stride_frames)
    Aw, _      = _windows(abs_a, win_frames, stride_frames)

    if Xw is None:
        return pd.DataFrame()

    ok  = _valid_windows(Xw, Yw, min_valid_frac)
    Wn  = ok.shape[1]

    def _mask(m): return np.where(ok[..., None], m, np.nan)

    vel_p95 = np.nanpercentile(_mask(Vw), 95, axis=2)
    vel_max = np.nanmax(_mask(Vw), axis=2)

    Vw_valid = np.isfinite(Vw)
    active   = (Vw > thr[:, None, :]) & Vw_valid
    num      = np.sum(np.where(ok[..., None], active, False), axis=2).astype(float)
    den      = np.sum(np.where(ok[..., None], Vw_valid, False), axis=2).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        vel_frac_active = num / den
    vel_frac_active[den == 0] = np.nan

    acc_p95   = np.nanpercentile(_mask(Aw), 95, axis=2)
    tau       = _tortuosity_windows(Xw, Yw)
    H_corr    = _path_complexity_windows(Xw, Yw)
    msd_slope = _msd_slope_windows(Xw, Yw, dt, max_lag_frames=3)

    out = keys_df.loc[np.repeat(np.arange(N), Wn)].reset_index(drop=True)
    out["WindowStart"]       = np.tile(starts, N)
    out["WindowStart_s"]     = out["WindowStart"] * dt
    out["vel_p95"]           = vel_p95.ravel("C")
    out["vel_max"]           = vel_max.ravel("C")
    out["vel_frac_active"]   = vel_frac_active.ravel("C")
    out["acc_p95"]           = acc_p95.ravel("C")
    out["tortuosity"]        = tau.ravel("C")
    out["path_complexity"]   = H_corr.ravel("C")
    out["msd_slope"]         = msd_slope.ravel("C")
    return out


def window_features_to_wide(win_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """Pivot long-format window features to wide format (Time_XXX columns)."""
    if win_df is None or win_df.empty or metric_col not in win_df.columns:
        return pd.DataFrame()
    k   = [c for c in _KCOLS if c in win_df.columns]
    sub = win_df[k + ["WindowStart", metric_col]].copy()
    sub[metric_col] = pd.to_numeric(sub[metric_col], errors="coerce")
    sub["_tcol"]    = sub["WindowStart"].apply(lambda x: f"Time_{int(x):03d}")
    try:
        wide = sub.pivot_table(index=k, columns="_tcol",
                               values=metric_col, aggfunc="first")
        wide = wide.reset_index()
        wide.columns.name = None
        non_t  = [c for c in wide.columns if not c.startswith("Time_")]
        t_cols = sorted([c for c in wide.columns if c.startswith("Time_")],
                        key=lambda c: int(c.split("_")[1]))
        return wide[non_t + t_cols]
    except Exception as e:
        print(f"    [window_to_wide] pivot failed for {metric_col}: {e}")
        return pd.DataFrame()


# =============================================================================
# SVD PATH COMPLEXITY — WINDOWED (Takens delay embedding)
# =============================================================================
def _obtain_embedding_matrix(X, Y, window):
    """Takens delay-embedding matrix from COM XY trajectory."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    min_len   = min(len(X), len(Y))
    X, Y      = X[:min_len], Y[:min_len]
    valid     = np.isfinite(X) & np.isfinite(Y)
    X, Y      = X[valid], Y[valid]
    if len(X) < window or np.std(X) == 0 or np.std(Y) == 0:
        return None
    X = (X - np.mean(X)) / np.std(X)
    Y = (Y - np.mean(Y)) / np.std(Y)
    n_rows = len(X) - window
    if n_rows < 1:
        return None
    Mx = np.stack([X[i:i + window] for i in range(n_rows)], axis=0)
    My = np.stack([Y[i:i + window] for i in range(n_rows)], axis=0)
    for col in range(Mx.shape[1]):
        if not np.isnan(Mx[:, col]).all():
            Mx[:, col] -= np.nanmean(Mx[:, col])
        if not np.isnan(My[:, col]).all():
            My[:, col] -= np.nanmean(My[:, col])
    return np.dstack([Mx, My])


def _svd_entropy(M):
    """Shannon entropy of normalised singular values."""
    if M is None or np.isnan(M).any():
        return None, None
    try:
        _, S, _ = np.linalg.svd(M, full_matrices=False)
    except np.linalg.LinAlgError:
        return None, None
    local_H = []
    for s in S:
        total = np.sum(s)
        if total <= 0:
            continue
        s_hat = s / total
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.sum(np.where(s_hat > 0, s_hat * np.log2(s_hat), 0.0))
        local_H.append(float(h))
    if not local_H:
        return None, None
    return local_H, float(np.sum(local_H))


def compute_svd_complexity_windowed(com_x_wide: pd.DataFrame,
                                    com_y_wide: pd.DataFrame,
                                    win_frames: int    = WIN_RS_FRAMES,
                                    stride_frames: int = STRIDE_RS_FR,
                                    embed_window: int  = COMPLEXITY_WIN_EMBED,
                                    min_valid_frac: float = COMPLEXITY_MIN_VALID_FRAC
                                    ) -> pd.DataFrame:
    """
    Windowed SVD path complexity (Takens delay embedding).
    Returns wide-format DataFrame with Time_XXX columns (one per window start frame).
    """
    if com_x_wide is None or com_x_wide.empty:
        return pd.DataFrame()
    if embed_window >= win_frames:
        raise ValueError(
            f"embed_window ({embed_window}) must be < win_frames ({win_frames}).")

    k     = [c for c in _KCOLS if c in com_x_wide.columns]
    tcols = time_cols_from(com_x_wide)
    T     = len(tcols)
    if T < win_frames:
        print(f"    [svd_complexity] Not enough frames ({T}) for window ({win_frames}). Skipping.")
        return pd.DataFrame()

    y_lookup: Dict[tuple, np.ndarray] = {}
    for _, yrow in com_y_wide.iterrows():
        y_lookup[tuple(yrow[c] for c in k)] = yrow[tcols].to_numpy(float)

    starts         = np.arange(0, T - win_frames + 1, stride_frames)
    time_col_names = [f"Time_{int(s):03d}" for s in starts]
    min_valid_n    = max(embed_window + 1, int(np.ceil(min_valid_frac * win_frames)))

    rows: List[Dict] = []
    for _, xrow in com_x_wide.iterrows():
        key    = tuple(xrow[c] for c in k)
        X_full = xrow[tcols].to_numpy(float)
        Y_full = y_lookup.get(key)
        if Y_full is None:
            continue
        H_vals: List[float] = []
        for s in starts:
            Xw     = X_full[s: s + win_frames]
            Yw     = Y_full[s: s + win_frames]
            n_valid = int((np.isfinite(Xw) & np.isfinite(Yw)).sum())
            if n_valid < min_valid_n:
                H_vals.append(np.nan)
                continue
            M = _obtain_embedding_matrix(Xw, Yw, embed_window)
            if M is None:
                H_vals.append(np.nan)
                continue
            _, global_H = _svd_entropy(M)
            H_vals.append(global_H if global_H is not None else np.nan)
        meta = {c: xrow[c] for c in k}
        meta.update(dict(zip(time_col_names, H_vals)))
        rows.append(meta)

    result = pd.DataFrame(rows)
    if not result.empty:
        n_fin = sum(1 for r in rows for kk, v in r.items()
                    if kk.startswith("Time_") and v is not None and np.isfinite(float(v)))
        print(f"    [svd_complexity] {len(result)} animals × {len(starts)} windows  "
              f"(embed={embed_window} fr={embed_window/FS_RS:.1f} s, "
              f"analysis={win_frames} fr={win_frames/FS_RS:.1f} s, "
              f"finite windows: {n_fin})")
    else:
        print("    [svd_complexity] WARNING: no valid animals produced.")
    return result


# =============================================================================
# OMEGA (body rotation)
# =============================================================================
def _unwrap_diff_per_run(theta: np.ndarray, dt: float) -> np.ndarray:
    theta  = np.asarray(theta, float)
    out    = np.full(theta.size, np.nan)
    m      = np.isfinite(theta).astype(int)
    padded = np.concatenate([[0], m, [0]])
    d      = np.diff(padded)
    for s, e in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        seg = np.unwrap(theta[s:e])
        if seg.size >= 2:
            o = np.full_like(seg, np.nan)
            o[1:] = np.diff(seg) / dt
            out[s:e] = o
    return out


def _axis_angle_from_full(full_rs: np.ndarray) -> np.ndarray:
    if full_rs.ndim != 3 or full_rs.shape[1] < 2:
        return np.full(full_rs.shape[0], np.nan)
    x, y  = full_rs[..., 0], full_rs[..., 1]
    n     = x.shape[1]
    head_x = np.nanmean(x[:, :min(10, n)], axis=1)
    head_y = np.nanmean(y[:, :min(10, n)], axis=1)
    tail_x = np.nanmean(x[:, -min(10, n):], axis=1)
    tail_y = np.nanmean(y[:, -min(10, n):], axis=1)
    return np.arctan2(tail_y - head_y, tail_x - head_x)


def build_omega_from_two_trunk_points(skeleton_store, dt=DT_RS) -> pd.DataFrame:
    rows = []
    for (exp_id, cond, indiv), meta in skeleton_store.items():
        full  = meta.get("full_resampled")
        trunk = meta.get("trunk_idx_rs", [])
        if full is None or full.size == 0:
            continue
        T     = full.shape[0]
        theta = np.full(T, np.nan)
        if isinstance(trunk, (list, tuple)) and len(trunk) >= 2:
            i1, i2 = int(trunk[0]), int(trunk[1])
            if i1 < full.shape[1] and i2 < full.shape[1] and i1 != i2:
                dx    = full[:, i2, 0] - full[:, i1, 0]
                dy    = full[:, i2, 1] - full[:, i1, 1]
                valid = np.isfinite(dx) & np.isfinite(dy) & (dx*dx + dy*dy > 1e-9)
                theta[valid] = np.arctan2(dy[valid], dx[valid])
        if np.isfinite(theta).sum() < max(1, int(0.2*T)):
            theta = _axis_angle_from_full(full)
        omega = np.abs(_unwrap_diff_per_run(theta, dt) * 180.0 / np.pi)
        tcols = build_time_cols(T)
        rows.append(pd.DataFrame([{"Experiment_ID": exp_id, "Condition": cond,
                                   "Individual": indiv} | dict(zip(tcols, omega))]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, axis=0, ignore_index=True)


# =============================================================================
# RDQ STATE CLASSIFICATION
# =============================================================================
def compute_roaming_threshold(velocity_wide: pd.DataFrame,
                               ref_cond: Optional[str] = RDQ_THRESHOLD_COND) -> float:
    """
    Derive the roaming threshold from the mean speed of reference-condition animals.
    If ref_cond is None, all animals are used.
    """
    if velocity_wide is None or velocity_wide.empty:
        return np.nan
    if ref_cond is not None and "Condition" in velocity_wide.columns:
        mask = velocity_wide["Condition"].str.lower() == ref_cond.lower()
        if not mask.any():
            print(f"[RDQ] WARNING: condition '{ref_cond}' not found; using all animals.")
            mask = pd.Series([True] * len(velocity_wide))
    else:
        mask = pd.Series([True] * len(velocity_wide))

    tcols = time_cols_from(velocity_wide)
    vals  = velocity_wide.loc[mask, tcols].to_numpy(float).ravel()
    vals  = vals[np.isfinite(vals) & (vals >= 0)]
    if vals.size == 0:
        return np.nan
    thr = float(vals.mean())
    print(f"[RDQ] Roaming threshold (mean speed of "
          f"{'all' if ref_cond is None else ref_cond} animals): {thr:.2f} µm/s")
    return thr


def classify_rdq_frames(speed_vec: np.ndarray, roaming_thr: float) -> np.ndarray:
    """
    Classify each frame:  0 = quiescent, 1 = dwelling, 2 = roaming.
    NaN speed → NaN state.
    """
    speed = np.asarray(speed_vec, float)
    valid = np.isfinite(speed)
    state = np.full_like(speed, np.nan, float)

    active   = valid & (speed >= QUIESCENCE_THRESHOLD_UM_S)
    roaming  = active & np.isfinite([roaming_thr]) & (speed >= roaming_thr)
    dwelling = active & ~roaming

    state[valid & ~active] = 0
    state[dwelling]         = 1
    state[roaming]          = 2

    # Single-frame noise smoothing
    for i in range(1, len(state) - 1):
        if (np.isfinite(state[i-1]) and np.isfinite(state[i+1])
                and state[i] != state[i-1] and state[i-1] == state[i+1]):
            state[i] = state[i-1]
    return state


def detect_bouts(state_vec: np.ndarray) -> List[Dict]:
    s = np.asarray(state_vec, float)
    n = s.size
    if n == 0 or not np.isfinite(s).any():
        return []
    s_f = s.copy()
    v   = np.isfinite(s_f)
    for i in range(n):
        if not v[i] and i > 0:
            for j in range(i + 1, min(i + BOUT_MAX_GAP + 1, n)):
                if v[j]:
                    s_f[i:j] = s_f[i - 1]
                    v[i:j]   = True
                    break
    s_int = np.full(n, -1, int)
    s_int[v] = s_f[v].astype(int)
    raw_bouts = []
    start = 0
    current = s_int[0]
    for i in range(1, n):
        if s_int[i] != current:
            if current >= 0 and (i - start) >= BOUT_MIN_FRAMES:
                raw_bouts.append((start, i, current))
            start = i
            current = s_int[i]
    if current >= 0 and (n - start) >= BOUT_MIN_FRAMES:
        raw_bouts.append((start, n, current))
    names = {0: "quiescent", 1: "dwelling", 2: "roaming"}
    return [{"start": st, "end": en,
             "duration_frames":  en - st,
             "duration_seconds": (en - st) * DT_RS,
             "state": int(state),
             "state_name": names.get(int(state), f"state_{state}")}
            for st, en, state in raw_bouts]


def summarise_animal_states(state_vec: np.ndarray) -> Optional[Dict]:
    s       = np.asarray(state_vec, float)
    v       = np.isfinite(s)
    n_valid = int(v.sum())
    if n_valid == 0:
        return None
    bouts = detect_bouts(s)

    def _bout_stats(state_id):
        bs = [b for b in bouts if b["state"] == state_id]
        if not bs:
            return {"n_bouts": 0, "mean_dur_s": np.nan,
                    "median_dur_s": np.nan, "max_dur_s": np.nan, "total_dur_s": 0.0}
        durs = np.array([b["duration_seconds"] for b in bs], float)
        return {"n_bouts": len(bs),
                "mean_dur_s":   float(np.nanmean(durs)),
                "median_dur_s": float(np.nanmedian(durs)),
                "max_dur_s":    float(np.nanmax(durs)),
                "total_dur_s":  float(np.nansum(durs))}

    sv  = s[v]
    n_q = int((sv == 0).sum())
    n_d = int((sv == 1).sum())
    n_r = int((sv == 2).sum())
    total_s = n_valid * DT_RS
    res = {"total_valid_frames": n_valid,
           "total_time_s":       total_s,
           "pct_quiescent":      100.0 * n_q / n_valid,
           "pct_dwelling":       100.0 * n_d / n_valid,
           "pct_roaming":        100.0 * n_r / n_valid,
           "pct_active":         100.0 * (n_d + n_r) / n_valid}
    for label, sid in [("quiescent", 0), ("dwelling", 1), ("roaming", 2)]:
        for k_, v_ in _bout_stats(sid).items():
            res[f"{label}_{k_}"] = v_
    distinct = [b["state"] for b in bouts]
    n_trans  = sum(1 for i in range(1, len(distinct)) if distinct[i] != distinct[i-1])
    res["n_transitions"]           = n_trans
    res["transition_rate_per_min"] = n_trans / (total_s / 60.0) if total_s > 0 else np.nan
    return res


def build_state_tables(velocity_wide: pd.DataFrame,
                       roaming_thr: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      state_ts_wide  – wide-format RDQ state time-series (0/1/2) per frame
      bout_summary   – one row per animal with scalar bout statistics
    """
    if velocity_wide is None or velocity_wide.empty:
        return pd.DataFrame(), pd.DataFrame()
    k     = [c for c in _KCOLS if c in velocity_wide.columns]
    tcols = time_cols_from(velocity_wide)
    state_rows = []
    bout_rows  = []
    for _, row in velocity_wide.iterrows():
        speed    = row[tcols].to_numpy(float)
        state_ts = classify_rdq_frames(speed, roaming_thr)
        meta     = {c: row[c] for c in k}
        state_rows.append(meta | dict(zip(tcols, state_ts)))
        stats = summarise_animal_states(state_ts)
        if stats:
            bout_rows.append(meta | stats)
    return pd.DataFrame(state_rows), pd.DataFrame(bout_rows)


# =============================================================================
# STAT HELPERS — each treatment vs. control (Holm-corrected MWU)
# =============================================================================
def _holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj   = np.empty(m)
    prev  = 0.0
    for rank, i in enumerate(order):
        val    = min(1.0, pvals[i] * (m - rank))
        adj[i] = max(prev, val)
        prev   = adj[i]
    return adj.tolist()


def mwu_vs_control(d: pd.DataFrame,
                   group_col: str,
                   groups_order: List[str],
                   control_name: str = CONTROL_NAME) -> List[Dict]:
    dd = d[[group_col, "Value"]].dropna()
    dd = dd[np.isfinite(dd["Value"])]
    present  = [g for g in groups_order if g in set(dd[group_col].dropna())]
    ctrl_key = next((g for g in present if g.lower() == control_name.lower()), None)
    if ctrl_key is None:
        return []
    ctrl_arr   = dd.loc[dd[group_col] == ctrl_key, "Value"].to_numpy()
    treatments = [g for g in present if g.lower() != control_name.lower()]
    if not treatments or ctrl_arr.size == 0:
        return []
    raw_ps, stat_info = [], []
    for trt in treatments:
        trt_arr = dd.loc[dd[group_col] == trt, "Value"].to_numpy()
        if trt_arr.size == 0:
            continue
        U, p = mannwhitneyu(ctrl_arr, trt_arr, alternative="two-sided")
        raw_ps.append(float(p))
        stat_info.append((ctrl_key, trt, U, ctrl_arr.size, trt_arr.size))
    if not raw_ps:
        return []
    adj_ps = _holm_adjust(raw_ps)
    return [{"Group": group_col, "A": ctrl_k, "B": trt, "Test": "Mann-Whitney",
             "U": U, "N1": n1, "N2": n2, "p_raw": p_raw, "p_holm": p_adj}
            for (ctrl_k, trt, U, n1, n2), p_raw, p_adj
            in zip(stat_info, raw_ps, adj_ps)]


# =============================================================================
# AGGREGATION HELPERS
# =============================================================================
def _per_animal_means_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=_KCOLS + ["Value"])
    tcols = time_cols_from(df)
    if not tcols:
        return pd.DataFrame(columns=_KCOLS + ["Value"])
    k   = [c for c in _KCOLS if c in df.columns]
    out = df[k].copy()
    out["Value"] = np.nanmean(df[tcols].to_numpy(float), axis=1)
    return out[np.isfinite(out["Value"])].reset_index(drop=True)


def _summarize_grouped(agg_df: pd.DataFrame, group_col: str,
                       metric_name: str, source: str) -> pd.DataFrame:
    cols = ["Metric","Source","Group","Level","N","Mean","Std","SEM","Min","P25","Median","P75","Max"]
    if agg_df is None or agg_df.empty or group_col not in agg_df.columns:
        return pd.DataFrame(columns=cols)
    d = agg_df[[group_col, "Value"]].dropna()
    d = d[np.isfinite(d["Value"])]
    if d.empty:
        return pd.DataFrame(columns=cols)
    g    = d.groupby(group_col)
    stat = g["Value"].agg(["count","mean","std","min","median","max"])
    q    = g["Value"].quantile([0.25, 0.75]).unstack(-1)
    stat["P25"] = q.get(0.25, np.nan)
    stat["P75"] = q.get(0.75, np.nan)
    stat["SEM"] = stat["std"] / np.sqrt(stat["count"])
    stat = stat.reset_index().rename(columns={group_col:"Level","count":"N","mean":"Mean",
                                              "std":"Std","min":"Min","median":"Median","max":"Max"})
    stat.insert(0,"Group",group_col)
    stat.insert(0,"Source",source)
    stat.insert(0,"Metric",metric_name)
    return stat[cols]


# =============================================================================
# SKELETON GEOMETRY (centerline building + normalisation)
# =============================================================================
def _strictly_increasing_s(ctrl, tol=1e-6):
    if ctrl.shape[0] <= 1:
        return np.array([0.0, 1.0]), np.vstack([ctrl[0], ctrl[0]])
    seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)
    s   = np.concatenate([[0.0], np.cumsum(seg)])
    L   = s[-1]
    if L <= 0:
        return np.array([0.0, 1.0]), np.vstack([ctrl[0], ctrl[0]])
    s /= L
    for i in range(1, s.size):
        if s[i] <= s[i-1] + tol:
            s[i] = s[i-1] + tol
    s = (s - s[0]) / (s[-1] - s[0])
    return s, ctrl


def _interp_centerline(ctrl, K):
    s, C  = _strictly_increasing_s(ctrl)
    s_new = np.linspace(0.0, 1.0, K)
    if _HAVE_PCHIP:
        x = PchipInterpolator(s, C[:, 0], extrapolate=False)(s_new)
        y = PchipInterpolator(s, C[:, 1], extrapolate=False)(s_new)
        x = np.where(np.isfinite(x), x, np.interp(s_new, s, C[:, 0]))
        y = np.where(np.isfinite(y), y, np.interp(s_new, s, C[:, 1]))
    else:
        x = np.interp(s_new, s, C[:, 0])
        y = np.interp(s_new, s, C[:, 1])
    return np.column_stack([x, y])


def _normalize_centerline(cl):
    if cl is None or cl.shape[0] < 2:
        return None
    P = cl.copy() - cl[0]
    L = float(np.hypot(P[-1, 0], P[-1, 1]))
    if L <= 0:
        return None
    ang = np.arctan2(P[-1, 1], P[-1, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    P    = P @ np.array([[c, -s], [s, c]]).T / L
    return P


def _find_bp_indices(bodyparts, keys):
    BP_PAT = {k: re.compile(rf"(?:^|[^0-9])(?:bodypart)?\s*{k[2:]}(?:[^0-9]|$)", re.I)
              for k in keys}
    out   = {k: None for k in keys}
    names = [str(x or "").lower() for x in bodyparts]
    for k, pat in BP_PAT.items():
        for j, nm in enumerate(names):
            if pat.search(nm):
                out[k] = j
                break
    return out


def _build_centerline_from_native_frames(full_nv, bodyparts, frame_start, frame_end, K=HE_K):
    T   = full_nv.shape[0]
    fs  = max(0, frame_start)
    fe  = min(T - 1, frame_end)
    if fe < fs:
        return None
    idxs = _find_bp_indices(bodyparts, ["bp5","bp6","bp7","bp8","bp9","bp10","bp11","bp12","bp13"])
    i5, i6, i7 = idxs["bp5"], idxs["bp6"], idxs["bp7"]
    if i5 is None or i6 is None:
        trunk, _ = split_trunk_tail(full_nv.shape[1])
        if trunk.size < 2:
            return None
        i5, i6 = int(trunk[0]), int(trunk[1])
        i7     = int(trunk[2]) if trunk.size > 2 else None
    window = full_nv[fs:fe + 1]
    T_win  = window.shape[0]
    CLn    = np.full((T_win, K, 2), np.nan)
    for t in range(T_win):
        P  = window[t]
        a0 = P[i5]; a1 = P[i6]
        if not (np.all(np.isfinite(a0)) and np.all(np.isfinite(a1))):
            continue
        v = a1 - a0
        L = float(np.hypot(v[0], v[1]))
        if L < HE_MIN_LEN_PX or L > HE_MAX_LEN_PX:
            continue
        u       = v / L
        ctrl_pts = [a0]
        if i7 is not None and i7 < P.shape[0] and np.all(np.isfinite(P[i7])):
            ctrl_pts.append(P[i7])
        else:
            ctrl_pts.append(a0 + 0.5 * L * u)
        ctrl_pts.append(a1)
        ctrl  = np.vstack(ctrl_pts)
        tvals = (ctrl - a0) @ u
        ctrl  = ctrl[np.argsort(tvals)]
        try:
            cl = _interp_centerline(ctrl, K)
        except Exception:
            continue
        if not np.all(np.isfinite(cl)):
            continue
        norm = _normalize_centerline(cl)
        if norm is not None:
            CLn[t] = norm
    return CLn


def _max_abs_y(frame): return float(np.nanmax(np.abs(frame[:, 1])))

def _max_adjacent_angle(frame):
    v     = np.diff(frame, axis=0)
    norms = np.linalg.norm(v, axis=1)
    good  = norms > 1e-9
    if np.sum(good) < 2:
        return 0.0
    vu = np.zeros_like(v)
    vu[good] = (v[good].T / norms[good]).T
    vi, vj = vu[:-1], vu[1:]
    dot = np.clip(np.sum(vi * vj, axis=1), -1, 1)
    ang = np.abs(np.arctan2(vi[:, 0]*vj[:, 1] - vi[:, 1]*vj[:, 0], dot))
    return float(np.nanmax(ang)) if ang.size else 0.0


# =============================================================================
# HOUSE-EXPANSION TAIL-BEAT METRICS
# =============================================================================
def _find_he_ranges(exp_id, he_dict):
    if exp_id in he_dict:
        return he_dict[exp_id]
    stripped = re.sub(r"_filtered$", "", exp_id, flags=re.I)
    if stripped in he_dict:
        return he_dict[stripped]
    for key in he_dict:
        if exp_id.startswith(key):
            return he_dict[key]
    for key in he_dict:
        if exp_id.lower().startswith(key.lower()):
            return he_dict[key]
    return None


def compute_he_tailbeat_metrics(skeleton_store, house_expansion_frames) -> pd.DataFrame:
    rows = []
    for (exp_id, cond, indiv), meta in skeleton_store.items():
        full_nv      = (meta.get("full_native_raw") if meta.get("full_native_raw") is not None
                        else meta.get("full_native"))
        trunk_idx_rs = meta.get("trunk_idx_rs", [])
        tail_idx_rs  = meta.get("tail_idx_rs", [])
        if full_nv is None or full_nv.size == 0 or not trunk_idx_rs:
            continue
        ranges = _find_he_ranges(exp_id, house_expansion_frames)
        if not ranges:
            continue
        if not tail_idx_rs:
            print(f"  [HE tailbeat] SKIP {exp_id[:40]} / {indiv}: no tail bodyparts")
            continue
        _, tail_lat_nv = tail_trunk_signals(full_nv, trunk_idx_rs, tail_idx_rs)
        for (fs, fe) in ranges:
            fs_c = max(0, fs)
            fe_c = min(full_nv.shape[0] - 1, fe)
            if fe_c <= fs_c:
                continue
            segment = tail_lat_nv[fs_c: fe_c + 1]
            seg_fin = segment[np.isfinite(segment)]
            if seg_fin.size < 4:
                print(f"  [HE tailbeat] SKIP {exp_id[:40]} / {indiv}: "
                      f"frames {fs}-{fe} → only {seg_fin.size} finite samples")
                continue
            rows.append({
                "Experiment_ID":    exp_id,
                "Condition":        cond,
                "Individual":       indiv,
                "HE_start":         fs,
                "HE_end":           fe,
                "tailbeat_freq_Hz": dominant_freq(segment, FPS_NATIVE),
                "tailbeat_amp_px":  amp_pp_over2(segment),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":

    # ── 1. Discover CSVs and assign conditions ─────────────────────────────
    OUTPUT_FOLDER_NAMES = {"data", "data1", "data2", "output", "outputs", "results"}

    def _is_output_path(fp):
        rel = os.path.relpath(fp, ROOT_DIR)
        return rel.split(os.sep)[0].lower() in OUTPUT_FOLDER_NAMES

    all_files = _collect_csvs(ROOT_DIR)
    print(f"Found {len(all_files)} CSV files under {ROOT_DIR}")
    all_files = [f for f in all_files if not _is_output_path(f)]
    print(f"After output-folder exclusion: {len(all_files)} CSV files")

    exp_to_condition: Dict[str, str] = {}
    file_info = []
    for fp in all_files:
        cond = _condition_from_path(fp, ROOT_DIR)
        if cond is None:
            print(f"[skip] no condition subfolder: {fp}")
            continue
        exp_id = os.path.splitext(os.path.basename(fp))[0]
        exp_to_condition[exp_id] = cond
        file_info.append((fp, cond))

    all_conditions   = sorted(set(c for _, c in file_info))
    condition_order, PALETTE = _build_condition_order_and_palette(all_conditions)
    print(f"Conditions: {condition_order}")

    # ── 2. Per-file processing ─────────────────────────────────────────────
    comx_list = []; comy_list = []
    curv_list = []; tan_list  = []; quir_list = []; axis_list = []
    skeleton_store: Dict[Tuple[str, str, str], Dict] = {}

    for fp, cond in file_info:
        try:
            exp_id = os.path.splitext(os.path.basename(fp))[0]
            print(f"  Processing: {exp_id}  |  condition={cond}")

            individuals, bodyparts, cx_nv, cy_nv, lk = read_filtered_csv(fp)

            cx_nv[lk < LIKELIHOOD_MIN] = np.nan
            cy_nv[lk < LIKELIHOOD_MIN] = np.nan
            cx_nv = iqr_outlier_removal(cx_nv, IQR_THRESHOLD)
            cy_nv = iqr_outlier_removal(cy_nv, IQR_THRESHOLD)
            cx_nv, cy_nv = filter_large_steps(cx_nv, cy_nv, DISTANCE_THRESHOLD)

            cxi_nv_raw = interpolate_bracketed(cx_nv.copy())
            cyi_nv_raw = interpolate_bracketed(cy_nv.copy())

            if MIN_CONTINUOUS_FRAMES > 1:
                print(f"    [continuity] min={MIN_CONTINUOUS_FRAMES} frames")
                cx_nv = mask_short_runs_native(cx_nv, MIN_CONTINUOUS_FRAMES)
                cy_nv = mask_short_runs_native(cy_nv, MIN_CONTINUOUS_FRAMES)
            cx_nv = interpolate_bracketed(cx_nv)
            cy_nv = interpolate_bracketed(cy_nv)

            cx_rs = resample_axis0(cx_nv, RESAMPLE_RATE)
            cy_rs = resample_axis0(cy_nv, RESAMPLE_RATE)

            time_cols    = build_time_cols(cx_rs.shape[0])
            indiv_labels = pd.Series(individuals).fillna("Unknown").astype(str)

            com_x_rows = []; com_y_rows = []
            curv_rows  = []; tan_rows  = []; quir_rows = []; axis_rows = []

            for indiv in sorted(indiv_labels.unique()):
                idx = np.where(indiv_labels.values == indiv)[0]
                if len(idx) == 0:
                    continue

                cxi_rs = cx_rs[:, idx]; cyi_rs = cy_rs[:, idx]
                cxi_nv = cx_nv[:, idx]; cyi_nv = cy_nv[:, idx]

                n_frames = cxi_rs.shape[0]
                valid_fr = (np.isfinite(cxi_rs).any(axis=1) &
                            np.isfinite(cyi_rs).any(axis=1)).sum()
                if n_frames == 0 or valid_fr / n_frames < MIN_VALID_FRAC_ANIMAL:
                    print(f"      [{indiv}] SKIP (ghost: {valid_fr}/{n_frames} valid frames)")
                    continue

                n_cols              = cxi_rs.shape[1]
                trunk_idx, tail_idx = split_trunk_tail(n_cols)
                print(f"      [{indiv}] n_cols={n_cols}  trunk={len(trunk_idx)}  tail={len(tail_idx)}")

                com_x = (row_nanmean(cxi_rs[:, trunk_idx]) if trunk_idx.size > 0
                         else np.full(cxi_rs.shape[0], np.nan))
                com_y = (row_nanmean(cyi_rs[:, trunk_idx]) if trunk_idx.size > 0
                         else np.full(cyi_rs.shape[0], np.nan))

                meta_base = {"Experiment_ID": exp_id, "Condition": cond, "Individual": indiv}
                com_x_rows.append(pd.DataFrame([meta_base | dict(zip(time_cols, com_x))]))
                com_y_rows.append(pd.DataFrame([meta_base | dict(zip(time_cols, com_y))]))

                skel_idx = skeleton_indices(n_cols)
                if skel_idx.size > 0:
                    x_df = pd.DataFrame(cxi_rs[:, skel_idx])
                    y_df = pd.DataFrame(cyi_rs[:, skel_idx])
                    skel = np.stack((x_df.values, y_df.values), axis=-1)
                    Np   = skel.shape[1]

                    if Np >= 3:
                        w  = max(3, SAVGOL_WL if Np >= SAVGOL_WL else (Np if Np%2==1 else Np-1))
                        kw = dict(window_length=w, polyorder=SAVGOL_ORDER, mode='nearest', axis=1)
                        dx  = savgol_filter(np.nan_to_num(skel[:,:,0]), deriv=1, **kw)
                        dy  = savgol_filter(np.nan_to_num(skel[:,:,1]), deriv=1, **kw)
                        ddx = savgol_filter(np.nan_to_num(skel[:,:,0]), deriv=2, **kw)
                        ddy = savgol_filter(np.nan_to_num(skel[:,:,1]), deriv=2, **kw)
                    else:
                        dx  = np.gradient(np.nan_to_num(skel[:,:,0]), axis=1)
                        dy  = np.gradient(np.nan_to_num(skel[:,:,1]), axis=1)
                        ddx = np.gradient(dx, axis=1); ddy = np.gradient(dy, axis=1)

                    eps      = 1e-6
                    num_c    = dx[:,:-2]*ddy[:,2:] - dy[:,:-2]*ddx[:,2:]
                    den_c    = (dx[:,:-2]**2 + dy[:,:-2]**2 + eps)**1.5
                    curv     = np.abs(num_c / den_c)
                    curv     = np.where(curv <= 0.2, curv, np.nan)
                    curv_avg = row_nanmean(curv) if curv.size else np.full(cxi_rs.shape[0], np.nan)

                    ang      = np.arctan2(np.diff(skel[:,:,1], axis=1), np.diff(skel[:,:,0], axis=1))
                    row_means = pd.DataFrame(ang).mean(axis=1, skipna=True).to_numpy()[:,None]
                    row_means = np.where(np.isfinite(row_means), row_means, 0.0)
                    ang_mc   = (ang - row_means)
                    ang_mc   = np.where(ang_mc < 0, ang_mc + 2*np.pi, ang_mc)
                    tan_avg  = row_nanmean(ang_mc) if ang_mc.size else np.full(cxi_rs.shape[0], np.nan)

                    hx = pd.DataFrame(x_df.iloc[:,:min(10, x_df.shape[1])]).mean(axis=1, skipna=True).to_numpy()
                    hy = pd.DataFrame(y_df.iloc[:,:min(10, y_df.shape[1])]).mean(axis=1, skipna=True).to_numpy()
                    tx = pd.DataFrame(x_df.iloc[:,-min(10, x_df.shape[1]):]).mean(axis=1, skipna=True).to_numpy()
                    ty = pd.DataFrame(y_df.iloc[:,-min(10, y_df.shape[1]):]).mean(axis=1, skipna=True).to_numpy()
                    axis_ang = np.arctan2(ty-hy, tx-hx)
                    dev      = ang - axis_ang[:,None]
                    dev      = (dev + np.pi) % (2*np.pi) - np.pi
                    quir     = np.nansum(np.abs(dev), axis=1)
                    axis_rows.append(pd.DataFrame([meta_base | dict(zip(time_cols, axis_ang))]))
                else:
                    curv_avg = tan_avg = quir = np.full(cxi_rs.shape[0], np.nan)

                curv_rows.append(pd.DataFrame([meta_base | dict(zip(time_cols, curv_avg))]))
                tan_rows.append( pd.DataFrame([meta_base | dict(zip(time_cols, tan_avg))]))
                quir_rows.append(pd.DataFrame([meta_base | dict(zip(time_cols, quir))]))

                full_nv_raw_store = np.stack((cxi_nv_raw[:, idx], cyi_nv_raw[:, idx]), axis=-1)
                full_nv_store     = np.stack((cxi_nv, cyi_nv), axis=-1)
                full_rs_store     = np.stack((cxi_rs, cyi_rs), axis=-1)
                trunk_rs, tail_rs = split_trunk_tail(cxi_rs.shape[1])
                skeleton_store[(exp_id, cond, indiv)] = {
                    "full_native":     full_nv_store,
                    "full_native_raw": full_nv_raw_store,
                    "full_resampled":  full_rs_store,
                    "bodyparts":       [bodyparts[j] for j in idx],
                    "trunk_idx_rs":    trunk_rs.tolist(),
                    "tail_idx_rs":     tail_rs.tolist(),
                    "skel_idx_rs":     skeleton_indices(cxi_rs.shape[1]).tolist(),
                }

            def _cat(lst):
                return pd.concat(lst, axis=0, ignore_index=True) if lst else pd.DataFrame()

            comx_list.append(_cat(com_x_rows));  comy_list.append(_cat(com_y_rows))
            curv_list.append(_cat(curv_rows));   tan_list.append(_cat(tan_rows))
            quir_list.append(_cat(quir_rows));   axis_list.append(_cat(axis_rows))

        except Exception as e:
            import traceback
            print(f"[WARN] Skipped {fp}: {e}")
            traceback.print_exc()

    # ── 3. Concatenate ─────────────────────────────────────────────────────
    def _concat(lst):
        ne = [d for d in lst if not d.empty]
        return pd.concat(ne, axis=0, ignore_index=True) if ne else pd.DataFrame()

    com_x_wide          = _concat(comx_list)
    com_y_wide          = _concat(comy_list)
    avg_curvature_wide  = _concat(curv_list)
    avg_tan_angles_wide = _concat(tan_list)
    quirkiness_wide     = _concat(quir_list)
    axis_angle_wide     = _concat(axis_list)

    if not com_x_wide.empty:
        print("COM rows by condition:", com_x_wide.groupby("Condition", dropna=False).size().to_dict())

    # ── 4. Velocity (deadbanded) ───────────────────────────────────────────
    def _mad_scalar(a):
        a = a[np.isfinite(a)]
        return np.nan if a.size == 0 else np.median(np.abs(a - np.median(a)))

    if not com_x_wide.empty and not com_y_wide.empty:
        tcols  = time_cols_from(com_x_wide)
        merged = com_x_wide[_KCOLS + tcols].merge(com_y_wide[_KCOLS + tcols],
                                                   on=_KCOLS, suffixes=("_x","_y"))
        X  = merged[[f"{c}_x" for c in tcols]].to_numpy(float)
        Y  = merged[[f"{c}_y" for c in tcols]].to_numpy(float)
        vel = (np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1)) / DT_RS) * PIXEL_TO_UM
        vel = np.concatenate([np.full((vel.shape[0],1), np.nan), vel], axis=1)
        vel[vel > VEL_CLIP_MAX] = np.nan
        vel_db = vel.copy()
        for i in range(vel_db.shape[0]):
            row  = vel_db[i]
            m    = np.isfinite(row)
            if not m.any():
                continue
            base = DEADBAND_K_MAD * _mad_scalar(row[m])
            if not (np.isfinite(base) and base > 0):
                continue
            thr_in  = base * HYST_IN_SCALE
            thr_out = base * HYST_OUT_SCALE
            active  = False
            for t in range(row.size):
                v_ = row[t]
                if not np.isfinite(v_): continue
                if active:
                    if v_ < thr_out: active = False; row[t] = 0.0
                else:
                    if v_ >= thr_in: active = True
                    else:            row[t] = 0.0
            vel_db[i] = row
        velocity_wide          = merged[_KCOLS].copy()
        velocity_wide[tcols]   = vel_db

        try:
            window_features = build_window_features(com_x_wide, com_y_wide)
            print(f"Window features: {window_features.shape}")
        except Exception as e:
            print(f"[warn] window features: {e}")
            window_features = pd.DataFrame()
    else:
        print("[warn] Missing COM wides; skipping velocity/window features.")
        velocity_wide   = pd.DataFrame()
        window_features = pd.DataFrame()

    # ── 5. Window metrics → wide format ───────────────────────────────────
    WINDOW_METRIC_COLS = [
        "vel_p95", "vel_max", "vel_frac_active",
        "acc_p95", "tortuosity", "path_complexity", "msd_slope",
    ]
    print("\nConverting window metrics to wide format…")
    window_wide_tables: Dict[str, pd.DataFrame] = {}
    for col in WINDOW_METRIC_COLS:
        wdf = window_features_to_wide(window_features, col)
        if not wdf.empty:
            window_wide_tables[col] = wdf
            print(f"  {col}: {len(wdf)} animals × {len(time_cols_from(wdf))} windows")
        else:
            print(f"  {col}: empty")

    # ── 6. SVD path complexity ─────────────────────────────────────────────
    print(f"\nComputing windowed SVD complexity "
          f"(embed={COMPLEXITY_WIN_EMBED} fr={COMPLEXITY_WIN_EMBED/FS_RS:.1f} s, "
          f"window={WIN_RS_FRAMES} fr={WIN_RS_FRAMES/FS_RS:.1f} s)…")
    complexity_win_wide = compute_svd_complexity_windowed(com_x_wide, com_y_wide)

    # ── 7. Tail-beat ───────────────────────────────────────────────────────
    tb_freq_rows = []; tb_amp_rows = []
    for (exp_id, cond, indiv), meta in skeleton_store.items():
        full_nv      = meta["full_native"]
        trunk_idx_rs = meta["trunk_idx_rs"]
        tail_idx_rs  = meta["tail_idx_rs"]
        if full_nv.size == 0 or not trunk_idx_rs:
            continue
        _, tail_lat_nv = tail_trunk_signals(full_nv, trunk_idx_rs, tail_idx_rs)
        T_rs_local = meta["full_resampled"].shape[0]
        domf = np.full(T_rs_local, np.nan)
        amp  = np.full(T_rs_local, np.nan)
        for k in range(T_rs_local):
            end_nv   = k * RESAMPLE_RATE
            start_nv = end_nv - WIN_NV_FRAMES + 1
            if start_nv < 0 or end_nv >= tail_lat_nv.size:
                continue
            domf[k] = dominant_freq(tail_lat_nv[start_nv:end_nv+1], FPS_NATIVE)
            s1 = end_nv - int(round(FPS_NATIVE)) + 1
            if s1 >= 0:
                amp[k] = amp_pp_over2(tail_lat_nv[s1:end_nv+1])
        tcols_l   = build_time_cols(T_rs_local)
        mb        = {"Experiment_ID": exp_id, "Condition": cond, "Individual": indiv}
        tb_freq_rows.append(pd.DataFrame([mb | dict(zip(tcols_l, domf))]))
        tb_amp_rows.append( pd.DataFrame([mb | dict(zip(tcols_l, amp))]))

    tailbeat_freq_wide = _concat(tb_freq_rows)
    tailbeat_amp_wide  = _concat(tb_amp_rows)

    # ── 8. Omega ───────────────────────────────────────────────────────────
    omega_abs_wide = build_omega_from_two_trunk_points(skeleton_store, dt=DT_RS)

    # ── 9. RDQ states & bouts ──────────────────────────────────────────────
    roaming_thr = compute_roaming_threshold(velocity_wide)
    print(f"[RDQ] Quiescence border: {QUIESCENCE_THRESHOLD_UM_S:.1f} µm/s  "
          f"|  Roaming border: {roaming_thr:.2f} µm/s")
    state_ts_wide, bout_summary = build_state_tables(velocity_wide, roaming_thr)

    # ── 10. House-expansion tail-beat ──────────────────────────────────────
    he_tb_df = compute_he_tailbeat_metrics(skeleton_store, HOUSE_EXPANSION_FRAMES)
    if not he_tb_df.empty:
        print(f"  HE tail-beat records: {len(he_tb_df)} "
              f"({he_tb_df.groupby('Condition').size().to_dict()})")
    else:
        print("  [HE tailbeat] No records found.")

    # ── 11. Save all outputs ───────────────────────────────────────────────
    print(f"\nSaving outputs to {WIDE_DIR} …")

    native_saves = [
        ("com_x_wide",          com_x_wide,          WIDE_DIR),
        ("com_y_wide",          com_y_wide,           WIDE_DIR),
        ("velocity_wide",       velocity_wide,         WIDE_DIR),
        ("avg_curvature_wide",  avg_curvature_wide,    WIDE_DIR),
        ("avg_tan_angles_wide", avg_tan_angles_wide,   WIDE_DIR),
        ("quirkiness_wide",     quirkiness_wide,       WIDE_DIR),
        ("tailbeat_freq_wide",  tailbeat_freq_wide,    WIDE_DIR),
        ("tailbeat_amp_wide",   tailbeat_amp_wide,     WIDE_DIR),
        ("omega_abs_wide",      omega_abs_wide,        WIDE_DIR),
        ("state_ts_wide",       state_ts_wide,         WIDE_DIR),
        ("bout_summary",        bout_summary,          WIDE_DIR),
    ]
    for name, df, d in native_saves:
        if df is not None and not df.empty:
            path = os.path.join(d, f"{name}.csv")
            df.to_csv(path, index=False, float_format="%.4f")
            print(path)

    for col, wdf in window_wide_tables.items():
        if not wdf.empty:
            path = os.path.join(WIDE_DIR, f"window_{col}_wide.csv")
            wdf.to_csv(path, index=False, float_format="%.4f")
            print(path)

    if not complexity_win_wide.empty:
        path = os.path.join(WIDE_DIR, "svd_complexity_wide.csv")
        complexity_win_wide.to_csv(path, index=False, float_format="%.4f")
        print(path)

    if not he_tb_df.empty:
        path = os.path.join(WIDE_DIR, "he_tailbeat.csv")
        he_tb_df.to_csv(path, index=False, float_format="%.4f")
        print(path)

    pd.DataFrame([{
        "quiescence_threshold_um_s": QUIESCENCE_THRESHOLD_UM_S,
        "roaming_threshold_um_s":    roaming_thr,
        "roaming_thr_source":        f"mean speed of {RDQ_THRESHOLD_COND or 'all'} animals",
    }]).to_csv(os.path.join(STATE_DIR, "rdq_thresholds.csv"), index=False)
    print(os.path.join(STATE_DIR, "rdq_thresholds.csv"))

    # Per-animal scalar feature matrix
    print("\nBuilding per-animal scalar feature matrix…")
    from functools import reduce
    scalar_parts = []
    for key, df in [("velocity",        velocity_wide),
                    ("avg_curvature",   avg_curvature_wide),
                    ("avg_tan_angles",  avg_tan_angles_wide),
                    ("quirkiness",      quirkiness_wide),
                    ("tailbeat_freq",   tailbeat_freq_wide),
                    ("tailbeat_amp",    tailbeat_amp_wide),
                    ("omega_abs",       omega_abs_wide)]:
        agg = _per_animal_means_from_wide(df)
        if not agg.empty:
            scalar_parts.append(agg.rename(columns={"Value": key}))
    for col in WINDOW_METRIC_COLS:
        if col in window_wide_tables:
            agg = _per_animal_means_from_wide(window_wide_tables[col])
            if not agg.empty:
                scalar_parts.append(agg.rename(columns={"Value": col}))
    if not complexity_win_wide.empty:
        agg = _per_animal_means_from_wide(complexity_win_wide)
        if not agg.empty:
            scalar_parts.append(agg.rename(columns={"Value": "svd_complexity"}))
    if scalar_parts:
        feature_matrix = reduce(
            lambda a, b: pd.merge(a, b,
                on=[c for c in _KCOLS if c in a.columns and c in b.columns],
                how="outer"),
            scalar_parts)
        path = os.path.join(WIDE_DIR, "per_animal_feature_matrix.csv")
        feature_matrix.to_csv(path, index=False, float_format="%.4f")
        print(path)
        print(f"  {feature_matrix.shape[1] - len(_KCOLS)} features × {len(feature_matrix)} animals")

    print(f"\n{'='*65}")
    print("✓ All outputs saved.")
    print(f"  Wide tables (PCA/HMM ready) → {WIDE_DIR}/")
    print(f"  RDQ thresholds              → {STATE_DIR}/rdq_thresholds.csv")
    print(f"{'='*65}")