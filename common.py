#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared configuration and helpers for all steps (from common import *):
paths, skeleton cleaning, 1 s window features, frame posture, skeleton
normalisation, statistics and figure saving.
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lombscargle
from scipy.stats import mannwhitneyu

# =============================================================================
# PATHS — edit before running
# =============================================================================
HD  = "/path/to/House_data"          # DLC csv folders + annotation txt files
OUT = f"{HD}/analysis_v4"            # all outputs
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# TIMING / WINDOWS
# =============================================================================
FPS, WIN, STRIDE = 30.0, 30, 15      # 1 s windows, 0.5 s stride
DT = STRIDE / FPS
MIN_GOOD = 15                        # clean frames needed inside a window
STILL = 0.1                          # Lref/s: a window is 'still' below this median speed
STILL_MIN_S = 5.0                    # still bout = still windows lasting ≥ 5 s
MIN_WIN_PHASE = {"free": 20, "pre": 10, "HE": 10, "in_house": 20, "housed": 20, "after_escape": 20}
FREQS = np.linspace(1.0, 14.0, 53)   # rhythmicity spectrum (Hz)
FREQ_W = np.geomspace(0.75, 12.0, 8) # embedding spectra (Hz)

# cleaning
LK_MIN, GAP_FILL = 0.80, 3           # likelihood threshold; interpolate gaps ≤ 3 frames
TREE_K = 3.5                         # anatomical link may be ≤ 3.5 × its median length
JUMP_WIN, JUMP_K = 61, 1.5           # whole-skeleton jump: centroid > 1.5 trunk lengths from 2 s median

# =============================================================================
# SKELETON (DLC tail order is not sequential: base → tip = bp5,10,8,12,7,13,9,11,6)
# =============================================================================
TRUNK = ["bodypart1", "bodypart2", "bodypart3", "bodypart4"]     # bp1/bp2 trunk ends, bp3 junction
TAIL4 = ["bodypart5", "bodypart10", "bodypart8", "bodypart12"]   # proximal tail, base → mid
BP_ALL = TRUNK + ["bodypart5", "bodypart10", "bodypart8", "bodypart12", "bodypart7",
                  "bodypart13", "bodypart9", "bodypart11", "bodypart6"]
TREE = [("bodypart3", "bodypart1"), ("bodypart3", "bodypart2"), ("bodypart3", "bodypart4"),
        ("bodypart3", "bodypart5"), ("bodypart5", "bodypart10"), ("bodypart10", "bodypart8"),
        ("bodypart8", "bodypart12"), ("bodypart12", "bodypart7"), ("bodypart7", "bodypart13"),
        ("bodypart13", "bodypart9"), ("bodypart9", "bodypart11"), ("bodypart11", "bodypart6")]
CHAIN = ["bodypart3", "bodypart5", "bodypart10", "bodypart8", "bodypart12",
         "bodypart7", "bodypart13", "bodypart9", "bodypart11", "bodypart6"]
SEGS = [f"a{k}" for k in range(len(CHAIN) - 1)]   # a0 = junction (3→5); a1–a3 proximal; a4–a8 distal
PROX = SEGS[:4]
TRUNK_DRAW = ["bodypart1", "bodypart3", "bodypart2", "bodypart4", "bodypart1"]
TAIL_DRAW = CHAIN
PROX_DRAW = TAIL_DRAW[:5]

# =============================================================================
# GROUPS / COLOURS
# =============================================================================
# C/O = control/6-OHDA; + builder, - non-builder (B1), ? non-builder (B2), h housed at start, R rescue
GROUP_COL = {"C+": "#2166AC", "C-": "#67A9CF", "C?": "#A6CEE3", "Ch": "#053061",
             "O+": "#B2182B", "O-": "#EF8A62", "O?": "#FDBF9F", "Oh": "#67001F", "R": "#999999"}
COND_COL = {"control": GROUP_COL["C+"], "6-OHDA": GROUP_COL["O+"]}
BASE_PAL = ["#4D4D4D", "#9E9E9E", "#1B9E77", "#66A61E", "#E6AB02", "#7570B3", "#A6761D", "#E7298A"]   # regions
SUSPECT_HOUSED = {"20250902_101434", "20250818_093335", "20250818_101055", "20250819_150131",
                  "20250819_144232", "20250818_152240", "20250819_142046"}   # B2 controls, likely in a house at start
SETS = [   # (tag, phase per group, plotted groups, tested pairs)
    ("free_behaviour", {"C-": "free", "O-": "free", "C?": "free", "O?": "free"},
     ["C-", "O-", "C?", "O?"], [("C-", "O-"), ("C?", "O?")]),
    ("during_expansion", {"C+": "HE", "O+": "HE"}, ["C+", "O+"], [("C+", "O+")]),
    ("inside_house", {"C+": "in_house", "O+": "in_house", "Ch": "housed", "Oh": "housed"},
     ["C+", "O+", "Ch", "Oh"], [("C+", "O+"), ("Ch", "Oh")]),
]
REGION_NAMES_W = {}                  # whole-video map region names, e.g. {1: "rest, curled", ...}

def rname_w(k):
    return REGION_NAMES_W.get(int(k) + 1, f"region {int(k) + 1}")

FEATS = {   # step-02 window features: (label, kind, log-scale in plots)
    "speed_med":  ("Speed, median (Lref/s)", "kin", True),
    "speed_p95":  ("Speed, p95 (Lref/s)", "kin", True),
    "omega_mean": ("Body rotation (deg/s)", "kin", True),
    "straight":   ("Path straightness", "kin", False),
    "fwd_mean":   ("Forward drift (+1 = forward)", "kin", False),
    "junc_abs":   ("Trunk–tail junction angle (deg)", "post", False),
    "junc_sd":    ("Junction angle SD (deg)", "post", True),
    "base_abs":   ("Tail-base angle (deg)", "post", False),
    "base_asym":  ("Tail-base asymmetry (deg)", "post", False),
    "base_sd":    ("Tail-base angle SD (deg)", "post", True),
    "bend_abs":   ("Proximal bend (deg)", "post", False),
    "bend_sd":    ("Proximal bend SD (deg)", "post", True),
    "lat_rms":    ("Lateral excursion (Lref)", "post", False),
    "base_vel":   ("Tail-base angular velocity (deg/s)", "post", True),
    "ls_pow":     ("Tail rhythmicity (LS power)", "post", False),
    "still":      ("Fraction of time still", "kin", False),
}

MM = 1 / 25.4
FIG_RC = {"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": 7,
          "axes.linewidth": 0.6, "svg.fonttype": "none"}

# =============================================================================
# LOADING
# =============================================================================
def load_meta2():
    return pd.read_csv(f"{OUT}/02_features/02_metadata_with_qc.csv", dtype={"Experiment_ID": str})

def load_windows():
    """1 s windows with features (step 02) and decoded movement modes (step 03)."""
    w = pd.read_csv(f"{OUT}/03b_decoding/03b_windows_long_decoded.csv", dtype={"Experiment_ID": str}, low_memory=False)
    return w.sort_values(["Experiment_ID", "WindowStart"]).reset_index(drop=True)

def valid_animals(meta2):
    """Not excluded, rescue group removed."""
    return meta2[(meta2["exclude_reason"].fillna("") == "") & (meta2["group"] != "R")].copy()

def load_abnormal():
    """Annotated abnormal starts / failed insertions (Experiment_ID, event, start, end; native frames)."""
    ev = pd.read_csv(f"{OUT}/01_annotation_events.csv", dtype={"Experiment_ID": str})
    if ev.empty or "event" not in ev:
        return pd.DataFrame(columns=["Experiment_ID", "event", "start", "end"])
    return ev[ev["event"].astype(str).str.contains("abnormal|failed", case=False)]

# =============================================================================
# READING + CLEANING
# =============================================================================
def read_dlc(fp):
    hdr = pd.read_csv(fp, nrows=4, header=None, low_memory=False)
    bps = hdr.iloc[2, 1::3].astype(str).str.strip().tolist()
    raw = pd.read_csv(fp, header=None, skiprows=4, low_memory=False)
    num = lambda s: raw.iloc[:, s::3].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    return bps, num(1), num(2), num(3)

ADJ = {b: [] for b in BP_ALL}
for _p, _c in TREE:
    ADJ[_p].append(_c); ADJ[_c].append(_p)

def bfs_links(root):
    order, seen, q = [], {root}, [root]
    while q:
        p = q.pop(0)
        for c in ADJ[p]:
            if c not in seen:
                seen.add(c); order.append((p, c)); q.append(c)
    return order
TREE_ORDERS = {r: bfs_links(r) for r in BP_ALL}

def attach_tree(X, Y, L, bps):
    """Keep, per frame, the largest anatomically connected set of points (tried from every root)."""
    j = {b: bps.index(b) for b in BP_ALL}
    P = {b: np.where((L[:, j[b]] >= LK_MIN)[:, None], np.column_stack([X[:, j[b]], Y[:, j[b]]]), np.nan)
         for b in BP_ALL}
    med = {}
    for p, c in TREE:
        med[(p, c)] = med[(c, p)] = np.nanmedian(np.hypot(*(P[c] - P[p]).T))
    T = X.shape[0]
    ATT = np.zeros((len(BP_ALL), len(BP_ALL), T), dtype=bool)
    for ri, r in enumerate(BP_ALL):
        att, R, A = {r: np.isfinite(P[r][:, 0])}, {r: P[r].copy()}, {r: np.zeros(T)}
        for p, c in TREE_ORDERS[r]:
            acc = A[p] + med[(p, c)]
            ok = np.isfinite(P[c][:, 0]) & (np.hypot(*(P[c] - R[p]).T) <= TREE_K * acc) & att[r]
            att[c] = ok
            R[c] = np.where(ok[:, None], P[c], R[p])
            A[c] = np.where(ok, 0.0, acc)
        ATT[ri] = np.stack([att[b] for b in BP_ALL])
    best = np.argmax(ATT.sum(axis=1), axis=0)
    keep = ATT[best, :, np.arange(T)]
    return {b: np.where(keep[:, i:i + 1], P[b], np.nan) for i, b in enumerate(BP_ALL)}

def clean_frames(fp):
    """Likelihood → anatomical attachment → whole-skeleton jump removal."""
    bps, X, Y, L = read_dlc(fp)
    P = attach_tree(X, Y, L, bps)
    tr = np.stack([P[b] for b in TRUNK])
    n_tr = np.isfinite(tr[..., 0]).sum(0)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cen = np.where((n_tr >= 2)[:, None], np.nanmean(tr, axis=0), np.nan)
        tl = np.nanmedian(np.hypot(*(P["bodypart1"] - P["bodypart2"]).T))
    rm = pd.DataFrame(cen).rolling(JUMP_WIN, center=True, min_periods=15).median().to_numpy()
    jump = np.hypot(*(cen - rm).T) / tl > JUMP_K
    for b in BP_ALL:
        P[b][jump] = np.nan
    return P, dict(T=X.shape[0], trunk_len_px=tl, frac_jump=float(jump.mean()))

def fill_gaps(v, gap=GAP_FILL):
    s = pd.Series(v)
    f = s.interpolate(limit_area="inside")
    na = s.isna()
    run = na.groupby((na != na.shift()).cumsum()).transform("sum")
    f[na & (run > gap)] = np.nan
    return f.to_numpy()

def wrap(a):
    return np.angle(np.exp(1j * a))

# =============================================================================
# PHASES (from annotations)
# =============================================================================
def phase_vector(r, T):
    """free | pre | HE | in_house | after_escape | housed — per frame."""
    if r["group"] in ("Ch", "Oh"):
        return np.full(T, "housed", dtype=object)
    ph = np.full(T, "free", dtype=object)
    if pd.isna(r["he_start"]):
        return ph
    hs, he = int(r["he_start"]), int(r["he_end"])
    ph[:hs] = "pre"
    ph[hs:he + 1] = "HE"
    ins = int(r["in_start"]) if pd.notna(r["in_start"]) else he + 1
    ine = int(r["in_end"]) if pd.notna(r["in_end"]) else T - 1
    ph[ins:ine + 1] = "in_house"
    if pd.notna(r["escape"]):
        ph[int(r["escape"]):] = "after_escape"
    return ph

# =============================================================================
# 1 s WINDOW FEATURES (step 02)
# =============================================================================
def window_features(P, ph):
    Q = {b: np.column_stack([fill_gaps(P[b][:, 0]), fill_gaps(P[b][:, 1])]) for b in TRUNK + TAIL4}
    T = len(ph)
    seg = lambda a, b: np.hypot(*(Q[b] - Q[a]).T)
    Lref = np.nanmedian(seg("bodypart5", "bodypart10") + seg("bodypart10", "bodypart8") + seg("bodypart8", "bodypart12"))
    # body kinematics
    tr = np.stack([Q[b] for b in TRUNK])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        com = np.where((np.isfinite(tr[..., 0]).sum(0) >= 2)[:, None], np.nanmean(tr, axis=0), np.nan)
    com = pd.DataFrame(com).rolling(5, center=True, min_periods=3).mean().to_numpy()
    step = np.diff(com, axis=0)
    speed = np.r_[np.nan, np.hypot(*step.T)] * FPS / Lref
    axis = Q["bodypart2"] - Q["bodypart1"]
    th = np.arctan2(axis[:, 1], axis[:, 0])
    omega = np.r_[np.nan, np.abs(wrap(np.diff(th)))] * FPS * 180 / np.pi
    moving = np.r_[False, np.hypot(*step.T) * FPS / Lref > STILL]
    fwd = np.where(moving, -np.cos(wrap(np.r_[np.nan, np.arctan2(step[:, 1], step[:, 0])] - th)), np.nan)
    # proximal tail posture (angles relative to the trunk axis)
    ang = lambda a, b: np.arctan2(*(Q[b] - Q[a])[:, ::-1].T)
    base = wrap(ang("bodypart5", "bodypart10") - th) * 180 / np.pi
    junc = wrap(ang("bodypart3", "bodypart5") - th) * 180 / np.pi
    bend = wrap(ang("bodypart8", "bodypart12") - ang("bodypart5", "bodypart10")) * 180 / np.pi
    ub = Q["bodypart5"] - Q["bodypart3"]
    ub = ub / np.linalg.norm(ub, axis=1, keepdims=True)
    d12 = Q["bodypart12"] - Q["bodypart5"]
    lat = (ub[:, 0] * d12[:, 1] - ub[:, 1] * d12[:, 0]) / Lref
    base_vel = np.r_[np.nan, np.abs(wrap(np.radians(np.diff(base))))] * FPS * 180 / np.pi
    post_ok = np.isfinite(base) & np.isfinite(bend) & np.isfinite(lat) & np.isfinite(junc)

    starts = np.arange(0, T - WIN + 1, STRIDE)
    V = lambda a: sliding_window_view(a, WIN)[starts]
    Vs, Vo, Vf, Vx, Vy = V(speed), V(omega), V(fwd), V(com[:, 0]), V(com[:, 1])
    Vb, Vj, Vbd, Vl, Vv, Vp = V(base), V(junc), V(bend), V(lat), V(base_vel), V(post_ok)
    rows = []
    for i, s in enumerate(starts):
        d = dict(WindowStart=int(s), t_s=(s + WIN / 2) / FPS, phase=ph[s + WIN // 2], n_post=int(Vp[i].sum()))
        sp = Vs[i][np.isfinite(Vs[i])]
        if sp.size >= MIN_GOOD:
            om, fw = Vo[i][np.isfinite(Vo[i])], Vf[i][np.isfinite(Vf[i])]
            ok = np.isfinite(Vx[i]) & np.isfinite(Vy[i])
            path = np.sum(np.hypot(np.diff(Vx[i][ok]), np.diff(Vy[i][ok])))
            net = np.hypot(Vx[i][ok][-1] - Vx[i][ok][0], Vy[i][ok][-1] - Vy[i][ok][0])
            d.update(speed_med=np.median(sp), speed_p95=np.percentile(sp, 95),
                     omega_mean=om.mean() if om.size >= MIN_GOOD else np.nan,
                     straight=net / path if path > 0.05 * Lref else np.nan,
                     fwd_mean=fw.mean() if fw.size >= 5 else np.nan,
                     still=float(np.median(sp) < STILL))
        ok = Vp[i]
        if ok.sum() >= MIN_GOOD:
            b = Vb[i][ok]
            bd = b - b.mean()
            pw = (lombscargle(np.arange(WIN)[ok] / FPS, bd, 2 * np.pi * FREQS, normalize=True)
                  if bd.std() > 0 else np.zeros(FREQS.size))
            d.update(junc_abs=np.abs(Vj[i][ok]).mean(), junc_sd=Vj[i][ok].std(),
                     base_abs=np.abs(b).mean(), base_asym=np.abs(b.mean()), base_sd=b.std(),
                     bend_abs=np.abs(Vbd[i][ok]).mean(), bend_sd=Vbd[i][ok].std(),
                     lat_rms=np.sqrt(np.mean(Vl[i][ok] ** 2)), base_vel=np.nanmean(Vv[i][ok]), ls_pow=pw.max())
        rows.append(d)
    return pd.DataFrame(rows), Lref

# =============================================================================
# FRAME POSTURE (steps 05–06)
# =============================================================================
def frame_posture(P, Lref):
    """Per frame: segment angles a0–a8 relative to the trunk axis (deg), speed (Lref/s), body rotation (deg/s)."""
    Q = {b: np.column_stack([fill_gaps(P[b][:, 0]), fill_gaps(P[b][:, 1])]) for b in dict.fromkeys(TRUNK + CHAIN)}
    axis = Q["bodypart2"] - Q["bodypart1"]
    th = np.arctan2(axis[:, 1], axis[:, 0])
    ang = lambda a, b: np.arctan2(Q[b][:, 1] - Q[a][:, 1], Q[b][:, 0] - Q[a][:, 0])
    A = np.column_stack([wrap(ang(a, b) - th) for a, b in zip(CHAIN[:-1], CHAIN[1:])]) * 180 / np.pi
    tr = np.stack([Q[b] for b in TRUNK])
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        com = np.where((np.isfinite(tr[..., 0]).sum(0) >= 2)[:, None], np.nanmean(tr, axis=0), np.nan)
    com = pd.DataFrame(com).rolling(5, center=True, min_periods=3).mean().to_numpy()
    speed = np.r_[np.nan, np.hypot(*np.diff(com, axis=0).T)] * FPS / Lref
    omega = np.r_[np.nan, np.abs(wrap(np.diff(th)))] * FPS * 180 / np.pi
    return A, speed, omega

def normalised(P, f, lref, flip, parts):
    """Anatomical frame: origin = trunk centre, bp1→bp2 along +x, junction (bp3) on the +y side; scaled to tail length.
    (flip is kept for call compatibility; the junction side is fixed per frame.)"""
    pts = np.array([P[b][f] for b in parts], float)
    a, b = P["bodypart1"][f], P["bodypart2"][f]
    if not (np.isfinite(pts).all() and np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(P["bodypart3"][f]).all()):
        return None
    th = np.arctan2(*(b - a)[::-1])
    R = np.array([[np.cos(-th), -np.sin(-th)], [np.sin(-th), np.cos(-th)]])
    origin = (a + b) / 2
    xy = (pts - origin) @ R.T / lref
    if ((P["bodypart3"][f] - origin) @ R.T)[1] < 0:
        xy[:, 1] *= -1
    return xy

def resample(xy, n=25):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    if d[-1] <= 0:
        return None
    s = np.linspace(0, d[-1], n)
    return np.column_stack([np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])])

def curvature(xy):
    """Signed and absolute total turning along the tail (deg)."""
    th = np.arctan2(*np.diff(xy, axis=0).T[::-1])
    turn = np.angle(np.exp(1j * np.diff(th)))
    return np.degrees(turn.sum()), np.degrees(np.abs(turn).sum())

def still_runs(g):
    """Runs of consecutive still windows lasting ≥ STILL_MIN_S: list of (first_idx, last_idx, duration_s).
    g: one animal's windows, sorted by WindowStart, index reset."""
    s = (g["still"] == 1).to_numpy()
    cont = np.r_[False, np.diff(g["WindowStart"].to_numpy()) == STRIDE]
    runs, i, n = [], 0, len(s)
    while i < n:
        if s[i]:
            j = i
            while j + 1 < n and s[j + 1] and cont[j + 1]:
                j += 1
            dur = (j - i + 1) * DT + (WIN - STRIDE) / FPS
            if dur >= STILL_MIN_S:
                runs.append((i, j, dur))
            i = j + 1
        else:
            i += 1
    return runs

# =============================================================================
# STATISTICS
# =============================================================================
def bh(p):
    p = np.asarray(p, float); out = np.full(p.shape, np.nan); ok = np.isfinite(p)
    if ok.any():
        q = p[ok]; o = np.argsort(q)
        r = np.minimum.accumulate((q[o] * q.size / np.arange(1, q.size + 1))[::-1])[::-1]
        res = np.empty(q.size); res[o] = np.minimum(r, 1); out[ok] = res
    return out

def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "ns"

def compare_cols(d, pairs, cols, tag):
    """Mann-Whitney per column for each group pair (per-animal values); BH across columns per pair."""
    rows = []
    for a, b in pairs:
        for f in cols:
            x, y = d.loc[d["group"] == a, f].dropna(), d.loc[d["group"] == b, f].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            U, p = mannwhitneyu(x, y, alternative="two-sided")
            rows.append(dict(comparison=tag, A=a, B=b, feature=f, nA=len(x), nB=len(y), mean_A=x.mean(),
                             mean_B=y.mean(), r_rb_B_higher=1 - 2 * U / (len(x) * len(y)), p=p))
    st = pd.DataFrame(rows)
    if len(st):
        st["q_BH"] = st.groupby(["A", "B"])["p"].transform(bh)
    return st

def desc(v, tag):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    q25, q75 = np.percentile(v, [25, 75]) if v.size else (np.nan, np.nan)
    return {f"{tag}_n": int(v.size), f"{tag}_mean": v.mean() if v.size else np.nan,
            f"{tag}_sd": v.std(ddof=1) if v.size > 1 else np.nan, f"{tag}_median": np.median(v) if v.size else np.nan,
            f"{tag}_q25": q25, f"{tag}_q75": q75}

def mwu_delta(x, y):
    """Mann-Whitney x vs y; Cliff's delta > 0 = y higher."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size < 1 or y.size < 1:
        return dict(U=np.nan, p=np.nan, cliffs_delta=np.nan)
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    return dict(U=U, p=p, cliffs_delta=1 - 2 * U / (x.size * y.size))

def chi2_stat(Cc, Co):
    tab = np.vstack([Cc, Co]); tab = tab[:, tab.sum(0) > 0]
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return np.sum((tab - exp) ** 2 / exp)

# =============================================================================
# FIGURES
# =============================================================================
def save_fig(fig, folder, name, dpi=300, pdf=False):
    """Every figure as PNG + SVG (+ PDF); dpi also sets the resolution of rasterized layers in SVG/PDF."""
    os.makedirs(folder, exist_ok=True)
    for ext in ("png", "svg") + (("pdf",) if pdf else ()):
        fig.savefig(f"{folder}/{name}.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def clean_axes(ax):
    """UMAP map axes: no ticks/spines, small axis arrows."""
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.annotate("", xy=(0.12, 0.0), xytext=(0.0, 0.0), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.annotate("", xy=(0.0, 0.12), xytext=(0.0, 0.0), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.5))
    ax.text(0.06, -0.03, "UMAP 1", transform=ax.transAxes, fontsize=5.5, ha="center", va="top")
    ax.text(-0.03, 0.06, "UMAP 2", transform=ax.transAxes, fontsize=5.5, ha="right", va="center", rotation=90)
