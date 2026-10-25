#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06 — Whole-video posture-dynamics embedding.
Non-overlapping 1 s windows over whole videos → 38 features (eigen-tail medians, Lomb-Scargle spectra,
speed, rotation) → UMAP fitted on a fit set balanced over builders/non-builders x control/6-OHDA →
behaviour-balanced watershed (σ = 6) → adjacent regions merged by behaviour composition (Jensen-Shannon) to 8.

Outputs (OUT/fig_embedding_whole/):
    windows_whole_embedded.parquet   every window with map coordinates (wx, wy), region, behaviour labels
    model_whole.pkl                  pca_t, scaler, UMAP reducer, watershed map, extent, features, parameters
    region_composition.csv, A_map_regions_whole, B_map_annotations_whole, C_region_composition_whole

The published maps and statistics use the SAVED embedding (UMAP is not bit-reproducible across
library versions/platforms); this step refuses to overwrite it unless OVERWRITE = True.
"""
import pickle
import umap
import seaborn as sns
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from sklearn.preprocessing import StandardScaler
from common import *

OUTW = f"{OUT}/fig_embedding_whole"; os.makedirs(OUTW, exist_ok=True)
OVERWRITE = False
if os.path.isfile(f"{OUTW}/model_whole.pkl") and not OVERWRITE:
    raise SystemExit(f"saved embedding exists in {OUTW} — set OVERWRITE = True to refit")

WIN_W, SPEC_MIN_W = 30, 20             # 1 s windows; frames with proximal posture needed
NN_W, MD_W, SIGMA_W, TARGET_W = 15, 0.1, 6, 8
EMB_SEED = 0
WS_BINS, WS_MIN_DIST = 200, 8
MIN_REGION_SHARE, MIN_REGION_ANIMALS = 0.003, 10
COHORT_OF = {"C+": "builders", "O+": "builders", "C-": "non-builders", "O-": "non-builders",
             "C?": "non-builders", "O?": "non-builders", "Ch": "housed", "Oh": "housed"}
MODE_NAME = {0: "rest", 1: "slow", 2: "tail beating", 3: "fast"}
BEH_W = ["rest", "slow", "tail beating", "fast", "expansion", "in house"]
PCN = [f"pc{k + 1}" for k in range(4)]
FEATS_E = [f"{p}_med" for p in PCN] + [f"{p}_f{j}" for p in PCN for j in range(len(FREQ_W))] + ["lspeed", "lomega"]
plt.rcParams.update(FIG_RC)

meta2 = load_meta2()
win = load_windows()
okm = valid_animals(meta2)
by_id = {e: g for e, g in win.groupby("Experiment_ID")}
lref_of = meta2.set_index("Experiment_ID")["Lref_px"]
abn = load_abnormal()
with open(f"{OUT}/05_posture/05_eigentail_pca.pkl", "rb") as fh:
    pca_t = pickle.load(fh)
ids_w = okm[okm["group"].isin(COHORT_OF) & ~okm["Experiment_ID"].isin(SUSPECT_HOUSED)]
print(f"animals: {len(ids_w)}", ids_w["group"].value_counts().to_dict())

# =============================================================================
# 1. window features over whole videos
# =============================================================================
rows = []
for i, r in enumerate(ids_w.itertuples()):
    e = r.Experiment_ID
    P, _ = clean_frames(r.path)
    A, speed, omega = frame_posture(P, lref_of.get(e, lref_of.median()))
    if np.nanmedian(A[:, 0]) < 0:                      # chirality: median junction angle positive
        A = -A
    T = A.shape[0]
    mrow = meta2[meta2["Experiment_ID"] == e].iloc[0]
    ph = phase_vector(mrow, T)
    prox = np.isfinite(A[:, :4]).all(axis=1)
    Zp = np.full((T, 4), np.nan); Zp[prox] = pca_t.transform(A[prox, :4])
    mode_at = dict(zip(by_id[e]["WindowStart"].to_numpy(), by_id[e]["mode"].to_numpy())) if e in by_id else {}
    ab = np.full(T, "", dtype=object)
    for x in abn[abn["Experiment_ID"] == e].itertuples():
        ab[max(0, int(x.start)):min(T, int(x.end) + 1)] = "abnormal start" if "abnormal" in str(x.event).lower() else "failed insertion"
    hs = mrow["he_start"] if pd.notna(mrow["he_start"]) else None
    tt = np.arange(T) / FPS
    for s in range(0, T - WIN_W + 1, WIN_W):
        sl = slice(s, s + WIN_W); ok = prox[sl]
        if ok.sum() < SPEC_MIN_W:
            continue
        row = dict(Experiment_ID=e, group=r.group, batch=r.batch, cond=r.cond, cohort=COHORT_OF[r.group],
                   WindowStart=s, phase=ph[s + WIN_W // 2], mode=mode_at.get(s, np.nan), a0=np.nanmedian(A[sl, 0]))
        for k in range(4):
            x = Zp[sl, k][ok]
            row[f"pc{k + 1}_med"] = np.median(x)
            x = x - x.mean()
            pw = lombscargle(tt[sl][ok], x, 2 * np.pi * FREQ_W, normalize=False) if x.std() > 0 else np.zeros(len(FREQ_W))
            row.update({f"pc{k + 1}_f{j}": np.log1p(v) for j, v in enumerate(pw)})
        row["lspeed"] = np.log10(np.nanmedian(speed[sl]) + 1e-3)
        row["lomega"] = np.log10(np.nanmedian(omega[sl]) + 1e-1)
        abw = pd.Series(ab[sl]); abw = abw[abw != ""]
        row["ab"] = abw.mode().iloc[0] if len(abw) else ""
        row["t_to_onset"] = (s + WIN_W // 2) / FPS - hs / FPS if hs is not None else np.nan
        rows.append(row)
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(ids_w)} animals, {len(rows):,} windows")
W = pd.DataFrame(rows).dropna(subset=FEATS_E).reset_index(drop=True)

# labels for colouring / naming only (not map inputs)
W["behav"] = np.select([W["phase"] == "HE", W["phase"].isin(["in_house", "housed"])], ["expansion", "in house"],
                       default=W["mode"].map(MODE_NAME).fillna("")).astype(str)
W["annot"] = np.select([W["ab"] != "", W["phase"] == "HE", W["phase"].isin(["in_house", "housed"]),
                        (W["phase"] == "pre") & W["t_to_onset"].between(-5, 0), W["phase"] == "after_escape"],
                       [W["ab"], "expansion", "inside house", "before expansion (5 s)", "after escape"], default="")
print(W.groupby(["cohort", "cond"]).agg(windows=("lspeed", "size"), animals=("Experiment_ID", "nunique")).to_string())

# =============================================================================
# 2. balanced fit → UMAP → project all windows
# =============================================================================
def take_balanced(g, n):
    per = int(np.ceil(n / g["Experiment_ID"].nunique()))
    out = g.groupby("Experiment_ID", group_keys=False).apply(lambda a: a.sample(min(len(a), per), random_state=EMB_SEED))
    return out.sample(min(n, len(out)), random_state=EMB_SEED)

n_min_w = int(W.groupby(["cohort", "cond"]).size().min())
fit_w = W.groupby(["cohort", "cond"], group_keys=False).apply(lambda g: take_balanced(g, n_min_w))
print(f"fit set: {n_min_w} windows per cohort x condition → {len(fit_w):,} windows")
scaler_w = StandardScaler().fit(fit_w[FEATS_E])
red_w = umap.UMAP(n_neighbors=NN_W, min_dist=MD_W, random_state=EMB_SEED).fit(scaler_w.transform(fit_w[FEATS_E]))
Ew_fit = red_w.embedding_
Ew = red_w.transform(scaler_w.transform(W[FEATS_E]))
W["wx"], W["wy"] = Ew[:, 0], Ew[:, 1]

# =============================================================================
# 3. regions: behaviour-balanced watershed → merge adjacent regions with similar composition → tidy
# =============================================================================
def ws_partition_w(E_fit, weights, E_all, sigma):
    """Watershed on the weighted, smoothed density of the fit windows; returns (label function, pixel map, extent)."""
    lo, hi = E_all.min(0), E_all.max(0)
    pad = 0.05 * (hi - lo)
    ext_ = (lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1])
    H, _, _ = np.histogram2d(E_fit[:, 1], E_fit[:, 0], bins=WS_BINS, range=[ext_[2:], ext_[:2]], weights=weights)
    Hs = gaussian_filter(H, sigma)
    mask = Hs > Hs.max() * 1e-3
    peaks = peak_local_max(Hs, min_distance=WS_MIN_DIST, threshold_rel=0.02, labels=mask.astype(int))
    markers = np.zeros(Hs.shape, int); markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    WS = watershed(-Hs, markers, mask=mask)
    idx = distance_transform_edt(WS == 0, return_distances=False, return_indices=True)
    WS = WS[idx[0], idx[1]]                                       # pixels outside the mapped area → nearest region
    def label(E):
        i = np.clip(((E[:, 1] - ext_[2]) / (ext_[3] - ext_[2]) * WS_BINS).astype(int), 0, WS_BINS - 1)
        j = np.clip(((E[:, 0] - ext_[0]) / (ext_[1] - ext_[0]) * WS_BINS).astype(int), 0, WS_BINS - 1)
        return WS[i, j] - 1
    return label, WS, ext_

def jsd_(p, q, eps=1e-9):
    p, q = p + eps, q + eps; p, q = p / p.sum(), q / q.sum(); m = (p + q) / 2
    return 0.5 * np.sum(p * np.log2(p / m)) + 0.5 * np.sum(q * np.log2(q / m))

def adjacency(ws):
    a, b = ws[:, :-1].ravel(), ws[:, 1:].ravel()
    c, d = ws[:-1, :].ravel(), ws[1:, :].ravel()
    pairs = np.vstack([np.c_[a, b], np.c_[c, d]])
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    return {tuple(sorted(p)) for p in map(tuple, np.unique(pairs, axis=0))}

def tidy_regions(S_, col):
    """Drop rare regions and renumber the rest by median speed (0 = most still)."""
    m = S_[S_["main_set"] & S_[col].notna()]
    share = m[col].value_counts(normalize=True)
    animals = m.groupby(col)["Experiment_ID"].nunique()
    keep = [k for k in share.index if share[k] >= MIN_REGION_SHARE and animals.get(k, 0) >= MIN_REGION_ANIMALS]
    dropped = ~S_[col].isin(keep)
    S_.loc[dropped, col] = np.nan
    speed_order = S_[S_["main_set"] & S_[col].notna()].groupby(col)["lspeed"].median().sort_values().index
    S_[col] = S_[col].map({old: new for new, old in enumerate(speed_order)})
    print(f"{col}: kept {len(keep)} regions; unassigned windows: {dropped[S_['main_set']].mean():.2%}")
    return len(keep)

bf = pd.Series(W.loc[fit_w.index, "behav"].to_numpy())
wts = (1.0 / bf.map(bf[bf != ""].value_counts())).to_numpy()            # each behaviour label weighs the same
wts[bf.to_numpy() == ""] = np.nanmedian(wts[bf.to_numpy() != ""]); wts = wts / wts.sum()
lab_fn_w, WSw, ext_w = ws_partition_w(Ew_fit, wts, Ew, SIGMA_W)
labels_w = lab_fn_w(Ew)
beh_w = W["behav"].to_numpy()

def comp_w(labels):
    return {r_: np.array([np.mean(beh_w[(labels == r_) & (beh_w != "")] == x) if ((labels == r_) & (beh_w != "")).any() else 0.0
                          for x in BEH_W]) for r_ in np.unique(labels)}

ws_m = WSw.copy()                                          # pixel labels are 1-based; window labels = pixel - 1
print(f"\nover-segmentation: {len(np.unique(ws_m))} regions")
while len(np.unique(ws_m)) > TARGET_W:
    cp = comp_w(labels_w)
    (p, q), d = min(((pq, jsd_(cp.get(pq[0] - 1, np.zeros(len(BEH_W))), cp.get(pq[1] - 1, np.zeros(len(BEH_W)))))
                     for pq in adjacency(ws_m)), key=lambda z: z[1])
    ws_m[ws_m == q] = p; labels_w[labels_w == q - 1] = p - 1
    print(f"  merged {q} → {p} (JSD {d:.3f}); {len(np.unique(ws_m))} left")
remap_w = {u: i + 1 for i, u in enumerate(np.unique(ws_m))}
WSw = np.vectorize(remap_w.get)(ws_m)
W["region"] = np.array([remap_w[l + 1] - 1 for l in labels_w], dtype=float)
W["main_set"] = True
n_w = tidy_regions(W, "region")
REGS_W = sorted(W["region"].dropna().unique().astype(int))
COL_W = {k: BASE_PAL[i % len(BASE_PAL)] for i, k in enumerate(REGS_W)}

compw = pd.crosstab(W.loc[W["behav"] != "", "region"].astype(int) + 1, W.loc[W["behav"] != "", "behav"],
                    normalize="index").reindex(columns=BEH_W, fill_value=0)
compw["windows"] = W.groupby(W["region"].dropna().astype(int) + 1).size()
compw["animals"] = W.groupby(W["region"].dropna().astype(int) + 1)["Experiment_ID"].nunique()
print("\nregion composition:"); print(compw.round(2).to_string())
compw.to_csv(f"{OUTW}/region_composition.csv")

# =============================================================================
# 4. save + quick-look figures (publication versions: step 08)
# =============================================================================
W.to_parquet(f"{OUTW}/windows_whole_embedded.parquet")
with open(f"{OUTW}/model_whole.pkl", "wb") as fh:
    pickle.dump(dict(reducer=red_w, scaler=scaler_w, feats=FEATS_E, pca_t=pca_t, ws_map=WSw, ext=ext_w,
                     nn=NN_W, md=MD_W, sigma=SIGMA_W, target=TARGET_W, fit_index=fit_w.index.to_numpy()), fh)

bgW = W.sample(min(len(W), 60000), random_state=2)
fig, ax = plt.subplots(figsize=(85 * MM, 70 * MM))
for k in REGS_W:
    m_ = bgW[bgW["region"] == k]
    ax.scatter(m_["wx"], m_["wy"], s=0.5, color=COL_W[k], alpha=0.5, lw=0, rasterized=True, label=rname_w(k))
ax.legend(fontsize=5.5, markerscale=5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
clean_axes(ax)
save_fig(fig, OUTW, "A_map_regions_whole", dpi=600, pdf=True)

ANN_W = {"before expansion (5 s)": "#4D4D4D", "expansion": "#D95F02", "inside house": "#7570B3",
         "after escape": "#A6761D", "abnormal start": "#E7298A", "failed insertion": "#1B9E77"}
fig, ax = plt.subplots(figsize=(85 * MM, 70 * MM))
ax.scatter(bgW["wx"], bgW["wy"], s=0.4, color="0.86", lw=0, rasterized=True)
for lab, col in ANN_W.items():
    m_ = W[W["annot"] == lab]
    if len(m_) > 4000:
        m_ = m_.sample(4000, random_state=1)
    ax.scatter(m_["wx"], m_["wy"], s=2.5 if lab in ("abnormal start", "failed insertion") else 0.9, color=col, alpha=0.7, lw=0,
               rasterized=True, label=f"{lab} (n={W.loc[W['annot'] == lab, 'Experiment_ID'].nunique()})")
ax.legend(fontsize=5.5, markerscale=4, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
clean_axes(ax)
save_fig(fig, OUTW, "B_map_annotations_whole", dpi=600, pdf=True)

fig, ax = plt.subplots(figsize=(90 * MM, 6 * MM * len(compw) + 15 * MM))
sns.heatmap(compw[BEH_W], annot=True, fmt=".2f", cmap="Greys", vmin=0, vmax=1, cbar=False, linewidths=0.4,
            annot_kws=dict(fontsize=6), ax=ax)
ax.set_yticklabels([rname_w(k - 1) for k in compw.index], rotation=0, fontsize=6)
ax.set_xticklabels(BEH_W, rotation=35, ha="right", fontsize=6); ax.set_xlabel(""); ax.set_ylabel("")
plt.tight_layout()
save_fig(fig, OUTW, "C_region_composition_whole", dpi=600, pdf=True)
print(f"\nsaved → {OUTW}")
