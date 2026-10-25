#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08 — Publication versions of the whole-video maps from the SAVED embedding (step 06).
Fixed mm layout (no bbox tight), no text over data points, region numbers as a separate transparent
overlay with identical geometry (paste in place, move/restyle). Text stays editable (svg/pdf fonttype).

Outputs (OUT/fig_embedding_whole/production/):
    E1_map_phases_overlay_nolabels (+ _LABELS)       free time grey, expansion / inside house by condition
    E2_map_phases_by_condition_nolabels (+ _LABELS)
    A_map_regions_points_vector                       subsampled vector points
    A_map_regions_filled (+ A_map_regions_LABELS)     solid region polygons clipped to the data
    region_label_positions.csv
"""
import pickle
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter
from common import *

OUTW = f"{OUT}/fig_embedding_whole"
OUTP = f"{OUTW}/production"; os.makedirs(OUTP, exist_ok=True)
plt.rcParams.update({**FIG_RC, "pdf.fonttype": 42, "ps.fonttype": 42})

def save_fixed(fig, name, transparent=False, exts=("png", "svg", "pdf")):
    for e_ in exts:                                   # no bbox_inches="tight": data figure and overlay stay aligned
        fig.savefig(f"{OUTP}/{name}.{e_}", dpi=600, transparent=transparent)
    plt.close(fig)

# =============================================================================
# 1. saved embedding + regions
# =============================================================================
W = pd.read_parquet(f"{OUTW}/windows_whole_embedded.parquet")
with open(f"{OUTW}/model_whole.pkl", "rb") as fh:
    M = pickle.load(fh)
WSw, ext_w = M["ws_map"], M["ext"]
REGS_W = sorted(W["region"].dropna().unique().astype(int))
COL_W = {k: BASE_PAL[i % len(BASE_PAL)] for i, k in enumerate(REGS_W)}
print(f"loaded {len(W):,} windows, {W['Experiment_ID'].nunique()} animals, {len(REGS_W)} regions")

W["phase_grp"] = np.select([W["phase"] == "HE", W["phase"].isin(["in_house", "housed"]),
                            W["phase"].isin(["free", "pre", "after_escape"])], ["expansion", "inside house", "free"], default="")
ext_d = (np.percentile(W["wx"], 0.2), np.percentile(W["wx"], 99.8), np.percentile(W["wy"], 0.2), np.percentile(W["wy"], 99.8))
pad = 0.05 * np.array([ext_d[1] - ext_d[0], ext_d[3] - ext_d[2]])
ext_d = (ext_d[0] - pad[0], ext_d[1] + pad[0], ext_d[2] - pad[1], ext_d[3] + pad[1])

# =============================================================================
# 2. colours / sampling (equal windows per condition, capped per animal)
# =============================================================================
GREY_C = {"control": "0.60", "6-OHDA": "0.84"}
PH_C = {("expansion", "control"): "#B35806", ("expansion", "6-OHDA"): "#FDB863",
        ("inside house", "control"): "#542788", ("inside house", "6-OHDA"): "#B2ABD2"}
CAP_FREE, CAP_PH = 150, 80

def per_animal_sample(d, cap, seed=1):
    return d.groupby("Experiment_ID", group_keys=False).apply(lambda g: g.sample(min(len(g), cap), random_state=seed))

def equalised(phase, cap):
    v = {c: per_animal_sample(W[(W["phase_grp"] == phase) & (W["cond"] == c)], cap) for c in ("control", "6-OHDA")}
    n = min(len(x) for x in v.values())
    return {c: x.sample(n, random_state=2) if n else x for c, x in v.items()}

FREE = equalised("free", CAP_FREE)
PHS = {ph: equalised(ph, CAP_PH) for ph in ("expansion", "inside house")}

LAB_POS = pd.DataFrame([dict(region=k + 1, x=W.loc[W["region"] == k, "wx"].median(), y=W.loc[W["region"] == k, "wy"].median())
                        for k in REGS_W])
LAB_POS.to_csv(f"{OUTP}/region_label_positions.csv", index=False)

# =============================================================================
# 3. drawing helpers
# =============================================================================
def fixed_fig(w_mm, h_mm, rects_mm):
    """Figure with axes at fixed mm positions [(left, bottom, width, height), ...]."""
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    return fig, [fig.add_axes([l / w_mm, b / h_mm, w / w_mm, h / h_mm]) for l, b, w, h in rects_mm]

def frame_axes(ax):
    ax.set_xlim(ext_d[:2]); ax.set_ylim(ext_d[2:])
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)

def septa(ax):
    ax.contour(WSw, levels=np.arange(1.5, WSw.max() + 0.5), extent=ext_w, colors="0.15", linewidths=0.45, zorder=5)

def draw(ax, conds):
    for c in conds:
        ax.scatter(FREE[c]["wx"], FREE[c]["wy"], s=0.6, color=GREY_C[c], lw=0, rasterized=True, zorder=1)
    for ph in ("inside house", "expansion"):
        for c in conds:
            v = PHS[ph][c]
            ax.scatter(v["wx"], v["wy"], s=2.2, color=PH_C[(ph, c)], alpha=0.75, lw=0, rasterized=True, zorder=3)

def label_overlay(w_mm, h_mm, rects_mm, name):
    """Transparent page, same geometry, region numbers only."""
    fig, axes = fixed_fig(w_mm, h_mm, rects_mm)
    for ax in axes:
        frame_axes(ax); ax.patch.set_alpha(0)
        for r in LAB_POS.itertuples():
            ax.text(r.x, r.y, str(r.region), fontsize=6, weight="bold", ha="center", va="center")
    save_fixed(fig, name, transparent=True, exts=("svg", "pdf"))

legend_items = [mpatches.Patch(color=GREY_C[c], label=f"free · {c}") for c in ("control", "6-OHDA")] + \
               [mpatches.Patch(color=PH_C[k], label=f"{k[0]} · {k[1]}") for k in PH_C]

# =============================================================================
# E1: overlay of both conditions · E2: split by condition
# =============================================================================
E1_W, E1_H, E1_R = 135, 72, [(8, 7, 78, 62)]
fig, (ax,) = fixed_fig(E1_W, E1_H, E1_R)
draw(ax, ("control", "6-OHDA")); septa(ax); frame_axes(ax); clean_axes(ax)
ax.legend(handles=legend_items, fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
save_fixed(fig, "E1_map_phases_overlay_nolabels")
label_overlay(E1_W, E1_H, E1_R, "E1_map_phases_overlay_LABELS")

E2_W, E2_H, E2_R = 195, 74, [(8, 7, 70, 56), (84, 7, 70, 56)]
fig, axes = fixed_fig(E2_W, E2_H, E2_R)
for ax, c in zip(axes, ("control", "6-OHDA")):
    draw(ax, (c,)); septa(ax); frame_axes(ax); clean_axes(ax)
    ax.set_title(f"{c}: {W.loc[(W['cond'] == c) & (W['phase_grp'] == 'expansion'), 'Experiment_ID'].nunique()} builders, "
                 f"{W.loc[(W['cond'] == c) & (W['phase_grp'] == 'inside house'), 'Experiment_ID'].nunique()} animals inside house",
                 fontsize=6.5, color=COND_COL[c], pad=4)
axes[-1].legend(handles=legend_items, fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
save_fixed(fig, "E2_map_phases_by_condition_nolabels")
label_overlay(E2_W, E2_H, E2_R, "E2_map_phases_by_condition_LABELS")

# =============================================================================
# A: regions — vector points (subsampled) and filled region polygons
# =============================================================================
reg_handles = [mpatches.Patch(color=COL_W[k], label=rname_w(k)) for k in REGS_W]
A_W, A_H, A_R = 125, 72, [(8, 7, 78, 62)]
N_PTS_A = 15000
ptsA = W[W["region"].notna()].sample(min(int(W["region"].notna().sum()), N_PTS_A), random_state=2)
fig, (ax,) = fixed_fig(A_W, A_H, A_R)
for k in REGS_W:
    m_ = ptsA[ptsA["region"] == k]
    ax.scatter(m_["wx"], m_["wy"], s=1.2, color=COL_W[k], alpha=0.8, lw=0, rasterized=False)
frame_axes(ax); clean_axes(ax)
ax.legend(handles=reg_handles, fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
save_fixed(fig, "A_map_regions_points_vector")

# pixel label → region (majority vote of the saved windows; robust to renumbering after the watershed)
ny, nx = WSw.shape
gi = np.clip(((W["wy"] - ext_w[2]) / (ext_w[3] - ext_w[2]) * ny).astype(int), 0, ny - 1)
gj = np.clip(((W["wx"] - ext_w[0]) / (ext_w[1] - ext_w[0]) * nx).astype(int), 0, nx - 1)
pix_lab = pd.Series(WSw[gi, gj]).where(W["region"].notna().to_numpy())
reg_of_pix = (pd.DataFrame(dict(pix=pix_lab, reg=W["region"].to_numpy())).dropna()
              .groupby("pix")["reg"].agg(lambda s: s.mode().iloc[0]).to_dict())
Rmap = np.vectorize(lambda v: reg_of_pix.get(v, np.nan))(WSw).astype(float)
H, _, _ = np.histogram2d(W["wy"], W["wx"], bins=[ny, nx], range=[ext_w[2:], ext_w[:2]])
Hs = gaussian_filter(H, 1.5)
v_ = np.sort(Hs.ravel())[::-1]
thr = v_[np.searchsorted(np.cumsum(v_), 0.995 * v_.sum())]          # outline encloses 99.5 % of windows
Rmask = np.ma.masked_where((Hs < thr) | ~np.isfinite(Rmap), Rmap)

cmapR = ListedColormap([COL_W[k] for k in REGS_W])
normR = BoundaryNorm(np.arange(min(REGS_W) - 0.5, max(REGS_W) + 1.5), cmapR.N)
fig, (ax,) = fixed_fig(A_W, A_H, A_R)
ax.contourf(Rmask, levels=np.arange(min(REGS_W) - 0.5, max(REGS_W) + 1.5), extent=ext_w, cmap=cmapR, norm=normR, alpha=0.9, antialiased=True)
ax.contour((Hs >= thr).astype(float), levels=[0.5], extent=ext_w, colors="0.15", linewidths=0.45)
septa(ax); frame_axes(ax); clean_axes(ax)
ax.legend(handles=reg_handles, fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
save_fixed(fig, "A_map_regions_filled")
label_overlay(A_W, A_H, A_R, "A_map_regions_LABELS")
print(f"saved → {OUTP}")
