#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07 — Quantification on the saved whole-video embedding (step 06).
    D   time density per condition (each animal weighted equally)
    E   annotated phases by condition
    E3  region composition per phase (free / expansion / inside house): χ² with batch-stratified
        animal-label permutation; per region: permutation on mean fraction, BH within phase
    G/H posture features (step 02) within regions, control vs 6-OHDA, per-animal medians:
        Cliff's delta, batch-stratified permutation (bubble plot) and plain Mann-Whitney (box plots), BH
    F   skeletons per region x condition

Outputs (OUT/fig_embedding_whole/): figures + STATS_*.csv
"""
import pickle
import seaborn as sns
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
from common import *

OUTW = f"{OUT}/fig_embedding_whole"
N_PERM_T = 10000
Q_FAMILY = "all"                 # bubble plot q: "all" (BH across all feature tests) | "within_region"
MW_FAMILY = "all"                # box-plot q, same choice for the Mann-Whitney tests
MIN_WIN_REG, MIN_WIN_F = 5, 3    # windows per animal: composition test / feature medians
DO_SKELETONS = True              # F re-reads every DLC file (slow)
WF = {"straight": "straightness", "fwd_mean": "forward drift", "junc_abs": "junction angle", "junc_sd": "junction SD",
      "base_abs": "tail-base angle", "base_asym": "tail-base asymmetry", "base_sd": "tail-base SD", "bend_abs": "bend",
      "bend_sd": "bend SD", "lat_rms": "lateral excursion", "base_vel": "tail-base velocity", "ls_pow": "rhythmicity"}
plt.rcParams.update(FIG_RC)

W = pd.read_parquet(f"{OUTW}/windows_whole_embedded.parquet")
with open(f"{OUTW}/model_whole.pkl", "rb") as fh:
    M = pickle.load(fh)
WSw, ext_w = M["ws_map"], M["ext"]
meta2 = load_meta2()
win = load_windows()
REGS_W = sorted(W["region"].dropna().unique().astype(int))
COL_W = {k: BASE_PAL[i % len(BASE_PAL)] for i, k in enumerate(REGS_W)}
W["phase_grp"] = np.select([W["phase"] == "HE", W["phase"].isin(["in_house", "housed"]),
                            W["phase"].isin(["free", "pre", "after_escape"])], ["expansion", "inside house", "free"], default="")
ext_d = (np.percentile(W["wx"], 0.2), np.percentile(W["wx"], 99.8), np.percentile(W["wy"], 0.2), np.percentile(W["wy"], 99.8))
pad = 0.05 * np.array([ext_d[1] - ext_d[0], ext_d[3] - ext_d[2]])
ext_d = (ext_d[0] - pad[0], ext_d[1] + pad[0], ext_d[2] - pad[1], ext_d[3] + pad[1])
bgW = W.sample(min(len(W), 60000), random_state=2)

def grey_background(ax):
    ax.scatter(bgW["wx"], bgW["wy"], s=0.35, color="0.82", lw=0, rasterized=True, zorder=0)
    ax.set_xlim(ext_d[:2]); ax.set_ylim(ext_d[2:])

# =============================================================================
# D. density per condition (HDR contours, each animal weighted equally)
# =============================================================================
NB_D, SM_D, HDR_LEVELS = 200, 3.0, (0.25, 0.5, 0.75, 0.9)

def animal_density(d):
    w_ = 1.0 / d.groupby("Experiment_ID")["Experiment_ID"].transform("size").to_numpy()
    H, _, _ = np.histogram2d(d["wx"], d["wy"], bins=NB_D, range=[ext_d[:2], ext_d[2:]], weights=w_)
    H = gaussian_filter(H, SM_D).T
    return H / H.sum()

def hdr_thresholds(H, masses):
    v = np.sort(H.ravel())[::-1]
    return sorted(v[np.searchsorted(np.cumsum(v), m)] for m in masses)

fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 62 * MM), sharex=True, sharey=True)
for ax, c in zip(axes, ("control", "6-OHDA")):
    d = W[W["cond"] == c]
    H = animal_density(d)
    th = hdr_thresholds(H, HDR_LEVELS) + [H.max() * 1.01]
    grey_background(ax)
    ax.contourf(H, levels=th, extent=ext_d, cmap="Blues", alpha=0.75, zorder=1, origin="lower")
    ax.contour(H, levels=th[:-1], extent=ext_d, colors="#08306B", linewidths=0.4, zorder=2, origin="lower")
    ax.set_title(f"{c} (n={d['Experiment_ID'].nunique()} animals)", fontsize=7, color=COND_COL[c])
    clean_axes(ax)
fig.text(0.5, 0.01, "contours enclose 25 / 50 / 75 / 90 % of time (each animal weighted equally)", ha="center", fontsize=5.5)
plt.tight_layout(rect=(0, 0.04, 1, 1))
save_fig(fig, OUTW, "D_map_density_by_condition", dpi=600, pdf=True)

# =============================================================================
# E. annotated phases by condition (equal numbers of windows per condition, ≤ 60 per animal)
# =============================================================================
PHASE_SETS = {"before expansion (5 s)": W["annot"] == "before expansion (5 s)", "expansion": W["annot"] == "expansion",
              "inside house": W["phase"].isin(["in_house", "housed"])}
PER_ANIMAL_E = 60
fig, axes = plt.subplots(1, 3, figsize=(180 * MM, 58 * MM), sharex=True, sharey=True)
for ax, (lab, mask) in zip(axes, PHASE_SETS.items()):
    grey_background(ax)
    sub = W[mask]
    per_cond = {c: sub[sub["cond"] == c].groupby("Experiment_ID", group_keys=False)
                     .apply(lambda g: g.sample(min(len(g), PER_ANIMAL_E), random_state=1)) for c in COND_COL}
    n_eq = min(len(v) for v in per_cond.values()) if all(len(v) for v in per_cond.values()) else 0
    for c in ("control", "6-OHDA"):
        v = per_cond[c]
        if len(v) == 0:
            continue
        if n_eq:
            v = v.sample(n_eq, random_state=2)
        ax.scatter(v["wx"], v["wy"], s=2.0, color=COND_COL[c], alpha=0.55, lw=0, rasterized=True, zorder=2,
                   label=f"{c} ({sub.loc[sub['cond'] == c, 'Experiment_ID'].nunique()} animals)")
    ax.set_title(lab, fontsize=7)
    ax.legend(fontsize=5, markerscale=3, frameon=False, loc="lower right")
    clean_axes(ax)
plt.tight_layout()
save_fig(fig, OUTW, "E_map_annotated_phases_by_condition", dpi=600, pdf=True)

# =============================================================================
# permutation helpers (batch-stratified: labels shuffled within batch)
# =============================================================================
rng_t = np.random.default_rng(41)

def permuted_labels(labs, bats, n):
    out = np.empty((n, len(labs)), dtype=object)
    for i in range(n):
        lp = labs.copy()
        if bats is not None:
            for b in np.unique(bats):
                ix = np.flatnonzero(bats == b); lp[ix] = rng_t.permutation(labs[ix])
        else:
            lp = rng_t.permutation(labs)
        out[i] = lp
    return out

def perm_diff(vals, labs, PL):
    """Difference of means (6-OHDA − control) with permutation p."""
    ok = np.isfinite(vals)
    v, l, P_ = vals[ok], labs[ok], PL[:, ok]
    if (l == "control").sum() < 3 or (l == "6-OHDA").sum() < 3:
        return np.nan, np.nan
    obs = np.nanmean(v[l == "6-OHDA"]) - np.nanmean(v[l == "control"])
    null = np.array([np.nanmean(v[p_ == "6-OHDA"]) - np.nanmean(v[p_ == "control"]) for p_ in P_])
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)

def perm_delta(vals, labs, PL):
    """Mann-Whitney statistic as Cliff's delta (6-OHDA higher > 0) with p from the given label permutations."""
    ok = np.isfinite(vals)
    v, l, P_ = vals[ok], labs[ok], PL[:, ok]
    nO, nC = (l == "6-OHDA").sum(), (l == "control").sum()
    if nO < 3 or nC < 3:
        return np.nan, np.nan
    r = rankdata(v)
    def delta(lbl):
        return 2 * (r[lbl == "6-OHDA"].sum() - nO * (nO + 1) / 2) / (nO * nC) - 1
    obs = delta(l)
    null = np.array([delta(p_) for p_ in P_])
    return obs, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (len(null) + 1)

# =============================================================================
# E3. region composition per phase
# =============================================================================
comp_rows, overall_rows, E3_FRAC = [], [], {}
for ph in ("free", "expansion", "inside house"):
    d = W[(W["phase_grp"] == ph) & W["region"].notna()]
    cnt = (d.groupby(["Experiment_ID", "cond", "batch"])["region"]
             .apply(lambda s: s.astype(int).value_counts().reindex(REGS_W, fill_value=0)).unstack(fill_value=0))
    cnt = cnt[cnt.sum(axis=1) >= MIN_WIN_REG]
    lab, bat = cnt.index.get_level_values("cond").to_numpy(), cnt.index.get_level_values("batch").to_numpy()
    C, frac = cnt.to_numpy(), cnt.div(cnt.sum(axis=1), axis=0)
    E3_FRAC[ph] = (frac, lab)
    PL_s, PL_u = permuted_labels(lab, bat, N_PERM_T), permuted_labels(lab, None, N_PERM_T)
    chi_obs = chi2_stat(C[lab == "control"].sum(0), C[lab == "6-OHDA"].sum(0))
    chi_null_s = np.array([chi2_stat(C[p_ == "control"].sum(0), C[p_ == "6-OHDA"].sum(0)) for p_ in PL_s])
    chi_null_u = np.array([chi2_stat(C[p_ == "control"].sum(0), C[p_ == "6-OHDA"].sum(0)) for p_ in PL_u])
    overall_rows.append(dict(phase=ph, control_n_animals=int((lab == "control").sum()), sixOHDA_n_animals=int((lab == "6-OHDA").sum()),
                             chi2=chi_obs, p_perm_stratified=(np.sum(chi_null_s >= chi_obs) + 1) / (N_PERM_T + 1),
                             p_perm_unstratified=(np.sum(chi_null_u >= chi_obs) + 1) / (N_PERM_T + 1)))
    rows_ph = []
    for k in REGS_W:
        v = frac[k].to_numpy(float)
        dm, p_s = perm_diff(v, lab, PL_s)
        _, p_u = perm_diff(v, lab, PL_u)
        rows_ph.append(dict(phase=ph, region=k + 1, region_name=rname_w(k), **desc(v[lab == "control"], "control"),
                            **desc(v[lab == "6-OHDA"], "6OHDA"), diff_mean_6OHDA_minus_control=dm,
                            p_perm_stratified=p_s, p_perm_unstratified=p_u,
                            cliffs_delta=mwu_delta(v[lab == "control"], v[lab == "6-OHDA"])["cliffs_delta"]))
    r_ = pd.DataFrame(rows_ph); r_["q_BH_within_phase"] = bh(r_["p_perm_stratified"]); comp_rows.append(r_)
COMP_OVERALL, COMP = pd.DataFrame(overall_rows), pd.concat(comp_rows, ignore_index=True)
COMP_OVERALL.to_csv(f"{OUTW}/STATS_composition_overall_permutation.csv", index=False)
COMP.to_csv(f"{OUTW}/STATS_composition_per_region_permutation.csv", index=False)
print("composition — overall (χ², batch-stratified permutation):"); print(COMP_OVERALL.round(4).to_string(index=False))
print(COMP[COMP["q_BH_within_phase"] < 0.05][["phase", "region_name", "control_mean", "6OHDA_mean", "p_perm_stratified",
                                             "q_BH_within_phase", "cliffs_delta"]].round(4).to_string(index=False))

PH_LABEL = {"free": "free time\n(non-builders, before expansion, after escape)", "expansion": "expansion", "inside house": "inside house"}
E3_P = COMP_OVERALL.set_index("phase")["p_perm_stratified"].to_dict()
fig, axes = plt.subplots(1, 3, figsize=(160 * MM, 55 * MM), sharey=True)
rows_e3 = []
for ax, ph in zip(axes, ("free", "expansion", "inside house")):
    frac, lab = E3_FRAC[ph]
    for xi, c in enumerate(("control", "6-OHDA")):
        mf, bottom = frac[lab == c].mean(), 0
        for k in REGS_W:
            ax.bar(xi, mf[k], bottom=bottom, color=COL_W[k], edgecolor="white", lw=0.3, width=0.7)
            bottom += mf[k]
        rows_e3.append(dict(phase=ph, cond=c, animals=int((lab == c).sum()), **{rname_w(k): mf[k] for k in REGS_W}))
    ax.set_title(f"{PH_LABEL[ph]}\nχ² permutation p = {E3_P[ph]:.3f}", fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"control\n{int((lab == 'control').sum())}", f"6-OHDA\n{int((lab == '6-OHDA').sum())}"], fontsize=6)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=5.5)
axes[0].set_ylabel("fraction of windows (mean over animals)", fontsize=6)
axes[-1].legend(handles=[mpatches.Patch(color=COL_W[k], label=f"{k + 1}: {rname_w(k)}") for k in REGS_W], fontsize=5.5,
                frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
save_fig(fig, OUTW, "E3_phase_region_distribution", dpi=600, pdf=True)
pd.DataFrame(rows_e3).to_csv(f"{OUTW}/E3_phase_region_distribution.csv", index=False)

# =============================================================================
# G/H. posture features within regions (step-02 window starting at the same frame)
# =============================================================================
wf = win[["Experiment_ID", "WindowStart"] + list(WF)].drop_duplicates(["Experiment_ID", "WindowStart"])
WM = W[W["region"].notna()].merge(wf, on=["Experiment_ID", "WindowStart"], how="left")
FAW = (WM.groupby(["region", "cond", "Experiment_ID"]).filter(lambda g: g[list(WF)].notna().any(axis=1).sum() >= MIN_WIN_F)
         .groupby(["region", "cond", "Experiment_ID"])[list(WF)].median().reset_index())
FAW["batch"] = FAW["Experiment_ID"].map(meta2.set_index("Experiment_ID")["batch"])
FAW.to_csv(f"{OUTW}/G_features_per_animal_region.csv", index=False)

feat_rows, mw_rows = [], []
for k in REGS_W:
    d = FAW[FAW["region"] == k]
    lab, bat = d["cond"].to_numpy(), d["batch"].to_numpy()
    if (lab == "control").sum() >= 3 and (lab == "6-OHDA").sum() >= 3:
        PL_s, PL_u = permuted_labels(lab, bat, N_PERM_T), permuted_labels(lab, None, N_PERM_T)
        for f in WF:
            v = d[f].to_numpy(float)
            dlt, p_s = perm_delta(v, lab, PL_s)                 # batch-stratified Mann-Whitney
            _, p_u = perm_delta(v, lab, PL_u)                   # unstratified ≈ ordinary Mann-Whitney
            feat_rows.append(dict(region=k + 1, region_k=k, region_name=rname_w(k), feature=f, feature_label=WF[f],
                                  **desc(v[lab == "control"], "control"), **desc(v[lab == "6-OHDA"], "6OHDA"),
                                  diff_median_6OHDA_minus_control=np.nanmedian(v[lab == "6-OHDA"]) - np.nanmedian(v[lab == "control"]),
                                  p_perm_stratified=p_s, p_perm_unstratified=p_u, cliffs_delta=dlt))
    for f in WF:                                                # plain Mann-Whitney (box plots)
        x, y = d.loc[d["cond"] == "control", f].dropna(), d.loc[d["cond"] == "6-OHDA", f].dropna()
        row = dict(region=k + 1, region_k=k, region_name=rname_w(k), feature=f, feature_label=WF[f], **desc(x, "control"), **desc(y, "6OHDA"))
        if len(x) >= 3 and len(y) >= 3:
            U, p = mannwhitneyu(x, y, alternative="two-sided")
            row.update(MW_U=U, MW_p=p, cliffs_delta=1 - 2 * U / (len(x) * len(y)))
        else:
            row.update(MW_U=np.nan, MW_p=np.nan, cliffs_delta=np.nan)
        mw_rows.append(row)
FEAT = pd.DataFrame(feat_rows)
FEAT["q_BH_within_region"] = FEAT.groupby("region")["p_perm_stratified"].transform(bh)
FEAT["q_BH_all_tests"] = bh(FEAT["p_perm_stratified"])
FEAT.to_csv(f"{OUTW}/STATS_features_per_region_permutation.csv", index=False)
MW = pd.DataFrame(mw_rows)
MW["MW_q_BH_within_region"] = MW.groupby("region")["MW_p"].transform(bh)
MW["MW_q_BH_all_tests"] = bh(MW["MW_p"])
MW.to_csv(f"{OUTW}/STATS_features_per_region_MannWhitney.csv", index=False)
qcol = "q_BH_all_tests" if Q_FAMILY == "all" else "q_BH_within_region"
print(f"\nfeatures: {FEAT['p_perm_stratified'].notna().sum()} permutation tests, q < 0.05: {(FEAT[qcol] < 0.05).sum()} ({qcol}); "
      f"Mann-Whitney q < 0.05: {(MW['MW_q_BH_all_tests' if MW_FAMILY == 'all' else 'MW_q_BH_within_region'] < 0.05).sum()}")

# G: bubble plot (permutation Cliff's delta; size = −log10 q; black edge = q < 0.05)
SG = FEAT.assign(q_BH=FEAT[qcol]).dropna(subset=["cliffs_delta", "q_BH"])
fig, ax = plt.subplots(figsize=(120 * MM, 9 * MM * len(REGS_W) + 25 * MM))
fx = {f: i for i, f in enumerate(WF)}; ry = {k: i for i, k in enumerate(REGS_W)}
sc_ = ax.scatter(SG["feature"].map(fx), SG["region_k"].map(ry), c=SG["cliffs_delta"], cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                 s=20 + 90 * np.clip(-np.log10(SG["q_BH"]), 0, 3), edgecolors=np.where(SG["q_BH"] < 0.05, "k", "none"), linewidths=0.8)
ax.set_xticks(range(len(WF))); ax.set_xticklabels(list(WF.values()), rotation=40, ha="right", fontsize=6)
ax.set_yticks(range(len(REGS_W))); ax.set_yticklabels([f"{k + 1}: {rname_w(k)}" for k in REGS_W], fontsize=6)
ax.set_xlim(-0.6, len(WF) - 0.4); ax.set_ylim(len(REGS_W) - 0.4, -0.6)
ax.grid(color="0.92", lw=0.5); ax.set_axisbelow(True)
cb = plt.colorbar(sc_, ax=ax, shrink=0.6); cb.set_label("Cliff's delta (6-OHDA higher →)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
for qv, lab in ((1, "q = 0.1"), (2, "q = 0.01"), (3, "q ≤ 0.001")):
    ax.scatter([], [], s=20 + 90 * qv, color="0.7", label=lab)
ax.legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(1.18, 1.0), title="size", title_fontsize=6)
plt.tight_layout()
save_fig(fig, OUTW, "G_features_bubble", dpi=600, pdf=True)

# H: top-5 features (mean |Cliff's delta| across regions), box/swarm per region, Mann-Whitney q
SG_BOX = MW.assign(q_BH=MW["MW_q_BH_all_tests"] if MW_FAMILY == "all" else MW["MW_q_BH_within_region"])
top5 = SG_BOX.groupby("feature")["cliffs_delta"].apply(lambda s: s.abs().mean()).sort_values(ascending=False).head(5).index.tolist()
print("top-5 features:", [WF[f] for f in top5])
fig, axes = plt.subplots(len(REGS_W), len(top5), figsize=(26 * MM * len(top5), 19 * MM * len(REGS_W)), squeeze=False)
for ri, k in enumerate(REGS_W):
    d = FAW[FAW["region"] == k]
    for ci, f in enumerate(top5):
        ax = axes[ri, ci]
        if d["cond"].nunique() < 2:
            ax.axis("off"); continue
        sns.boxplot(data=d, x="cond", y=f, order=list(COND_COL), hue="cond", hue_order=list(COND_COL), palette=COND_COL,
                    showfliers=False, width=0.55, boxprops=dict(alpha=0.45), linewidth=0.5, legend=False, ax=ax)
        sns.swarmplot(data=d, x="cond", y=f, order=list(COND_COL), hue="cond", hue_order=list(COND_COL), palette=COND_COL,
                      size=1.6, edgecolor="k", linewidth=0.2, legend=False, ax=ax)
        s_ = SG_BOX[(SG_BOX["region_k"] == k) & (SG_BOX["feature"] == f)].dropna(subset=["q_BH"])
        if len(s_):
            q = s_["q_BH"].iloc[0]
            ax.set_title(f"{stars(q)}  δ={s_['cliffs_delta'].iloc[0]:+.2f}", fontsize=5.5, pad=1, color="k" if q < 0.05 else "0.5")
        ax.set_xlabel(""); ax.set_ylabel(f"{k + 1}: {rname_w(k)}" if ci == 0 else "", fontsize=5.5, color=COL_W[k])
        ax.set_xticks([0, 1]); ax.set_xticklabels(["C", "6-OHDA"] if ri == len(REGS_W) - 1 else ["", ""], fontsize=5)
        ax.tick_params(labelsize=4.5); ax.spines[["top", "right"]].set_visible(False)
        if ri == 0:
            ax.text(0.5, 1.35, WF[f], transform=ax.transAxes, ha="center", fontsize=6)
plt.tight_layout(w_pad=0.25, h_pad=0.3)
save_fig(fig, OUTW, "H_top5_features_boxswarm", dpi=600, pdf=True)

# =============================================================================
# F. skeletons per region x condition (attached, normal-length skeletons; evenly across animals)
# =============================================================================
if DO_SKELETONS:
    N_PICK_W, N_SK_F, ALPHA_SK_F, MAD_K, JUNC_TOL = 600, 80, 0.1, 4.0, 0.2       # JUNC_TOL in tail lengths
    path_of = meta2.set_index("Experiment_ID")["path"]
    lref_of = meta2.set_index("Experiment_ID")["Lref_px"]
    W["frame_c"] = W["WindowStart"] + 15
    picks = []
    for (c, k), d in W[W["region"].notna()].groupby(["cond", "region"]):
        per = int(np.ceil(N_PICK_W / d["Experiment_ID"].nunique()))
        for e, g in d.groupby("Experiment_ID"):
            g = g.sort_values("frame_c")
            picks.append(g.iloc[np.linspace(0, len(g) - 1, min(per, len(g))).round().astype(int)][["Experiment_ID", "cond", "region", "frame_c"]])
    picks = pd.concat(picks, ignore_index=True)
    SKW = {}
    for i, (e, g) in enumerate(picks.groupby("Experiment_ID")):
        P, _ = clean_frames(path_of[e])
        lref = lref_of.get(e, lref_of.median())
        for row in g.itertuples():
            tr, tl = normalised(P, int(row.frame_c), lref, 1.0, TRUNK_DRAW), normalised(P, int(row.frame_c), lref, 1.0, TAIL_DRAW)
            if tr is not None and tl is not None:
                SKW.setdefault((row.cond, int(row.region)), []).append((e, tr, tl))
        if (i + 1) % 20 == 0:
            print(f"  skeletons: {i + 1}/{picks['Experiment_ID'].nunique()} animals")

    def arc_len(xy):
        return float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))

    def mad_ok(s):
        m, mad = s.median(), np.median(np.abs(s - s.median())) * 1.4826 + 1e-9
        return (s - m).abs() <= MAD_K * mad

    G = pd.DataFrame([dict(cond=c, region=k, i=i, jx=tl[0, 0], jy=tl[0, 1], tail_len=arc_len(tl), trunk_len=float(np.hypot(*(tr[2] - tr[0]))))
                      for (c, k), items in SKW.items() for i, (e, tr, tl) in enumerate(items)])
    G["junc_dev"] = np.hypot(G["jx"] - G.groupby("region")["jx"].transform("median"), G["jy"] - G.groupby("region")["jy"].transform("median"))
    G["keep"] = ((G["junc_dev"] <= JUNC_TOL) & G.groupby("region")["tail_len"].transform(mad_ok)
                 & G.groupby("region")["trunk_len"].transform(mad_ok))
    print(G.groupby("region")["keep"].apply(lambda s: f"{s.sum()} of {len(s)} kept ({s.mean():.0%})").to_string())
    keep_idx = {(c, k): set(g.loc[g["keep"], "i"]) for (c, k), g in G.groupby(["cond", "region"])}

    PANEL, crow = {}, []
    for (c, k), items in SKW.items():
        ok_items = [items[i] for i in sorted(keep_idx.get((c, k), set()))]
        if len(ok_items) < 10:
            continue
        by_an = {}
        for e, tr, tl in ok_items:
            by_an.setdefault(e, []).append((tr, tl))
        per = int(np.ceil(N_SK_F / len(by_an)))
        chosen = []
        for e, lst in by_an.items():
            chosen += [lst[j] for j in np.linspace(0, len(lst) - 1, min(per, len(lst))).round().astype(int)]
        PANEL[(c, k)] = [chosen[j] for j in np.linspace(0, len(chosen) - 1, min(N_SK_F, len(chosen))).round().astype(int)]
        crow.append(dict(cond=c, region=k + 1, name=rname_w(k), animals=len(by_an), skeletons_drawn=len(PANEL[(c, k)])))

    pts = np.vstack([np.vstack([tl, tr[:4]]) for v in PANEL.values() for tr, tl in v])
    lo_xy, hi_xy = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
    pad_xy = 0.08 * (hi_xy - lo_xy)
    XL, YL = (lo_xy[0] - pad_xy[0], hi_xy[0] + pad_xy[0]), (lo_xy[1] - pad_xy[1], hi_xy[1] + pad_xy[1])
    fig, axes = plt.subplots(2, len(REGS_W), figsize=(20 * MM * len(REGS_W), 48 * MM), sharex=True, sharey=True, squeeze=False)
    for ri, c in enumerate(("control", "6-OHDA")):
        for ci, k in enumerate(REGS_W):
            ax = axes[ri, ci]
            for tr, tl in PANEL.get((c, k), []):
                ax.plot(tr[:, 0], tr[:, 1], color="0.55", lw=0.35, alpha=ALPHA_SK_F * 0.8)
                ax.plot(tl[:, 0], tl[:, 1], color=COND_COL[c], lw=0.45, alpha=ALPHA_SK_F, solid_capstyle="round")
            ax.set_xlim(XL); ax.set_ylim(YL); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if ri == 0:
                ax.set_title(rname_w(k), fontsize=6, color=COL_W[k], pad=2)
            if ci == 0:
                ax.set_ylabel(c, color=COND_COL[c], fontsize=6.5)
    sb_x0, sb_y = XL[0] + 0.05 * (XL[1] - XL[0]), YL[0] + 0.04 * (YL[1] - YL[0])
    axes[-1, 0].plot([sb_x0, sb_x0 + 0.5], [sb_y, sb_y], color="k", lw=0.8)
    axes[-1, 0].text(sb_x0 + 0.25, sb_y + 0.03 * (YL[1] - YL[0]), "0.5 L", ha="center", va="bottom", fontsize=5)
    plt.tight_layout(w_pad=0.1, h_pad=0.2)
    save_fig(fig, OUTW, "F_region_skeletons", dpi=600, pdf=True)
    pd.DataFrame(crow).to_csv(f"{OUTW}/F_region_skeletons_counts.csv", index=False)
print(f"\nsaved → {OUTW}")
