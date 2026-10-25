#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04 — What precedes house expansion: onset-aligned curves, look-back prediction, still-bout outcomes
and conversion, execution of the expansion, stillness panels.

Outputs:
    OUT/04_precursor/  04a_onset_aligned_unit_curves.csv, 04b_lag_prediction.csv, 04c_*.csv, 04d_execution_stats.csv,
                       04_check_pre_onset_per_builder.csv, figures/
    OUT/fig_panels/    P_stillness_before_onset, P_stillness_paired (+ csv)
"""
import seaborn as sns
from scipy.stats import fisher_exact, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import roc_auc_score
from common import *

OUT4, FIG4, FIGP = f"{OUT}/04_precursor", f"{OUT}/04_precursor/figures", f"{OUT}/fig_panels"
for d in (FIG4, FIGP):
    os.makedirs(d, exist_ok=True)

ALIGN, BIN = (-60.0, 30.0), 5.0                 # onset-aligned range and bin (s)
LAGS, GAP, PERM = [5, 10, 20, 30], 2.0, 200     # look-back windows, gap before onset, permutations
AFTER_S = 5.0                                   # outcome of a still bout = next 5 s
KIN = ["still_frac", "speed_med", "omega_mean"]
FREE_OF = {("B1", "control"): ["C-"], ("B2", "control"): ["C?"], ("B1", "6-OHDA"): ["O-"], ("B2", "6-OHDA"): ["O?"]}
GM_K, MODES = 4, [f"M{k}" for k in range(4)]
MODE_COL = ["#4D4D4D", "#1B9E77", "#7570B3", "#D95F02"]

meta2 = load_meta2()
win = load_windows()
okm = valid_animals(meta2)
by_id = {e: g.reset_index(drop=True) for e, g in win[win["Experiment_ID"].isin(okm["Experiment_ID"])].groupby("Experiment_ID")}
builders = okm[okm["group"].isin(["C+", "O+"])].copy()
builders["t_on"] = builders["he_start"] / FPS

def free_ids(batch, cond):
    """Non-builders of the same batch and condition (B2 controls: suspected-housed removed)."""
    m = okm["group"].isin(FREE_OF.get((batch, cond), [])) & ~okm["Experiment_ID"].isin(SUSPECT_HOUSED)
    return okm.loc[m, "Experiment_ID"].tolist()

print("builders per batch x condition:", builders.groupby(["batch", "cond"]).size().to_dict())

# =============================================================================
# 04a. onset-aligned curves (builders vs time-matched non-builders)
# =============================================================================
parts = []
for _, r in builders.iterrows():
    for e, role in [(r["Experiment_ID"], "builder")] + [(x, "matched") for x in free_ids(r["batch"], r["cond"])]:
        g = by_id[e]
        d = g[(g["t_s"] >= r["t_on"] + ALIGN[0]) & (g["t_s"] < r["t_on"] + ALIGN[1])]
        if role == "matched":
            d = d[d["phase"] == "free"]
        if len(d):
            parts.append(d[["t_s", "still", "mode", "junc_abs"]].assign(
                t_rel=d["t_s"] - r["t_on"], unit=e, role=role, batch=r["batch"], cond=r["cond"]))
al = pd.concat(parts, ignore_index=True)
for k in range(GM_K):
    al[MODES[k]] = (al["mode"] == k).astype(float).where(al["mode"].notna())
al["bin"] = np.floor(al["t_rel"] / BIN) * BIN
CURVE_F = {"still": "still (fraction)", "M0": "M0 rest", "M3": "M3 fast/expansion", "M2": "M2 tail beating",
           "junc_abs": "junction angle (deg)"}
uc = al.groupby(["batch", "cond", "role", "unit", "bin"])[list(CURVE_F)].mean().reset_index()   # one curve per animal
uc.to_csv(f"{OUT4}/04a_onset_aligned_unit_curves.csv", index=False)

keys = [k for k in [("B1", "control"), ("B2", "control"), ("B2026", "control"), ("B1", "6-OHDA"), ("B2026", "6-OHDA")]
        if ((uc["batch"] == k[0]) & (uc["cond"] == k[1]) & (uc["role"] == "builder")).any()]
fig, axes = plt.subplots(len(keys), len(CURVE_F), figsize=(3.3 * len(CURVE_F), 2.5 * len(keys)), sharex=True, squeeze=False)
for ri, (b, c) in enumerate(keys):
    for ci, (f, lab) in enumerate(CURVE_F.items()):
        ax = axes[ri, ci]
        for role, ls in (("builder", "-"), ("matched", "--")):
            d = uc[(uc["batch"] == b) & (uc["cond"] == c) & (uc["role"] == role)]
            if d.empty:
                continue
            s = d.groupby("bin")[f].agg(["mean", "sem"])
            x = s.index + BIN / 2
            col = COND_COL[c] if role == "builder" else "0.45"
            ax.plot(x, s["mean"], ls, color=col, lw=1.6, label=f"{role} (n={d['unit'].nunique()})")
            ax.fill_between(x, s["mean"] - s["sem"], s["mean"] + s["sem"], color=col, alpha=0.18, lw=0)
        ax.axvline(0, color="k", lw=0.8)
        if f != "junc_abs":
            ax.set_ylim(0, 1)
        if ri == 0:
            ax.set_title(lab, fontsize=9)
        if ci == 0:
            ax.set_ylabel(f"{c}\n{b}", fontsize=9); ax.legend(fontsize=6, frameon=False)
for ax in axes[-1]:
    ax.set_xlabel("time from expansion onset (s)")
plt.tight_layout()
save_fig(fig, FIG4, "04a_onset_aligned_curves", dpi=200)

# =============================================================================
# 04b. expansion predicted from the last L s (leave-one-animal-out logistic regression, permutation null)
# =============================================================================
def lag_rows(L, batch, cond):
    rows = []
    for _, r in builders[(builders["batch"] == batch) & (builders["cond"] == cond)].iterrows():
        if r["t_on"] < L + GAP:
            continue
        for e, y in [(r["Experiment_ID"], 1)] + [(x, 0) for x in free_ids(batch, cond)]:
            g = by_id[e]
            d = g[(g["t_s"] >= r["t_on"] - L - GAP) & (g["t_s"] < r["t_on"] - GAP)]
            if y == 0:
                d = d[d["phase"] == "free"]
            sp = d["speed_med"].dropna()
            if len(sp) < max(2, 0.5 * L / DT):
                continue
            rows.append(dict(onset=r["Experiment_ID"], animal=e, y=y, still_frac=float((sp < STILL).mean()),
                             speed_med=sp.mean(), omega_mean=d["omega_mean"].mean()))
    return pd.DataFrame(rows)

def lr_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced", max_iter=5000))

def cv_auc(D, y):
    p = cross_val_predict(lr_model(), D[KIN].to_numpy(), y, groups=D["animal"].to_numpy(),
                          cv=LeaveOneGroupOut(), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)

rng4 = np.random.default_rng(4)
def perm_null(D):
    out = []
    for _ in range(PERM):
        yp = np.zeros(len(D), int)
        for _, ix in D.groupby("onset").groups.items():
            yp[rng4.choice(np.asarray(ix))] = 1
        out.append(cv_auc(D, yp))
    return np.array(out)

CMPS = {"control B1": [("B1", "control")], "control B2": [("B2", "control")],
        "control B1+B2": [("B1", "control"), ("B2", "control")], "6-OHDA B1": [("B1", "6-OHDA")]}
res, coefs = [], {}
for L in LAGS:
    tabs = {name: pd.concat([lag_rows(L, b, c) for b, c in ks], ignore_index=True).dropna(subset=KIN) for name, ks in CMPS.items()}
    for name, D in tabs.items():
        if D.empty or D["y"].sum() < 4:
            res.append(dict(L=L, comparison=name, n_pos=int(D["y"].sum()) if len(D) else 0)); continue
        D = D.reset_index(drop=True)
        y = D["y"].to_numpy()
        auc, null = cv_auc(D, y), perm_null(D)
        row = dict(L=L, comparison=name, n_pos=int(y.sum()), n_neg=int((1 - y).sum()), auc=auc,
                   p_perm=(np.sum(null >= auc) + 1) / (PERM + 1), null_lo=np.quantile(null, 0.025), null_hi=np.quantile(null, 0.975))
        if name == "control B1+B2":
            m = lr_model().fit(D[KIN].to_numpy(), y)
            coefs[L] = pd.Series(m[-1].coef_[0], index=KIN)
            T = tabs["6-OHDA B1"]
            if len(T) and T["y"].sum() >= 3:
                row["auc_on_6OHDA_B1"] = roc_auc_score(T["y"], m.predict_proba(T[KIN].to_numpy())[:, 1])
        res.append(row)
res = pd.DataFrame(res)
res.to_csv(f"{OUT4}/04b_lag_prediction.csv", index=False)
print("\n04b — expansion predicted from the last L s:"); print(res.round(3).to_string(index=False))
print(pd.DataFrame(coefs).round(2).to_string())

fig, axes = plt.subplots(1, len(CMPS), figsize=(4.2 * len(CMPS), 3.6), sharey=True)
for ax, name in zip(axes, CMPS):
    d = res[(res["comparison"] == name) & res["auc"].notna()] if "auc" in res else pd.DataFrame()
    if len(d):
        ax.fill_between(d["L"], d["null_lo"], d["null_hi"], color="0.85", label="permutation 95%")
        ax.plot(d["L"], d["auc"], "o-", color=COND_COL["control" if name.startswith("control") else "6-OHDA"],
                label="leave-one-animal-out AUC")
        for _, r in d.iterrows():
            ax.text(r["L"], r["auc"] + 0.03, f"n+={int(r['n_pos'])}", ha="center", fontsize=7)
        if name == "6-OHDA B1" and "auc_on_6OHDA_B1" in res:
            t = res[(res["comparison"] == "control B1+B2") & res["auc_on_6OHDA_B1"].notna()]
            ax.plot(t["L"], t["auc_on_6OHDA_B1"], "s--", color=COND_COL["control"], label="control model → 6-OHDA")
    ax.axhline(0.5, color="k", lw=0.6, ls=":")
    ax.set_title(name, fontsize=10); ax.set_xlabel("look-back L (s)"); ax.set_xticks(LAGS); ax.set_ylim(0.2, 1.05)
axes[0].set_ylabel("AUC (builder vs time-matched non-builder)")
axes[-1].legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout()
save_fig(fig, FIG4, "04b_lag_prediction_auc", dpi=200)

# =============================================================================
# 04c. still bouts (opportunities): what follows them, and conversion to expansion
# =============================================================================
def outcome(gp, j, t_on, g_full):
    t_last = gp["t_s"].iloc[j]
    t_end = t_last + WIN / FPS / 2
    if t_on is not None and t_on - AFTER_S <= t_end <= t_on + 1:
        return "expansion"
    nxt = g_full[(g_full["t_s"] > t_last) & (g_full["t_s"] <= t_last + AFTER_S)]
    if (nxt["phase"] == "HE").any():
        return "expansion"
    m = nxt["mode"].dropna()
    if len(nxt) < 3 or len(m) < 0.5 * len(nxt):
        return "unknown"
    for k, lab in ((3, "fast swim"), (2, "tail beating"), (1, "slow")):
        if (m == k).any():
            return lab
    return "rest"

OUTC = ["expansion", "fast swim", "tail beating", "slow", "rest", "unknown"]
OUTC_COL = {"expansion": "gold", "fast swim": MODE_COL[3], "tail beating": MODE_COL[2], "slow": MODE_COL[1],
            "rest": MODE_COL[0], "unknown": "0.85"}
ob_rows, opp_rows = [], []
for _, r in okm[okm["group"].isin(["C+", "O+", "C-", "O-", "C?", "O?"])].iterrows():
    e = r["Experiment_ID"]
    if e in SUSPECT_HOUSED:
        continue
    g = by_id[e]
    t_on = r["he_start"] / FPS if pd.notna(r["he_start"]) else None
    pre = g["phase"].isin(["free", "pre"]).to_numpy()
    gp = g[pre].reset_index(drop=True)                          # still bouts are cut at expansion onset
    runs = still_runs(gp) if len(gp) else []
    for i, j, dur in runs:
        ob_rows.append(dict(Experiment_ID=e, group=r["group"], batch=r["batch"], cond=r["cond"], dur_s=dur,
                            outcome=outcome(gp, j, t_on, g)))
    minutes = g.loc[pre, "speed_med"].notna().sum() * DT / 60
    opp_rows.append(dict(Experiment_ID=e, group=r["group"], batch=r["batch"], cond=r["cond"],
                         n_opp=len(runs), minutes=minutes, opp_per10=len(runs) / minutes * 10 if minutes > 0 else np.nan))
obo, opp = pd.DataFrame(ob_rows), pd.DataFrame(opp_rows)
obo.to_csv(f"{OUT4}/04c_still_bout_outcomes.csv", index=False)
opp.to_csv(f"{OUT4}/04c_opportunities_per_animal.csv", index=False)

conv = []
for b in ("B1", "B2"):
    for c, gp_ in (("control", "C+"), ("6-OHDA", "O+")):
        ids = okm[(okm["batch"] == b) & okm["group"].isin([gp_] + FREE_OF[(b, c)]) & ~okm["Experiment_ID"].isin(SUSPECT_HOUSED)]
        o = opp[opp["Experiment_ID"].isin(ids["Experiment_ID"])]
        conv.append(dict(batch=b, cond=c, expanded=int((ids["group"] == gp_).sum()), n_free=len(ids),
                         opp_per10_median=o["opp_per10"].median(), opportunities=int(o["n_opp"].sum())))
conv = pd.DataFrame(conv)
conv["frac_expanded"] = conv["expanded"] / conv["n_free"]
conv["expansions_per_100_opp"] = conv["expanded"] / conv["opportunities"].replace(0, np.nan) * 100
for b in ("B1", "B2"):
    cc = conv[conv["batch"] == b].set_index("cond")
    _, p = fisher_exact([[cc.loc["control", "expanded"], cc.loc["control", "n_free"] - cc.loc["control", "expanded"]],
                         [cc.loc["6-OHDA", "expanded"], cc.loc["6-OHDA", "n_free"] - cc.loc["6-OHDA", "expanded"]]])
    o = opp[(opp["batch"] == b) & ~opp["group"].str.endswith("+")]
    po = mannwhitneyu(o.loc[o["cond"] == "control", "opp_per10"].dropna(), o.loc[o["cond"] == "6-OHDA", "opp_per10"].dropna())[1]
    conv.loc[conv["batch"] == b, "fisher_p_expanded"] = p
    conv.loc[conv["batch"] == b, "mwu_p_opportunities_nonbuilders"] = po
conv.to_csv(f"{OUT4}/04c_conversion_by_batch.csv", index=False)
print("\n04c — opportunity vs conversion:"); print(conv.round(3).to_string(index=False))

pa = obo.groupby(["Experiment_ID", "group", "batch", "cond"])["outcome"].value_counts(normalize=True).unstack(fill_value=0)
pa = pa.reindex(columns=OUTC, fill_value=0).reset_index()
pa["n_bouts"] = obo.groupby("Experiment_ID").size().reindex(pa["Experiment_ID"]).to_numpy()
pa = pa[pa["n_bouts"] >= 3]
st_o = [compare_cols(pa, [(gc, go)], OUTC[1:5], f"still-bout outcome {b}") for b, (gc, go) in (("B1", ("C-", "O-")), ("B2", ("C?", "O?")))]
pd.concat(st_o).to_csv(f"{OUT4}/04c_stats_still_bout_outcomes.csv", index=False)

order_o = ["C+", "C-", "C?", "O+", "O-", "O?"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
ax = axes[0]
means = pa.groupby("group")[OUTC].mean().reindex(order_o)
bottom = np.zeros(len(order_o))
for o in OUTC:
    v = means[o].fillna(0).to_numpy()
    ax.bar(range(len(order_o)), v, bottom=bottom, color=OUTC_COL[o], edgecolor="white", label=o)
    bottom += v
ax.set_xticks(range(len(order_o))); ax.set_xticklabels([f"{g}\nn={int((pa['group'] == g).sum())}" for g in order_o])
ax.set_ylabel("mean fraction of still bouts (≥ 5 s)"); ax.set_title("What follows a still bout", fontsize=10)
ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
ax = axes[1]
for i, (_, r) in enumerate(conv.iterrows()):
    ax.bar(i, r["frac_expanded"], color=COND_COL[r["cond"]], alpha=0.75, width=0.7)
    ax.text(i, r["frac_expanded"] + 0.02, f"{r['expanded']}/{r['n_free']}", ha="center", fontsize=8)
for k, b in enumerate(("B1", "B2")):
    ax.text(2 * k + 0.5, 0.92, f"Fisher p={conv.loc[conv['batch'] == b, 'fisher_p_expanded'].iloc[0]:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(conv))); ax.set_xticklabels([f"{r.batch}\n{r.cond}" for r in conv.itertuples()], fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("fraction of free animals that expanded"); ax.set_title("Conversion", fontsize=10)
plt.tight_layout()
save_fig(fig, FIG4, "04c_still_bouts_and_conversion", dpi=200)

# =============================================================================
# 04d. execution of the expansion (annotations)
# =============================================================================
bld = okm[okm["group"].isin(["C+", "O+"])].copy()
bld["abnormal start"] = bld["n_abnormal_start"] > 0
bld["failed insertion"] = bld["n_failed_insertion"] > 0
bld["any abnormal event"] = bld["abnormal start"] | bld["failed insertion"] | (bld["n_twitching"] > 0) | (bld["dragging"] == True)
bld["escaped"] = bld["escape"].notna()
bld["HE duration (s)"] = (bld["he_end"] - bld["he_start"]) / FPS
SHOW_M = ["any abnormal event", "abnormal start", "failed insertion", "escaped"]
rows = []
for m in SHOW_M:
    k = bld.groupby("cond")[m].agg(["sum", "size"])
    kc, nc, ko, no = int(k.loc["control", "sum"]), int(k.loc["control", "size"]), int(k.loc["6-OHDA", "sum"]), int(k.loc["6-OHDA", "size"])
    rows.append(dict(measure=m, control=f"{kc}/{nc}", sixOHDA=f"{ko}/{no}", frac_c=kc / nc, frac_o=ko / no,
                     test="Fisher", p=fisher_exact([[kc, nc - kc], [ko, no - ko]])[1]))
x, y = (bld.loc[bld["cond"] == c, "HE duration (s)"] for c in ("control", "6-OHDA"))
rows.append(dict(measure="HE duration (s)", control=f"{x.median():.1f}", sixOHDA=f"{y.median():.1f}",
                 test="Mann-Whitney", p=mannwhitneyu(x, y, alternative="two-sided")[1]))
ann = pd.DataFrame(rows)
ann["q_BH"] = bh(ann["p"])
ann.to_csv(f"{OUT4}/04d_execution_stats.csv", index=False)
print("\n04d — execution of the expansion:"); print(ann.round(4).to_string(index=False))

LAB_M = {"any abnormal event": "any\nabnormal", "abnormal start": "abnormal\nstart", "failed insertion": "failed\ninsertion", "escaped": "escape"}
A_ = ann.set_index("measure")
fig, axes = plt.subplots(1, 2, figsize=(3.9, 2.2), gridspec_kw=dict(width_ratios=[3, 1.1]))
ax = axes[0]
xx = np.arange(len(SHOW_M))
for off, (c, fr_, col_txt) in zip((-0.19, 0.19), (("control", "frac_c", "control"), ("6-OHDA", "frac_o", "sixOHDA"))):
    ax.bar(xx + off, A_.loc[SHOW_M, fr_].to_numpy(float), width=0.36, color=COND_COL[c], alpha=0.85, label=c, lw=0)
    for xi, m in zip(xx, SHOW_M):
        ax.text(xi + off, A_.loc[m, fr_] + 0.02, A_.loc[m, col_txt], ha="center", va="bottom", fontsize=4.8, rotation=90)
for xi, m in zip(xx, SHOW_M):
    ax.text(xi, 0.97, stars(A_.loc[m, "q_BH"]), ha="center", fontsize=6)
ax.set_xticks(xx); ax.set_xticklabels([LAB_M[m] for m in SHOW_M], fontsize=5.5)
ax.set_ylim(0, 1.05); ax.set_ylabel("fraction of builders", fontsize=6.5); ax.tick_params(labelsize=6)
ax.legend(fontsize=5.5, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.92))
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
sns.boxplot(data=bld, x="group", y="HE duration (s)", order=["C+", "O+"], hue="group", hue_order=["C+", "O+"],
            palette=GROUP_COL, showfliers=False, width=0.55, boxprops=dict(alpha=0.5), linewidth=0.7, legend=False, ax=ax)
sns.swarmplot(data=bld, x="group", y="HE duration (s)", order=["C+", "O+"], hue="group", hue_order=["C+", "O+"],
              palette=GROUP_COL, size=2.4, edgecolor="k", linewidth=0.3, legend=False, ax=ax)
ax.set_title(stars(A_.loc["HE duration (s)", "q_BH"]), fontsize=6, pad=1)
ax.set_xlabel(""); ax.set_xticks([0, 1]); ax.set_xticklabels(["control", "6-OHDA"], fontsize=5.5)
ax.set_ylabel("expansion duration (s)", fontsize=6.5); ax.tick_params(labelsize=6)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(w_pad=0.6)
save_fig(fig, FIG4, "04d_execution_compact")

# =============================================================================
# 04-check. what precedes each onset (per builder)
# =============================================================================
diag = []
for _, r in builders.iterrows():
    g = by_id[r["Experiment_ID"]]
    pre = g[g["phase"] == "pre"].reset_index(drop=True)
    runs = still_runs(pre) if len(pre) else []
    last = runs[-1] if runs else None
    w7 = g[(g["t_s"] >= r["t_on"] - 7) & (g["t_s"] < r["t_on"] - 2)]
    diag.append(dict(Experiment_ID=r["Experiment_ID"], group=r["group"], batch=r["batch"], onset_s=round(r["t_on"], 1),
                     pre_tracked_s=pre["speed_med"].notna().sum() * DT, n_still_runs_5s=len(runs),
                     longest_still_s=max((d for *_, d in runs), default=0),
                     last_still_end_to_onset_s=(r["t_on"] - (pre["t_s"].iloc[last[1]] + 0.5)) if last else np.nan,
                     still_frac_7_to_2s=w7["still"].mean(),
                     mode_7_to_2s=w7["mode"].mode().iloc[0] if w7["mode"].notna().any() else np.nan,
                     annotated_abnormal_start=r["n_abnormal_start"] > 0))
diag = pd.DataFrame(diag).sort_values(["group", "batch", "onset_s"])
diag.to_csv(f"{OUT4}/04_check_pre_onset_per_builder.csv", index=False)

# =============================================================================
# FIG panel — stillness before onset vs time-matched non-builders
# =============================================================================
L_PANEL = 5
SP = pd.concat([lag_rows(L_PANEL, b, c).assign(cond=c, batch=b) for b, c in [("B1", "control"), ("B2", "control"), ("B1", "6-OHDA")]],
               ignore_index=True)
SPa = SP.groupby(["animal", "cond", "y"])["still_frac"].mean().reset_index()      # one value per animal
SPa["cat"] = np.where(SPa["y"] == 1, "builder\n(before onset)", "non-builder\n(same clock time)")
fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.4), sharey=True)
for ax, c in zip(axes, ("control", "6-OHDA")):
    d = SPa[SPa["cond"] == c]
    order_ = ["non-builder\n(same clock time)", "builder\n(before onset)"]
    pal = {order_[0]: "0.7", order_[1]: COND_COL[c]}
    sns.boxplot(data=d, x="cat", y="still_frac", order=order_, hue="cat", hue_order=order_, palette=pal, showfliers=False,
                width=0.55, boxprops=dict(alpha=0.5), linewidth=0.7, legend=False, ax=ax)
    sns.swarmplot(data=d, x="cat", y="still_frac", order=order_, hue="cat", hue_order=order_, palette=pal, size=2.6,
                  edgecolor="k", linewidth=0.3, legend=False, ax=ax)
    p = mannwhitneyu(d.loc[d["y"] == 0, "still_frac"], d.loc[d["y"] == 1, "still_frac"], alternative="two-sided")[1]
    ax.plot([0, 0, 1, 1], [1.04, 1.07, 1.07, 1.04], color="k", lw=0.6)
    ax.text(0.5, 1.08, f"{stars(p)}  p={p:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_title(c, fontsize=7, color=COND_COL[c]); ax.set_xlabel("")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"{o}\nn={int((d['cat'] == o).sum())}" for o in order_], fontsize=5.5)
    ax.tick_params(labelsize=6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(f"fraction still, {L_PANEL + 2}–2 s before", fontsize=6.5); axes[0].set_ylim(-0.03, 1.2)
plt.tight_layout(w_pad=0.5)
save_fig(fig, FIGP, "P_stillness_before_onset")

# =============================================================================
# FIG panel — stillness just before expansion vs each builder's own average (paired Wilcoxon)
# =============================================================================
PRE_A, PRE_B, EXCL_S, MIN_BASE_S = 7.0, 2.0, 10.0, 30.0
rows = []
for _, r in builders[~builders["Experiment_ID"].isin(SUSPECT_HOUSED)].iterrows():
    g = by_id[r["Experiment_ID"]]
    before = g[(g["t_s"] >= r["t_on"] - PRE_A) & (g["t_s"] < r["t_on"] - PRE_B)]["still"].dropna()
    base = g[g["phase"].isin(["pre", "after_escape"]) & ~g["t_s"].between(r["t_on"] - EXCL_S, r["t_on"])]["still"].dropna()
    if len(before) >= 4 and len(base) * DT >= MIN_BASE_S:
        rows.append(dict(Experiment_ID=r["Experiment_ID"], cond=r["cond"], batch=r["batch"],
                         average=base.mean(), before_expansion=before.mean(), base_s=len(base) * DT))
PS = pd.DataFrame(rows)
PS.to_csv(f"{FIGP}/P_stillness_paired.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(3.2, 2.4), sharey=True)
for ax, c in zip(axes, ("control", "6-OHDA")):
    d = PS[PS["cond"] == c]
    for _, r in d.iterrows():
        ax.plot([0, 1], [r["average"], r["before_expansion"]], color=COND_COL[c], lw=0.7, alpha=0.6)
    ax.scatter(np.zeros(len(d)), d["average"], s=10, color="0.6", edgecolor="k", linewidth=0.3, zorder=3)
    ax.scatter(np.ones(len(d)), d["before_expansion"], s=10, color=COND_COL[c], edgecolor="k", linewidth=0.3, zorder=3)
    ax.plot([-0.15, 0.15], [d["average"].median()] * 2, color="k", lw=1.5)
    ax.plot([0.85, 1.15], [d["before_expansion"].median()] * 2, color="k", lw=1.5)
    p = wilcoxon(d["before_expansion"], d["average"])[1] if len(d) >= 5 else np.nan
    ax.plot([0, 0, 1, 1], [1.04, 1.07, 1.07, 1.04], color="k", lw=0.6)
    ax.text(0.5, 1.08, f"{stars(p)}  p={p:.3f}" if np.isfinite(p) else "n/a", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["average", "before\nexpansion"], fontsize=6)
    ax.set_xlim(-0.4, 1.4); ax.set_title(f"{c} (n={len(d)})", fontsize=7, color=COND_COL[c])
    ax.tick_params(labelsize=6); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("probability of being still", fontsize=6.5); axes[0].set_ylim(-0.03, 1.2)
plt.tight_layout(w_pad=0.5)
save_fig(fig, FIGP, "P_stillness_paired")
print(f"\nsaved → {OUT4} and {FIGP}")
