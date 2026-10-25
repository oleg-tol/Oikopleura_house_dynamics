#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02 — Skeleton cleaning + 1 s window features (30 fps, 0.5 s stride).

Outputs (OUT/02_features/):
    02_windows_long.csv              one row per animal x window
    02_metadata_with_qc.csv          metadata + Lref + window yield (+ implausible-Lref exclusions)
    wide/<feature>_wide.csv          Time_XXXXX columns per feature
    02_animal_phase_summary.csv      per-animal medians per phase
    02_stats_mwu_bh.csv              Mann-Whitney + BH (free / during expansion / inside house)
    02_stats_B2_without_suspected_housed.csv
    figures/                         QC yield, box/swarm per comparison
"""
import seaborn as sns
from common import *

OUT2, WIDE, FIG = f"{OUT}/02_features", f"{OUT}/02_features/wide", f"{OUT}/02_features/figures"
for d in (OUT2, WIDE, FIG):
    os.makedirs(d, exist_ok=True)
meta = pd.read_csv(f"{OUT}/01_metadata_animals.csv", dtype={"Experiment_ID": str})

# =============================================================================
# 1. run all animals
# =============================================================================
wins, qc_rows = [], []
for k, r in meta.iterrows():
    P, info = clean_frames(r["path"])
    w, Lref = window_features(P, phase_vector(r, info["T"]))
    w.insert(0, "Experiment_ID", r["Experiment_ID"])
    wins.append(w)
    qc_rows.append(dict(Experiment_ID=r["Experiment_ID"], Lref_px=Lref, frac_jump=info["frac_jump"],
                        win_kin_ok=w["speed_med"].notna().mean(), win_post_ok=w["base_abs"].notna().mean()))
    print(f"  [{k + 1:3d}/{len(meta)}] {r['Experiment_ID']} {r['group']:2s} {r['batch']:5s} "
          f"kin={qc_rows[-1]['win_kin_ok']:.2f} posture={qc_rows[-1]['win_post_ok']:.2f} Lref={Lref:.1f}px")
win = pd.concat(wins, ignore_index=True)
meta2 = meta.merge(pd.DataFrame(qc_rows), on="Experiment_ID")
meta2["exclude_reason"] = meta2["exclude_reason"].fillna("")
lref_med = meta2.groupby("batch")["Lref_px"].transform("median")
bad_lref = (meta2["Lref_px"] < 0.5 * lref_med) | (meta2["Lref_px"] > 2 * lref_med)
meta2.loc[bad_lref, "exclude_reason"] = (meta2.loc[bad_lref, "exclude_reason"] + "; implausible tail length (Lref)").str.strip("; ")
print("excluded for implausible Lref:", meta2.loc[bad_lref, ["Experiment_ID", "group", "Lref_px"]].round(1).values.tolist())
KEYS = ["Experiment_ID", "cond", "group", "batch", "snapshot", "track_ok", "exclude_reason"]
win = win.merge(meta2[KEYS], on="Experiment_ID")

# =============================================================================
# 2. save: long + wide
# =============================================================================
win.to_csv(f"{OUT2}/02_windows_long.csv", index=False)
meta2.to_csv(f"{OUT2}/02_metadata_with_qc.csv", index=False)
for f in list(FEATS) + ["phase"]:
    wide = win.pivot(index="Experiment_ID", columns="WindowStart", values=f)
    wide.columns = [f"Time_{c:05d}" for c in wide.columns]
    meta2[KEYS].merge(wide, left_on="Experiment_ID", right_index=True).to_csv(f"{WIDE}/{f}_wide.csv", index=False)
print(f"\nsaved {len(win)} windows; wide tables → {WIDE}")

# =============================================================================
# 3. QC plot
# =============================================================================
order_all = ["C+", "C-", "C?", "Ch", "O+", "O-", "O?", "Oh", "R"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, (col, lab) in zip(axes, [("win_kin_ok", "windows with body kinematics"),
                                 ("win_post_ok", "windows with tail posture"), ("Lref_px", "proximal tail length (px)")]):
    sns.stripplot(data=meta2, x="group", y=col, order=order_all, hue="batch", dodge=False, size=4, ax=ax)
    ax.set_title(lab, fontsize=10); ax.set_xlabel("")
    if col != "Lref_px":
        ax.set_ylim(0, 1.02)
    if ax is not axes[-1] and ax.get_legend():
        ax.get_legend().remove()
plt.tight_layout()
save_fig(fig, FIG, "02a_qc_yield", dpi=200)
print(meta2.groupby("group")[["win_kin_ok", "win_post_ok", "frac_jump", "Lref_px"]].median().reindex(order_all).round(2).to_string())

# =============================================================================
# 4. per-animal summaries per phase
# =============================================================================
use = win[(win["exclude_reason"].fillna("") == "") & (win["group"] != "R")]
gk = ["Experiment_ID", "group", "batch", "phase"]
summ = use.groupby(gk)[list(FEATS)].median()
summ = summ.join(use.groupby(gk)["speed_med"].count().rename("n_kin")).join(use.groupby(gk)["base_abs"].count().rename("n_post")).reset_index()
summ["still"] = use.groupby(gk)["still"].mean().to_numpy()
need = summ["phase"].map(MIN_WIN_PHASE).fillna(20)
for f, (_, kind, _) in FEATS.items():
    summ.loc[(summ["n_kin"] if kind == "kin" else summ["n_post"]) < need, f] = np.nan
summ.to_csv(f"{OUT2}/02_animal_phase_summary.csv", index=False)

# =============================================================================
# 5. stats + box/swarm
# =============================================================================
def compare(d, pairs, tag):
    rows = []
    for a, b in pairs:
        for f in FEATS:
            x, y = d.loc[d["group"] == a, f].dropna(), d.loc[d["group"] == b, f].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            U, p = mannwhitneyu(x, y, alternative="two-sided")
            rows.append(dict(comparison=tag, A=a, B=b, feature=f, nA=len(x), nB=len(y), median_A=x.median(),
                             median_B=y.median(), r_rb_B_higher=1 - 2 * U / (len(x) * len(y)), p=p))
    st = pd.DataFrame(rows)
    st["q_BH"] = st.groupby(["A", "B"])["p"].transform(bh)            # BH across features, per pair
    return st

def box_swarm(d, order, pairs, st, title, fname):
    n = len(FEATS)
    fig, axes = plt.subplots(int(np.ceil(n / 4)), 4, figsize=(4.2 * 4, 3.4 * int(np.ceil(n / 4))))
    for ax, (f, (lab, _, logy)) in zip(axes.ravel(), FEATS.items()):
        dd = d[d["group"].isin(order)].dropna(subset=[f])
        sns.boxplot(data=dd, x="group", y=f, order=order, hue="group", hue_order=order, palette=GROUP_COL,
                    showfliers=False, width=0.55, boxprops=dict(alpha=0.45), legend=False, ax=ax)
        sns.swarmplot(data=dd, x="group", y=f, order=order, hue="group", hue_order=order, palette=GROUP_COL,
                      size=3.2, edgecolor="k", linewidth=0.3, legend=False, ax=ax)
        if logy and (dd[f] > 0).all() and len(dd):
            ax.set_yscale("log")
        ax.set_title(lab, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([f"{g}\nn={int((dd['group'] == g).sum())}" for g in order], fontsize=8)
        y_top = ax.get_ylim()[1]
        for k_, (a, b) in enumerate(pairs):
            s = st[(st["A"] == a) & (st["B"] == b) & (st["feature"] == f)]
            if s.empty:
                continue
            i, j = order.index(a), order.index(b)
            y = y_top * (1.02 + 0.08 * k_) if ax.get_yscale() == "linear" else y_top * (1.15 ** (k_ + 1))
            ax.plot([i, i, j, j], [y * 0.98, y, y, y * 0.98], color="k", lw=0.8)
            q = float(s["q_BH"].iloc[0])
            ax.text((i + j) / 2, y, stars(q), ha="center", va="bottom", fontsize=8 if q < 0.05 else 7,
                    color="k" if q < 0.05 else "0.5")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{title}  —  animal medians; Mann-Whitney, BH across features per pair", fontsize=10)
    plt.tight_layout()
    save_fig(fig, FIG, fname, dpi=200)

all_st = []
for i, (tag, ph_of, order, pairs) in enumerate(SETS):
    d = summ[summ.apply(lambda r: ph_of.get(r["group"]) == r["phase"], axis=1)]
    st = compare(d, pairs, tag)
    all_st.append(st)
    box_swarm(d, order, pairs, st, tag.replace("_", " "), f"02{chr(98 + i)}_{tag}")
    sig = st[st["q_BH"] < 0.05].sort_values("q_BH")
    print(f"\n{tag}: animals per group {d.groupby('group')['Experiment_ID'].nunique().reindex(order).to_dict()}")
    print(sig[["A", "B", "feature", "median_A", "median_B", "r_rb_B_higher", "q_BH"]].round(3).to_string(index=False)
          if len(sig) else "  nothing significant after BH")
pd.concat(all_st, ignore_index=True).to_csv(f"{OUT2}/02_stats_mwu_bh.csv", index=False)

# =============================================================================
# 6. check: B1 vs B2 side by side; B2 without suspected-housed controls
# =============================================================================
st = pd.concat(all_st, ignore_index=True)
cols = ["median_A", "median_B", "r_rb_B_higher", "p", "q_BH"]
fb = st[st["comparison"] == "free_behaviour"]
side = fb[fb["A"] == "C-"].set_index("feature")[cols].add_suffix("_B1").join(
       fb[fb["A"] == "C?"].set_index("feature")[cols].add_suffix("_B2"))
side["same_direction"] = np.sign(side["r_rb_B_higher_B1"]) == np.sign(side["r_rb_B_higher_B2"])
print(f"\nfree behaviour, same direction in both batches: {int(side['same_direction'].sum())}/{len(side)} features")
d = summ[(summ["phase"] == "free") & summ["group"].isin(["C?", "O?"]) & ~summ["Experiment_ID"].isin(SUSPECT_HOUSED)]
st_s = compare(d, [("C?", "O?")], "free_B2_without_suspected_housed")
print(st_s.set_index("feature")[cols].round(3).to_string())
st_s.to_csv(f"{OUT2}/02_stats_B2_without_suspected_housed.csv", index=False)
print(f"\nstats → {OUT2} | figures → {FIG}")
