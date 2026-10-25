#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03 — Movement modes: GMM (K=4) on log speed x log tail-base angular velocity, fitted on control windows;
choice of K; decoding of all windows (speed-only fallback for fast swimming), smoothing, bouts,
time budgets, transitions, rasters, ethograms.

Outputs:
    OUT/03_modes/      03_mode_occupancy_animal_phase.csv, 03_stats_mode_occupancy.csv,
                       03_select_K_and_features.csv, 03_gmm.pkl, figures/
    OUT/03b_decoding/  03b_windows_long_decoded.csv (input for steps 04–07), 03b_bouts.csv,
                       03b_time_budget_bouts_animal_phase.csv, stats, figures/ (+ ethograms/)
"""
import pickle
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GroupKFold
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from common import *

OUT3, FIG3 = f"{OUT}/03_modes", f"{OUT}/03_modes/figures"
OUT3b, FIGB = f"{OUT}/03b_decoding", f"{OUT}/03b_decoding/figures"
ETHO = f"{FIGB}/ethograms"
for d in (FIG3, ETHO):
    os.makedirs(d, exist_ok=True)

GM_K, GM_N_PER, GM_SEED = 4, 300, 0
MODES = [f"M{k}" for k in range(GM_K)]
MODE_COL = ["#4D4D4D", "#1B9E77", "#7570B3", "#D95F02"]

win = pd.read_csv(f"{OUT}/02_features/02_windows_long.csv", dtype={"Experiment_ID": str}, low_memory=False)
meta2 = load_meta2()

# =============================================================================
# 1. fit on control windows
# =============================================================================
valid = (win["exclude_reason"].fillna("") == "") & (win["group"] != "R")
win["ls"] = np.log10(win["speed_med"].where(win["speed_med"] > 0))
win["lv"] = np.log10(win["base_vel"].where(win["base_vel"] > 0))
pool = win[valid & (win["cond"] == "control") & win["ls"].notna() & win["lv"].notna()]
fit = pool.groupby("Experiment_ID", group_keys=False).apply(lambda g: g.sample(min(len(g), GM_N_PER), random_state=GM_SEED))
Xf = fit[["ls", "lv"]].to_numpy()
bic = {k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=GM_SEED).fit(Xf).bic(Xf) for k in range(2, 8)}
print("BIC by number of modes:", {k: round(v) for k, v in bic.items()})
gm = GaussianMixture(GM_K, covariance_type="full", n_init=10, random_state=GM_SEED).fit(Xf)
order = np.lexsort((gm.means_[:, 1], gm.means_[:, 0]))          # M0 = slowest … M3 = fastest
relab = {old: new for new, old in enumerate(order)}
with open(f"{OUT3}/03_gmm.pkl", "wb") as fh:
    pickle.dump(dict(gm=gm, relab=relab), fh)
has = win["ls"].notna() & win["lv"].notna()
win["mode"] = np.nan
win.loc[has, "mode"] = pd.Series(gm.predict(win.loc[has, ["ls", "lv"]].to_numpy()), index=win.index[has]).map(relab)
print(f"windows with a mode: {has.mean():.1%} (needs body speed + tail posture)")

# =============================================================================
# 2. mode profiles
# =============================================================================
cw = win[valid & win["mode"].notna() & (win["cond"] == "control")]
prof = cw.groupby("mode")[["speed_med", "base_vel", "junc_abs", "straight", "omega_mean", "fwd_mean"]].median()
prof["share_of_control_windows"] = cw["mode"].value_counts(normalize=True).sort_index()
for ph in ["free", "pre", "HE", "in_house", "housed"]:
    prof[f"frac_in_{ph}"] = cw[cw["phase"] == ph]["mode"].value_counts(normalize=True).reindex(prof.index).fillna(0)
prof.index = MODES
print("\nmode profile (control windows):"); print(prof.round(2).T.to_string())
prof.to_csv(f"{OUT3}/03_mode_profiles.csv")

# =============================================================================
# 3. occupancy per animal x phase (undecoded GMM modes)
# =============================================================================
use = win[valid & win["mode"].notna()]
gcols = ["Experiment_ID", "group", "batch", "phase"]
occ = use.groupby(gcols)["mode"].value_counts(normalize=True).unstack(fill_value=0)
occ.columns = [f"M{int(c)}" for c in occ.columns]
occ = occ.reindex(columns=MODES, fill_value=0)
occ["n_mode_win"] = use.groupby(gcols).size()
occ = occ.reset_index()
occ = occ[occ["n_mode_win"] >= occ["phase"].map(MIN_WIN_PHASE).fillna(20)]
occ.to_csv(f"{OUT3}/03_mode_occupancy_animal_phase.csv", index=False)

def occ_box(d, order_, pairs, st, title, fname):
    fig, axes = plt.subplots(1, GM_K, figsize=(3.6 * GM_K, 3.6), sharey=True)
    for ax, m in zip(axes, MODES):
        sns.boxplot(data=d, x="group", y=m, order=order_, hue="group", hue_order=order_, palette=GROUP_COL,
                    showfliers=False, width=0.55, boxprops=dict(alpha=0.45), legend=False, ax=ax)
        sns.swarmplot(data=d, x="group", y=m, order=order_, hue="group", hue_order=order_, palette=GROUP_COL,
                      size=3.2, edgecolor="k", linewidth=0.3, legend=False, ax=ax)
        ax.set_ylim(-0.02, 1.25); ax.set_title(m, color=MODE_COL[MODES.index(m)], fontsize=10)
        ax.set_xlabel(""); ax.set_ylabel("fraction of time" if m == "M0" else "")
        ax.set_xticks(range(len(order_)))
        ax.set_xticklabels([f"{g}\nn={int((d['group'] == g).sum())}" for g in order_], fontsize=8)
        for k_, (a, b) in enumerate(pairs):
            s = st[(st["A"] == a) & (st["B"] == b) & (st["feature"] == m)]
            if s.empty:
                continue
            i, j, y = order_.index(a), order_.index(b), 1.05 + 0.08 * k_
            q = float(s["q_BH"].iloc[0])
            ax.plot([i, i, j, j], [y - 0.02, y, y, y - 0.02], color="k", lw=0.8)
            ax.text((i + j) / 2, y, stars(q), ha="center", va="bottom", fontsize=8, color="k" if q < 0.05 else "0.5")
    fig.suptitle(f"{title} — mode occupancy per animal; Mann-Whitney, BH across modes per pair", fontsize=10)
    plt.tight_layout()
    save_fig(fig, FIG3, fname, dpi=200)

all_st = []
for i, (tag, ph_of, order_, pairs) in enumerate(SETS):
    d = occ[occ.apply(lambda r: ph_of.get(r["group"]) == r["phase"], axis=1)]
    st = compare_cols(d, pairs, MODES, tag)
    all_st.append(st)
    occ_box(d, order_, pairs, st, tag.replace("_", " "), f"03{chr(98 + i)}_occupancy_{tag}")
    print(f"\n{tag}:"); print(st[["A", "B", "feature", "mean_A", "mean_B", "r_rb_B_higher", "p", "q_BH"]].round(3).to_string(index=False))
d = occ[(occ["phase"] == "free") & occ["group"].isin(["C?", "O?"]) & ~occ["Experiment_ID"].isin(SUSPECT_HOUSED)]
all_st.append(compare_cols(d, [("C?", "O?")], MODES, "free_B2_without_suspected_housed"))
pd.concat(all_st, ignore_index=True).to_csv(f"{OUT3}/03_stats_mode_occupancy.csv", index=False)

# density panels with mode ellipses
def ellipse(ax, mu, cov, color):
    w_, v_ = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 100)
    e = (v_ @ np.diag(2 * np.sqrt(w_)) @ np.vstack([np.cos(t), np.sin(t)])).T + mu
    ax.plot(e[:, 0], e[:, 1], color=color, lw=1.3)

panels = [("C-", "free"), ("O-", "free"), ("C?", "free"), ("O?", "free"), ("C+", "pre"), ("O+", "pre"),
          ("C+", "HE"), ("O+", "HE"), ("C+", "in_house"), ("O+", "in_house"), ("Ch", "housed"), ("Oh", "housed")]
fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
for ax, (g, ph) in zip(axes.ravel(), panels):
    dd = use[(use["group"] == g) & (use["phase"] == ph)]
    if len(dd):
        wgt = 1.0 / dd.groupby("Experiment_ID")["Experiment_ID"].transform("size")
        ax.hexbin(dd["ls"], dd["lv"], C=wgt, reduce_C_function=np.sum, gridsize=45,
                  extent=(-3, 1, 0, 3.7), cmap="Greys", bins="log", mincnt=1)
    for k, old in enumerate(order):
        ellipse(ax, gm.means_[old], gm.covariances_[old], MODE_COL[k])
        ax.text(*gm.means_[old], MODES[k], color=MODE_COL[k], fontsize=10, weight="bold")
    ax.set_title(f"{g} · {ph} (n={dd['Experiment_ID'].nunique()})", fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("log10 speed (Lref/s)")
for ax in axes[:, 0]:
    ax.set_ylabel("log10 tail-base velocity (deg/s)")
plt.tight_layout()
save_fig(fig, FIG3, "03a_mode_density_panels", dpi=200)

# =============================================================================
# 4. choice of K (control windows): BIC + stability across animal bootstraps
# =============================================================================
SEL_K, SEL_BOOT, SEL_FOLDS, SEL_SEED = list(range(2, 9)), 10, 5, 0
ctrl = win[valid & (win["cond"] == "control")].dropna(subset=["ls", "lv"])
ctrl = ctrl.groupby("Experiment_ID", group_keys=False).apply(lambda g: g.sample(min(len(g), GM_N_PER), random_state=SEL_SEED)).reset_index(drop=True)
X = StandardScaler().fit_transform(ctrl[["ls", "lv"]].to_numpy())
grp, ph = ctrl["Experiment_ID"].to_numpy(), ctrl["phase"].to_numpy()
animals = np.unique(grp)
sel = []
for K in SEL_K:
    g0 = GaussianMixture(K, covariance_type="full", n_init=3, random_state=SEL_SEED).fit(X)
    lab0 = g0.predict(X)
    ll = [GaussianMixture(K, covariance_type="full", n_init=2, random_state=SEL_SEED).fit(X[tr]).score(X[te])
          for tr, te in GroupKFold(SEL_FOLDS).split(X, groups=grp)]
    rng = np.random.default_rng(1)
    ari = []
    for b in range(SEL_BOOT):
        idx = np.concatenate([np.flatnonzero(grp == a) for a in rng.choice(animals, len(animals), replace=True)])
        ari.append(adjusted_rand_score(lab0, GaussianMixture(K, covariance_type="full", n_init=2, random_state=b).fit(X[idx]).predict(X)))
    sel.append(dict(K=K, BIC=g0.bic(X), heldout_ll=np.mean(ll), stability_ARI=np.mean(ari), stability_sd=np.std(ari),
                    NMI_with_phase=normalized_mutual_info_score(ph, lab0)))
    print(f"  K={K}  held-out LL={sel[-1]['heldout_ll']:.3f}  stability ARI={np.mean(ari):.2f}±{np.std(ari):.2f}")
sel = pd.DataFrame(sel)
sel.to_csv(f"{OUT3}/03_select_K_and_features.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
axes[0].plot(sel["K"], sel["BIC"], "o-", color="k"); axes[0].set_ylabel("BIC (lower = better)"); axes[0].set_title("model fit", fontsize=10)
axes[1].errorbar(sel["K"], sel["stability_ARI"], yerr=sel["stability_sd"], marker="o", capsize=3, color="k")
axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("stability across animal bootstraps (ARI)"); axes[1].set_title("reproducibility", fontsize=10)
for ax in axes:
    ax.axvline(GM_K, color=MODE_COL[3], ls=":", lw=1.2); ax.set_xlabel("number of states K"); ax.set_xticks(SEL_K)
fig.suptitle("Choice of K: log speed × log tail-base velocity, control windows (dotted = K used)", fontsize=10)
plt.tight_layout()
save_fig(fig, FIG3, "03_select_K_bic_stability", dpi=200)

# =============================================================================
# 5. decode all windows (2D, then speed-only for fast swimming) + smoothing
# =============================================================================
SPEED_ONLY_P = 0.80        # min posterior to assign a mode from speed alone
GAP_BRIDGE = 2             # windows (1 s): bridge a gap if the same mode is on both sides
MIN_BOUT = 3               # windows (1.5 s): shorter runs are absorbed by the longer neighbour
MODE_RGBA = np.array([to_rgba(c) for c in MODE_COL])
mode_patches = [mpatches.Patch(color=MODE_COL[k], label=MODES[k]) for k in range(GM_K)] + \
               [mpatches.Patch(facecolor="white", edgecolor="0.6", label="unknown")]

full = win["ls"].notna() & win["lv"].notna()
win["mode_raw"], win["mode_src"] = np.nan, "none"
win.loc[full, "mode_raw"] = pd.Series(gm.predict(win.loc[full, ["ls", "lv"]].to_numpy()), index=win.index[full]).map(relab)
win.loc[full, "mode_src"] = "2D"

def speed_only(ls_vals):
    """Posterior over modes given speed only (tail dimension integrated out). Returns (mode, confidence)."""
    lik = gm.weights_ * norm.pdf(ls_vals[:, None], gm.means_[:, 0], np.sqrt(gm.covariances_[:, 0, 0]))
    post = lik / lik.sum(axis=1, keepdims=True)
    return np.array([relab[k] for k in post.argmax(1)]), post.max(1)

m_chk, c_chk = speed_only(win.loc[full, "ls"].to_numpy())
agree = c_chk >= SPEED_ONLY_P
print(f"\nspeed-only decoding confident in {agree.mean():.1%} of 2D windows; matches the 2D mode in "
      f"{(m_chk[agree] == win.loc[full, 'mode_raw'].to_numpy()[agree]).mean():.1%}")
only_s = win["ls"].notna() & ~full
m_s, c_s = speed_only(win.loc[only_s, "ls"].to_numpy())
ok_s = (c_s >= SPEED_ONLY_P) & (m_s == 3)          # speed alone is reliable only for fast swimming
idx = win.index[only_s][ok_s]
win.loc[idx, "mode_raw"], win.loc[idx, "mode_src"] = m_s[ok_s], "speed-only"

def debounce(seq, min_len):
    seq = seq.copy()
    while True:
        b = np.flatnonzero(np.diff(seq)) + 1
        st, en = np.r_[0, b], np.r_[b, len(seq)]
        if len(st) == 1:
            return seq
        L = en - st
        i = int(np.argmin(L))
        if L[i] >= min_len:
            return seq
        t = 1 if i == 0 else i - 1 if i == len(st) - 1 else (i - 1 if L[i - 1] >= L[i + 1] else i + 1)
        seq[st[i]:en[i]] = seq[st[t]]

def smooth(m):
    m = m.copy()
    na = np.isnan(m)
    rid = np.cumsum(np.r_[True, na[1:] != na[:-1]])
    for r in np.unique(rid[na]):                                   # bridge short gaps
        ix = np.flatnonzero(rid == r)
        a, b = ix[0] - 1, ix[-1] + 1
        if len(ix) <= GAP_BRIDGE and a >= 0 and b < len(m) and m[a] == m[b]:
            m[ix] = m[a]
    out, known = m.copy(), ~np.isnan(m)
    rid = np.cumsum(np.r_[True, known[1:] != known[:-1]])
    for r in np.unique(rid[known]):                                # debounce inside known stretches
        ix = np.flatnonzero(rid == r)
        if len(ix) >= MIN_BOUT:
            out[ix] = debounce(m[ix].astype(int), MIN_BOUT)
    return out

win = win.sort_values(["Experiment_ID", "WindowStart"]).reset_index(drop=True)
win["mode"] = np.nan
for e, g in win.groupby("Experiment_ID", sort=False):
    win.loc[g.index, "mode"] = smooth(g["mode_raw"].to_numpy(float))
print(f"windows with a mode: raw {win['mode_raw'].notna().mean():.1%} → decoded + smoothed {win['mode'].notna().mean():.1%}")
win.drop(columns=["ls", "lv"]).to_csv(f"{OUT3b}/03b_windows_long_decoded.csv", index=False)

# =============================================================================
# 6. bouts + time budgets
# =============================================================================
bout_rows = []
for e, g in win.groupby("Experiment_ID", sort=False):
    m, t, ph = g["mode"].to_numpy(), g["t_s"].to_numpy(), g["phase"].to_numpy()
    n = len(m)
    brk = np.r_[True, ~((m[1:] == m[:-1]) & ~np.isnan(m[1:]))]
    st = np.flatnonzero(brk); en = np.r_[st[1:], n] - 1
    for i0, i1 in zip(st, en):
        if np.isnan(m[i0]):
            continue
        prev = m[i0 - 1] if i0 > 0 else np.nan
        nxt = m[i1 + 1] if i1 < n - 1 else np.nan
        bout_rows.append(dict(Experiment_ID=e, mode=int(m[i0]), start_s=t[i0] - WIN / FPS / 2,
                              dur_s=(i1 - i0 + 1) * DT + (WIN - STRIDE) / FPS,
                              phase=pd.Series(ph[i0:i1 + 1]).mode().iloc[0],
                              censored=bool(np.isnan(prev) or np.isnan(nxt)), next_mode=nxt))
bouts = pd.DataFrame(bout_rows).merge(meta2[["Experiment_ID", "group", "batch", "cond", "exclude_reason"]], on="Experiment_ID")
bouts.to_csv(f"{OUT3b}/03b_bouts.csv", index=False)

bb = bouts[(bouts["exclude_reason"].fillna("") == "") & (bouts["group"] != "R")]
use = win[(win["exclude_reason"].fillna("") == "") & (win["group"] != "R")]
gk = ["Experiment_ID", "group", "batch", "phase"]
bud = use.dropna(subset=["mode"]).groupby(gk)["mode"].value_counts(normalize=True).unstack(fill_value=0)
bud.columns = MODES
bud["coverage"] = use.groupby(gk)["mode"].apply(lambda s: s.notna().mean())
bud["n_known"] = use.groupby(gk)["mode"].count()
bud = bud.reset_index()
bud = bud[bud["n_known"] >= bud["phase"].map(MIN_WIN_PHASE).fillna(20)]
for k, mname in enumerate(MODES):
    sub = bb[bb["mode"] == k]
    rate = sub.groupby(gk).size().rename(f"rate_{mname}")
    dur = sub[~sub["censored"]].groupby(gk)["dur_s"].median().rename(f"dur_{mname}")
    bud = bud.merge(rate.reset_index(), on=gk, how="left").merge(dur.reset_index(), on=gk, how="left")
    bud[f"rate_{mname}"] = bud[f"rate_{mname}"].fillna(0) / (bud["n_known"] * DT / 60)   # bouts / min of known time
bud.to_csv(f"{OUT3b}/03b_time_budget_bouts_animal_phase.csv", index=False)

def subset(df, ph_of):
    return df[df.apply(lambda r: ph_of.get(r["group"]) == r["phase"], axis=1)]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), gridspec_kw=dict(width_ratios=[4, 2, 4, 8]))
for ax, (tag, ph_of, order_, _) in zip(axes[:3], SETS):
    d = subset(bud, ph_of)
    means = d.groupby("group")[MODES].mean().reindex(order_)
    ns = d.groupby("group").size().reindex(order_).fillna(0).astype(int)
    bottom = np.zeros(len(order_))
    for k, mname in enumerate(MODES):
        ax.bar(range(len(order_)), means[mname].fillna(0), bottom=bottom, color=MODE_COL[k], edgecolor="white", width=0.7)
        bottom += means[mname].fillna(0).to_numpy()
    ax.set_xticks(range(len(order_))); ax.set_xticklabels([f"{g}\nn={n}" for g, n in zip(order_, ns)], fontsize=8)
    ax.set_ylim(0, 1); ax.set_title(tag.replace("_", " "), fontsize=10)
axes[0].set_ylabel("mean fraction of time")
ax = axes[3]
xpos, labels = 0, []
for g in ["C+", "O+"]:
    for ph in ["pre", "HE", "in_house", "after_escape"]:
        d = bud[(bud["group"] == g) & (bud["phase"] == ph)]
        if d.empty:
            continue
        bottom = 0
        for k, mname in enumerate(MODES):
            ax.bar(xpos, d[mname].mean(), bottom=bottom, color=MODE_COL[k], edgecolor="white", width=0.7)
            bottom += d[mname].mean()
        labels.append((xpos, f"{g}\n{ph}\nn={len(d)}"))
        xpos += 1
    xpos += 0.6
ax.set_xticks([p for p, _ in labels]); ax.set_xticklabels([l for _, l in labels], fontsize=7)
ax.set_ylim(0, 1); ax.set_title("builders by phase", fontsize=10)
ax.legend(handles=mode_patches[:GM_K], fontsize=8, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
save_fig(fig, FIGB, "03b_time_budget", dpi=200)

# bout rate + duration (free behaviour) with stats
def box_cols(d, order_, pairs, st, cols, titles, ylabel, fname, logy=False):
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.6))
    for ax, c, tt in zip(axes, cols, titles):
        dd = d.dropna(subset=[c])
        sns.boxplot(data=dd, x="group", y=c, order=order_, hue="group", hue_order=order_, palette=GROUP_COL,
                    showfliers=False, width=0.55, boxprops=dict(alpha=0.45), legend=False, ax=ax)
        sns.swarmplot(data=dd, x="group", y=c, order=order_, hue="group", hue_order=order_, palette=GROUP_COL,
                      size=3.2, edgecolor="k", linewidth=0.3, legend=False, ax=ax)
        if logy and (dd[c] > 0).all() and len(dd):
            ax.set_yscale("log")
        ax.set_title(tt, fontsize=9); ax.set_xlabel(""); ax.set_ylabel(ylabel if c == cols[0] else "")
        ax.set_xticks(range(len(order_)))
        ax.set_xticklabels([f"{g}\nn={int((dd['group'] == g).sum())}" for g in order_], fontsize=8)
        ytop = ax.get_ylim()[1]
        for k_, (a, b) in enumerate(pairs):
            s = st[(st["A"] == a) & (st["B"] == b) & (st["feature"] == c)] if len(st) else st
            if s.empty:
                continue
            i, j = order_.index(a), order_.index(b)
            y = ytop * (1.03 + 0.08 * k_) if ax.get_yscale() == "linear" else ytop * 1.2 ** (k_ + 1)
            q = float(s["q_BH"].iloc[0])
            ax.plot([i, i, j, j], [y * 0.98, y, y, y * 0.98], color="k", lw=0.8)
            ax.text((i + j) / 2, y, stars(q), ha="center", va="bottom", fontsize=8, color="k" if q < 0.05 else "0.5")
    plt.tight_layout()
    save_fig(fig, FIGB, fname, dpi=200)

free = subset(bud, SETS[0][1])
for kind, ylabel, logy in [("rate", "bouts per min", False), ("dur", "median bout duration (s)", True)]:
    cols = [f"{kind}_{m}" for m in MODES]
    st = compare_cols(free, SETS[0][3], cols, f"free_{kind}")
    box_cols(free, SETS[0][2], SETS[0][3], st, cols, MODES, ylabel, f"03b_bout_{kind}_free", logy)
    st.to_csv(f"{OUT3b}/03b_stats_bout_{kind}_free.csv", index=False)

# =============================================================================
# 7. transition matrices (bout → next bout)
# =============================================================================
def trans_matrix(df):
    d = df[df["next_mode"].notna()]
    C = pd.crosstab(d["mode"], d["next_mode"].astype(int)).reindex(index=range(GM_K), columns=range(GM_K), fill_value=0)
    return C.div(C.sum(1).replace(0, np.nan), axis=0)

tm_groups = [("C-", "free"), ("O-", "free"), ("C?", "free"), ("O?", "free"), ("C+", "in_house"), ("O+", "in_house")]
fig, axes = plt.subplots(2, 4, figsize=(17, 8))
TM = {}
for ax, (g, ph) in zip(axes.ravel()[:6], tm_groups):
    d = bb[(bb["group"] == g) & (bb["phase"] == ph)]
    TM[(g, ph)] = trans_matrix(d)
    sns.heatmap(TM[(g, ph)], annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax, xticklabels=MODES, yticklabels=MODES)
    ax.set_title(f"{g} · {ph}  (n={d['Experiment_ID'].nunique()}, {int(d['next_mode'].notna().sum())} transitions)", fontsize=9)
    ax.set_xlabel("next"); ax.set_ylabel("from")
for ax, (a, b) in zip(axes.ravel()[6:], [(("O-", "free"), ("C-", "free")), (("O?", "free"), ("C?", "free"))]):
    D = TM[a] - TM[b]
    v = np.nanmax(np.abs(D.to_numpy())) or 1
    sns.heatmap(D, annot=True, fmt="+.2f", cmap="RdBu_r", vmin=-v, vmax=v, cbar=False, ax=ax, xticklabels=MODES, yticklabels=MODES)
    ax.set_title(f"{a[0]} − {b[0]} (free)", fontsize=9); ax.set_xlabel("next"); ax.set_ylabel("from")
plt.tight_layout()
save_fig(fig, FIGB, "03b_transitions", dpi=200)

# =============================================================================
# 8. raster overview + ethograms (batch B1: each animal recorded once)
# =============================================================================
ETHO_BATCH = "B1"
ETHO_FEATS = [("speed_med", "Speed\n(Lref/s)", True), ("base_vel", "Tail-base vel.\n(deg/s)", True),
              ("junc_abs", "Junction\nangle (deg)", False), ("omega_mean", "Body rotation\n(deg/s)", True)]
ETHO_CATS = {"control · builder": ["C+"], "control · non-builder": ["C-"], "6-OHDA · builder": ["O+"], "6-OHDA · non-builder": ["O-"]}
CAT_OF = {g: c for c, gs in ETHO_CATS.items() for g in gs}
etho_meta = meta2[(meta2["batch"] == ETHO_BATCH) & (meta2["exclude_reason"].fillna("") == "") & meta2["group"].isin(list(CAT_OF))].copy()
etho_meta["category"] = etho_meta["group"].map(CAT_OF)

gsize = [int((etho_meta["category"] == c).sum()) for c in ETHO_CATS]
ncol = int(win["WindowStart"].max() // STRIDE) + 1
fig, axes = plt.subplots(len(ETHO_CATS), 1, figsize=(12, 0.16 * sum(gsize) + 3), sharex=True,
                         gridspec_kw=dict(height_ratios=[max(n, 1) for n in gsize]))
for ax, cat in zip(axes, ETHO_CATS):
    ids = etho_meta[etho_meta["category"] == cat].copy()
    ids["sortkey"] = ids["he_start"].fillna(0) if cat.endswith("· builder") else -ids["win_kin_ok"]
    ids = ids.sort_values("sortkey")["Experiment_ID"].tolist()
    M = np.ones((len(ids), ncol, 4))
    for r_, e in enumerate(ids):
        d = use[use["Experiment_ID"] == e]
        ok = d["mode"].notna().to_numpy()
        M[r_, (d["WindowStart"].to_numpy()[ok] // STRIDE).astype(int)] = MODE_RGBA[d["mode"].to_numpy()[ok].astype(int)]
    ax.imshow(M, aspect="auto", interpolation="nearest", extent=[0, ncol * DT / 60, len(ids), 0])
    for r_, e in enumerate(ids):
        rr = meta2[meta2["Experiment_ID"] == e].iloc[0]
        if pd.notna(rr["he_start"]):
            ax.plot([rr["he_start"] / FPS / 60] * 2, [r_, r_ + 1], color="gold", lw=2)
        if pd.notna(rr["escape"]):
            ax.plot([rr["escape"] / FPS / 60] * 2, [r_, r_ + 1], color="red", lw=1.5)
    ax.set_yticks([])
    ax.set_ylabel(f"{cat}\nn={len(ids)}", rotation=0, ha="right", va="center", fontsize=9,
                  color=COND_COL["control" if cat.startswith("control") else "6-OHDA"])
axes[-1].set_xlabel("time (min) — gold: expansion onset, red: escape, white: unknown")
axes[0].legend(handles=mode_patches, ncol=5, fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02))
plt.tight_layout()
save_fig(fig, FIGB, f"03b_raster_overview_{ETHO_BATCH}", dpi=200)

def ethogram(e):
    r = meta2[meta2["Experiment_ID"] == e].iloc[0]
    g = win[win["Experiment_ID"] == e].sort_values("WindowStart")
    t = g["t_s"].to_numpy() / 60
    edges = np.r_[t - DT / 120, t[-1] + DT / 120]
    m = g["mode"].to_numpy()
    ras = np.ones((len(m), 4)); ok = ~np.isnan(m)
    ras[ok] = MODE_RGBA[m[ok].astype(int)]
    bg = ras.copy(); bg[:, 3] = np.where(ok, 0.15, 0)
    fig, axes = plt.subplots(1 + len(ETHO_FEATS), 1, figsize=(11, 1.2 + 1.15 * len(ETHO_FEATS)), sharex=True,
                             gridspec_kw=dict(height_ratios=[0.4] + [1] * len(ETHO_FEATS)))
    axes[0].pcolormesh(edges, [0, 1], ras[None], shading="flat", rasterized=True)
    axes[0].set_yticks([])
    for ax, (f, lab, logy) in zip(axes[1:], ETHO_FEATS):
        ax.pcolormesh(edges, [0, 1], bg[None], shading="flat", rasterized=True, transform=ax.get_xaxis_transform())
        y = g[f].to_numpy(float)
        if logy:
            y = np.where(y > 0, y, np.nan); ax.set_yscale("log")
        ax.plot(t, y, color="0.15", lw=0.5)
        ax.set_ylabel(lab, fontsize=7); ax.tick_params(labelsize=7)
    for ax in axes:
        if pd.notna(r["he_start"]):
            ax.axvspan(r["he_start"] / FPS / 60, r["he_end"] / FPS / 60, facecolor="none", edgecolor="gold", lw=1.6)
            ine = r["in_end"] if pd.notna(r["in_end"]) else g["WindowStart"].max() + WIN
            if pd.notna(r["in_start"]):
                ax.axvspan(r["in_start"] / FPS / 60, ine / FPS / 60, color="gold", alpha=0.06, lw=0)
        if pd.notna(r["escape"]):
            ax.axvline(r["escape"] / FPS / 60, color="red", lw=1, ls="--")
    axes[0].set_title(f"{e}  ·  {CAT_OF[r['group']]}  ·  {r['batch']}  ·  state coverage {ok.mean():.0%}", fontsize=8, loc="left")
    fig.legend(handles=mode_patches, ncol=5, fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(0.98, 1.0))
    axes[-1].set_xlabel("time (min) — gold box: expansion, gold shade: inside house, red: escape", fontsize=8)
    cat_dir = CAT_OF[r["group"]].replace(" · ", "_").replace("-", "")
    save_fig(fig, f"{ETHO}/{ETHO_BATCH}_{cat_dir}", f"{cat_dir}_{e}", dpi=160)

for i, e in enumerate(etho_meta["Experiment_ID"]):
    ethogram(e)
print(f"\nsaved → {OUT3} and {OUT3b}")
