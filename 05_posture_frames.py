#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05 — Frame-level posture around the decision point (expansion onsets and still-bout ends),
eigen-tail PCA on control frames (input for the embedding), skeleton panels per annotated phase.

Outputs:
    OUT/05_posture/   05a_events.csv, 05a_frames.parquet, 05a_qc_frames.csv, 05_eigentail_pca.pkl, figures/
    OUT/fig_panels/   P_skeletons_annotated_phases, P_percentile_skeletons (+ curvature csv)
"""
import pickle
from sklearn.decomposition import PCA
from common import *

OUT5, FIG5, FIGP = f"{OUT}/05_posture", f"{OUT}/05_posture/figures", f"{OUT}/fig_panels"
for d in (FIG5, FIGP):
    os.makedirs(d, exist_ok=True)

PRE_S, POST_S = 10.0, 10.0             # s before the event / after its end
NB_EVENTS_MAX = 10                     # still-bout ends per non-builder (evenly spread)
MAIN_BATCHES = {"B1", "B2"}

meta2 = load_meta2()
win = load_windows()
okm = valid_animals(meta2)
by_id = {e: g.reset_index(drop=True) for e, g in win[win["Experiment_ID"].isin(okm["Experiment_ID"])].groupby("Experiment_ID")}
lref_of = meta2.set_index("Experiment_ID")["Lref_px"]
abn = load_abnormal()

# =============================================================================
# 1. events: expansion onsets, non-converted still-bout ends (builders, well before onset), still-bout ends (non-builders)
# =============================================================================
ev_rows = []
for _, r in okm.iterrows():
    e, g_ = r["Experiment_ID"], r["group"]
    if e in SUSPECT_HOUSED or g_ in ("Ch", "Oh"):
        continue
    g = by_id[e]
    base = dict(Experiment_ID=e, group=g_, batch=r["batch"], cond=r["cond"])
    if g_ in ("C+", "O+"):
        t_on, t_end = r["he_start"] / FPS, r["he_end"] / FPS
        ev_rows.append(dict(base, kind="expansion", t0=t_on, t1=t_end))
        gp = g[g["phase"] == "pre"].reset_index(drop=True)
        for i, j, d in (still_runs(gp) if len(gp) else []):
            t = gp["t_s"].iloc[j] + WIN / FPS / 2
            if PRE_S <= t <= t_on - POST_S - 5:
                ev_rows.append(dict(base, kind="still_end", t0=t, t1=t))
    elif g_ in ("C-", "O-", "C?", "O?"):
        gp = g[g["phase"] == "free"].reset_index(drop=True)
        ends = [gp["t_s"].iloc[j] + WIN / FPS / 2 for i, j, d in still_runs(gp)]
        ends = [t for t in ends if PRE_S <= t <= gp["t_s"].max() - POST_S]
        if len(ends) > NB_EVENTS_MAX:
            ends = list(np.asarray(ends)[np.linspace(0, len(ends) - 1, NB_EVENTS_MAX).round().astype(int)])
        ev_rows += [dict(base, kind="still_end", t0=t, t1=t) for t in ends]
events = pd.DataFrame(ev_rows)
events["event_id"] = events["Experiment_ID"] + "_" + events.groupby("Experiment_ID").cumcount().astype(str)
events.to_csv(f"{OUT5}/05a_events.csv", index=False)
print(events.groupby(["group", "kind"]).agg(animals=("Experiment_ID", "nunique"), events=("event_id", "size")).to_string())

# =============================================================================
# 2. frame posture per event
# =============================================================================
parts = []
for k, (e, evs) in enumerate(events.groupby("Experiment_ID", sort=False)):
    r = okm[okm["Experiment_ID"] == e].iloc[0]
    P, info = clean_frames(r["path"])
    A, speed, omega = frame_posture(P, lref_of[e])
    T = A.shape[0]
    mode_at = dict(zip(by_id[e]["WindowStart"].to_numpy(), by_id[e]["mode"].to_numpy()))
    for _, ev in evs.iterrows():
        f0, f1 = max(0, int(round((ev["t0"] - PRE_S) * FPS))), min(T - 1, int(round((ev["t1"] + POST_S) * FPS)))
        f = np.arange(f0, f1 + 1)
        t = f / FPS
        ws = np.clip(np.round((f - WIN // 2) / STRIDE) * STRIDE, 0, None).astype(int)
        d = pd.DataFrame(A[f], columns=SEGS)
        d.insert(0, "frame", f)
        d.insert(0, "seg", np.where(t < ev["t0"], "pre", np.where(t > ev["t1"], "post", "HE")))
        for c in ("event_id", "kind", "cond", "batch", "group", "Experiment_ID"):
            d.insert(0, c, ev[c])
        d["t_rel"], d["t_rel_end"] = t - ev["t0"], t - ev["t1"]
        d["speed"], d["omega"] = speed[f], omega[f]
        d["mode"] = [mode_at.get(x, np.nan) for x in ws]
        parts.append(d)
    if (k + 1) % 20 == 0:
        print(f"  frames extracted: {k + 1}/{events['Experiment_ID'].nunique()} animals")
fr = pd.concat(parts, ignore_index=True)

# chirality: one mirror per animal so that the median junction angle is positive
flip = fr.groupby("Experiment_ID")["a0"].median().apply(lambda v: -1.0 if v < 0 else 1.0)
fr["flip"] = fr["Experiment_ID"].map(flip)
fr[SEGS] = fr[SEGS].mul(fr["flip"], axis=0)
fr["prox_ok"] = fr[PROX].notna().all(axis=1) & fr["speed"].notna()
fr["full_ok"] = fr[SEGS].notna().all(axis=1) & fr["speed"].notna()
fr["main_set"] = fr["batch"].isin(MAIN_BATCHES)
fr.to_parquet(f"{OUT5}/05a_frames.parquet", index=False)
print(f"\nsaved {len(fr):,} frames; animals mirrored: {int((flip < 0).sum())}/{len(flip)}")

qcf = (fr.groupby(["batch", "group", "kind", "seg"])
         .agg(animals=("Experiment_ID", "nunique"), frames=("frame", "size"), prox_ok=("prox_ok", "mean"), full_ok=("full_ok", "mean")).round(2))
qcf.to_csv(f"{OUT5}/05a_qc_frames.csv")

q = fr[fr["main_set"]].groupby(["group", "kind", "seg"])[["prox_ok", "full_ok"]].mean().reset_index()
q["label"] = q["group"] + "\n" + q["kind"].str.replace("still_end", "still end").str.replace("expansion", "exp.") + "\n" + q["seg"]
fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(q)), 3.8))
x = np.arange(len(q))
ax.bar(x - 0.2, q["prox_ok"], 0.4, color="0.55", label="proximal chain + trunk")
ax.bar(x + 0.2, q["full_ok"], 0.4, color="0.2", label="full chain")
ax.set_xticks(x); ax.set_xticklabels(q["label"], fontsize=6)
ax.set_ylim(0, 1); ax.set_ylabel("fraction of frames with clean skeleton"); ax.legend(fontsize=8, frameon=False)
plt.tight_layout()
save_fig(fig, FIG5, "05a_qc_skeleton_availability", dpi=200)

# =============================================================================
# 3. eigen-tail PCA: proximal angles a0–a3 of control frames (B1 + B2) → used by step 06
# =============================================================================
ctrl_post = fr[(fr["cond"] == "control") & fr["main_set"] & fr["prox_ok"]][PROX]
pca_t = PCA(n_components=len(PROX), random_state=0).fit(ctrl_post.to_numpy())
print("eigen-tail variance explained:", np.round(pca_t.explained_variance_ratio_, 3))
pd.DataFrame(pca_t.components_.T, index=PROX, columns=[f"pc{k + 1}" for k in range(len(PROX))]).to_csv(
    f"{OUT5}/05_eigentail_loadings.csv")
with open(f"{OUT5}/05_eigentail_pca.pkl", "wb") as fh:
    pickle.dump(pca_t, fh)

# =============================================================================
# FIG panels — skeletons per annotated phase (raw overlays + percentile skeletons)
# =============================================================================
N_SKEL, ALPHA_SKEL = 120, 0.08
N_RES, PER_ANIMAL, BAND = 25, 30, 5
PCTS = (10, 50, 90)
STYLE = {10: dict(ls="--", lw=1.2, alpha=0.9), 50: dict(ls="-", lw=2.0, alpha=1.0), 90: dict(ls=":", lw=1.4, alpha=0.9)}
PHASES = ["before (−5–0 s)", "expansion · start (0–3 s)", "expansion · middle", "expansion · end (last 3 s)",
          "in house (0–10 s)", "abnormal start", "failed insertion"]

def phase_frames(r):
    hs, he = int(r["he_start"]), int(r["he_end"])
    d = {"before (−5–0 s)": np.arange(max(0, hs - int(5 * FPS)), hs),
         "expansion · start (0–3 s)": np.arange(hs, min(he, hs + int(3 * FPS))),
         "expansion · middle": np.arange(hs + (he - hs) // 3, hs + 2 * (he - hs) // 3),
         "expansion · end (last 3 s)": np.arange(max(hs, he - int(3 * FPS)), he)}
    if pd.notna(r["in_start"]):
        d["in house (0–10 s)"] = np.arange(int(r["in_start"]), int(r["in_start"]) + int(10 * FPS))
    a = abn[abn["Experiment_ID"] == r["Experiment_ID"]]
    abn_frames = np.array([], int)
    for lab, pat in (("abnormal start", "abnormal"), ("failed insertion", "failed")):
        s = a[a["event"].astype(str).str.contains(pat, case=False)]
        if len(s):
            fr_ = np.concatenate([np.arange(int(x.start), int(x.end) + 1) for x in s.itertuples()])
            d[lab] = fr_
            abn_frames = np.union1d(abn_frames, fr_)
    for k in list(d):                                  # regular phases free of annotated abnormal frames
        if k not in ("abnormal start", "failed insertion"):
            d[k] = np.setdiff1d(d[k], abn_frames)
    return d

SK = {}                                                # (cond, phase) -> list of (animal, P, frames, parts, lref)
bl = okm[okm["group"].isin(["C+", "O+"])]
for k, (_, r) in enumerate(bl.iterrows()):
    P, info = clean_frames(r["path"])
    T = len(P["bodypart1"])
    full_ok = np.isfinite(np.stack([P[b][:, 0] for b in TAIL_DRAW + TRUNK_DRAW[:4]])).all(0)
    prox_ok = np.isfinite(np.stack([P[b][:, 0] for b in PROX_DRAW + TRUNK_DRAW[:4]])).all(0)
    for ph, frames in phase_frames(r).items():
        frames = frames[(frames >= 0) & (frames < T)]
        good, tail_parts = frames[full_ok[frames]], TAIL_DRAW
        if len(good) < 10:                             # poor distal tracking: proximal tail only
            good, tail_parts = frames[prox_ok[frames]], PROX_DRAW
        if len(good):
            SK.setdefault((r["cond"], ph), []).append((r["Experiment_ID"], P, good, tail_parts, lref_of[r["Experiment_ID"]]))
    if (k + 1) % 10 == 0:
        print(f"  skeletons: {k + 1}/{len(bl)} builders")

def finish_axes(axes, legend=False):
    for ax in axes.ravel():
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axes[0, 0].set_xlim(-1.0, 2.8); axes[0, 0].set_ylim(-1.4, 2.4)
    if legend:
        axes[0, 0].legend(fontsize=5, frameon=False, loc="lower left")
    axes[-1, 0].plot([-0.9, -0.4], [-1.3, -1.3], color="k", lw=1)
    axes[-1, 0].text(-0.65, -1.25, "0.5 L", ha="center", va="bottom", fontsize=5)

# raw overlays
fig, axes = plt.subplots(2, len(PHASES), figsize=(2.3 * len(PHASES), 4.9), sharex=True, sharey=True)
for ri, c in enumerate(COND_COL):
    for ci, ph in enumerate(PHASES):
        ax = axes[ri, ci]
        items, drawn = SK.get((c, ph), []), 0
        if items:
            per = int(np.ceil(N_SKEL / len(items)))
            for e, P, good, parts_, lref in items:
                for f in good[np.linspace(0, len(good) - 1, min(per, len(good))).round().astype(int)]:
                    tr, tl = normalised(P, f, lref, 1.0, TRUNK_DRAW), normalised(P, f, lref, 1.0, parts_)
                    if tr is None or tl is None:
                        continue
                    ax.plot(tr[:, 0], tr[:, 1], color="0.7", lw=0.4, alpha=ALPHA_SKEL * 0.6)
                    ax.plot(tl[:, 0], tl[:, 1], color=COND_COL[c], lw=0.5, alpha=ALPHA_SKEL)
                    drawn += 1
                    if drawn >= N_SKEL:
                        break
                if drawn >= N_SKEL:
                    break
        ax.set_title(f"{ph}\n{len(items)} animals, {drawn} frames" if ri == 0 else f"{len(items)} animals, {drawn} frames", fontsize=6)
        if ci == 0:
            ax.set_ylabel(c, color=COND_COL[c], fontsize=8)
finish_axes(axes)
plt.tight_layout(w_pad=0.2, h_pad=0.4)
save_fig(fig, FIGP, "P_skeletons_annotated_phases")

# percentile skeletons (p10 / median / p90 of signed tail curvature; full tail only)
PC_ROWS, PC_SHAPES = [], {}
for (c, ph), items in SK.items():
    shapes, trunks, kap, kabs, an = [], [], [], [], []
    for e, P, good, parts_, lref in items:
        if parts_ != TAIL_DRAW:
            continue
        for f in good[np.linspace(0, len(good) - 1, min(PER_ANIMAL, len(good))).round().astype(int)]:
            tl, tr = normalised(P, f, lref, 1.0, TAIL_DRAW), normalised(P, f, lref, 1.0, TRUNK_DRAW)
            rs = resample(tl, N_RES) if tl is not None and tr is not None else None
            if rs is None:
                continue
            k_s, k_a = curvature(tl)
            shapes.append(rs); trunks.append(tr); kap.append(k_s); kabs.append(k_a); an.append(e)
    if len(shapes) < 20:
        continue
    shapes, kap, kabs = np.array(shapes), np.array(kap), np.array(kabs)
    reps = {}
    for p in PCTS:
        lo, hi = np.percentile(kap, [max(0, p - BAND), min(100, p + BAND)])
        reps[p] = shapes[(kap >= lo) & (kap <= hi)].mean(axis=0)
    PC_SHAPES[(c, ph)] = dict(reps=reps, trunk=np.median(np.array(trunks), axis=0), bg=shapes, n_an=len(set(an)))
    PC_ROWS.append(dict(cond=c, phase=ph, animals=len(set(an)), frames=len(shapes),
                        curv_p10=np.percentile(kap, 10), curv_median=np.median(kap), curv_p90=np.percentile(kap, 90),
                        abs_curv_p10=np.percentile(kabs, 10), abs_curv_median=np.median(kabs), abs_curv_p90=np.percentile(kabs, 90)))
PCT = pd.DataFrame(PC_ROWS)
PCT["phase"] = pd.Categorical(PCT["phase"], PHASES, ordered=True)
PCT.sort_values(["phase", "cond"]).to_csv(f"{FIGP}/P_percentile_skeletons_curvature.csv", index=False)

fig, axes = plt.subplots(2, len(PHASES), figsize=(2.3 * len(PHASES), 4.9), sharex=True, sharey=True)
for ri, c in enumerate(COND_COL):
    for ci, ph in enumerate(PHASES):
        ax = axes[ri, ci]
        S_ = PC_SHAPES.get((c, ph))
        if S_ is not None:
            for sh in S_["bg"][np.linspace(0, len(S_["bg"]) - 1, min(60, len(S_["bg"]))).astype(int)]:
                ax.plot(sh[:, 0], sh[:, 1], color=COND_COL[c], lw=0.4, alpha=0.1)
            ax.fill(S_["trunk"][:4, 0], S_["trunk"][:4, 1], color="0.8", lw=0)
            for p, sh in S_["reps"].items():
                ax.plot(sh[:, 0], sh[:, 1], color=COND_COL[c], **STYLE[p], label=f"p{p}" if p != 50 else "median")
            ttl = f"{S_['n_an']} animals"
        else:
            ttl = "(too few full-tail frames)"
        ax.set_title(f"{ph}\n{ttl}" if ri == 0 else ttl, fontsize=6)
        if ci == 0:
            ax.set_ylabel(c, color=COND_COL[c], fontsize=8)
finish_axes(axes, legend=True)
fig.suptitle("Tail shapes ranked by signed curvature: p10 (dashed), median (solid), p90 (dotted); faint = sampled frames", fontsize=8)
plt.tight_layout(w_pad=0.2, h_pad=0.4)
save_fig(fig, FIGP, "P_percentile_skeletons")
print(f"\nsaved → {OUT5} and {FIGP}")
